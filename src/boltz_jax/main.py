"""Main entry point for JAX-based Boltz protein structure prediction."""

import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import click
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints, train_state
from tqdm import tqdm

from boltz_jax.data import const
from boltz_jax.data.module.inference import BoltzInferenceDataModule
from boltz_jax.data.msa.mmseqs2 import run_mmseqs2
from boltz_jax.data.parse.a3m import parse_a3m
from boltz_jax.data.parse.csv import parse_csv
from boltz_jax.data.parse.fasta import parse_fasta
from boltz_jax.data.parse.yaml import parse_yaml
from boltz_jax.data.types import MSA, Manifest, Record
from boltz_jax.data.write.writer import BoltzWriter
from boltz_jax.model.model import Boltz1, Boltz1TrainState

CCD_URL = "https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl"
MODEL_URL = (
    "https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt"
)


@dataclass
class BoltzProcessedInput:
    """Processed input data."""

    manifest: Manifest
    targets_dir: Path
    msa_dir: Path


@dataclass
class BoltzDiffusionParams:
    """Diffusion process parameters."""

    gamma_0: float = 0.605
    gamma_min: float = 1.107
    noise_scale: float = 0.901
    rho: float = 8
    step_scale: float = 1.638
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    P_mean: float = -1.2
    P_std: float = 1.5
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


def download(cache: Path) -> None:
    """Download all the required data.

    Parameters
    ----------
    cache : Path
        The cache directory.
    """
    # Download CCD
    ccd = cache / "ccd.pkl"
    if not ccd.exists():
        click.echo(
            f"Downloading the CCD dictionary to {ccd}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(CCD_URL, str(ccd))  # noqa: S310

    # Download model
    model = cache / "boltz1_conf.ckpt"
    if not model.exists():
        click.echo(
            f"Downloading the model weights to {model}. You may "
            "change the cache directory with the --cache flag."
        )
        urllib.request.urlretrieve(MODEL_URL, str(model))  # noqa: S310


def check_inputs(
    data: Path,
    outdir: Path,
    override: bool = False,
) -> list[Path]:
    """Check the input data and output directory.

    If the input data is a directory, it will be expanded
    to all files in this directory. Then, we check if there
    are any existing predictions and remove them from the
    list of input data, unless the override flag is set.

    Parameters
    ----------
    data : Path
        The input data.
    outdir : Path
        The output directory.
    override: bool
        Whether to override existing predictions.

    Returns
    -------
    list[Path]
        The list of input data.
    """
    click.echo("Checking input data.")

    # Check if data is a directory
    if data.is_dir():
        data: list[Path] = list(data.glob("*"))

        # Filter out non .fasta or .yaml files, raise
        # an error on directory and other file types
        filtered_data = []
        for d in data:
            if d.suffix in (".fa", ".fas", ".fasta", ".yml", ".yaml"):
                filtered_data.append(d)
            elif d.is_dir():
                msg = f"Found directory {d} instead of .fasta or .yaml."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {d.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

        data = filtered_data
    else:
        data = [data]

    # Check if existing predictions are found
    existing = (outdir / "predictions").rglob("*")
    existing = {e.name for e in existing if e.is_dir()}

    # Remove them from the input data
    if existing and not override:
        data = [d for d in data if d.stem not in existing]
        num_skipped = len(existing) - len(data)
        msg = (
            f"Found some existing predictions ({num_skipped}), "
            f"skipping and running only the missing ones, "
            "if any. If you wish to override these existing "
            "predictions, please set the --override flag."
        )
        click.echo(msg)
    elif existing and override:
        msg = "Found existing predictions, will override."
        click.echo(msg)

    return data


def compute_msa(
    data: dict[str, str],
    target_id: str,
    msa_dir: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
) -> None:
    """Compute the MSA for the input data.

    Parameters
    ----------
    data : dict[str, str]
        The input protein sequences.
    target_id : str
        The target id.
    msa_dir : Path
        The msa directory.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    """
    if len(data) > 1:
        paired_msas = run_mmseqs2(
            list(data.values()),
            msa_dir / f"{target_id}_paired_tmp",
            use_env=True,
            use_pairing=True,
            host_url=msa_server_url,
            pairing_strategy=msa_pairing_strategy,
        )
    else:
        paired_msas = [""] * len(data)

    unpaired_msa = run_mmseqs2(
        list(data.values()),
        msa_dir / f"{target_id}_unpaired_tmp",
        use_env=True,
        use_pairing=False,
        host_url=msa_server_url,
        pairing_strategy=msa_pairing_strategy,
    )

    for idx, name in enumerate(data):
        # Get paired sequences
        paired = paired_msas[idx].strip().splitlines()
        paired = paired[1::2]  # ignore headers
        paired = paired[: const.max_paired_seqs]

        # Set key per row and remove empty sequences
        keys = [idx for idx, s in enumerate(paired) if s != "-" * len(s)]
        paired = [s for s in paired if s != "-" * len(s)]

        # Combine paired-unpaired sequences
        unpaired = unpaired_msa[idx].strip().splitlines()
        unpaired = unpaired[1::2]
        unpaired = unpaired[: (const.max_msa_seqs - len(paired))]
        if paired:
            unpaired = unpaired[1:]  # ignore query is already present

        # Combine
        seqs = paired + unpaired
        keys = keys + [-1] * len(unpaired)

        # Dump MSA
        csv_str = ["key,sequence"] + [f"{key},{seq}" for key, seq in zip(keys, seqs)]

        msa_path = msa_dir / f"{name}.csv"
        with msa_path.open("w") as f:
            f.write("\n".join(csv_str))


def process_inputs(
    data: list[Path],
    out_dir: Path,
    ccd_path: Path,
    msa_server_url: str,
    msa_pairing_strategy: str,
    max_msa_seqs: int = 4096,
    use_msa_server: bool = False,
) -> None:
    """Process the input data and output directory.

    Parameters
    ----------
    data : list[Path]
        The list of input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The CCD dictionary path.
    msa_server_url : str
        The MSA server URL.
    msa_pairing_strategy : str
        The MSA pairing strategy.
    max_msa_seqs : int
        The maximum number of MSA sequences.
    use_msa_server : bool
        Whether to use the MSA server.
    """
    # Make temporary directories
    click.echo("Processing input data")
    msa_dir = out_dir / "msa"
    targets_dir = out_dir / "targets"
    msa_dir.mkdir(parents=True, exist_ok=True)
    targets_dir.mkdir(parents=True, exist_ok=True)

    # Process each input file
    records = []
    for input_file in data:
        target_id = input_file.stem
        click.echo(f"Processing {target_id}")

        # Parse the input file
        if input_file.suffix in (".yml", ".yaml"):
            record = parse_yaml(input_file, target_id)
        elif input_file.suffix in (".fa", ".fas", ".fasta"):
            record = parse_fasta(input_file, target_id)
        else:
            msg = (
                f"Unable to parse filetype {input_file.suffix}, "
                "please provide a .fasta or .yaml file."
            )
            raise RuntimeError(msg)

        # Compute MSA
        if record.msa is None:
            if use_msa_server:
                click.echo(f"Computing MSA for {target_id}")
                compute_msa(
                    record.data,
                    target_id,
                    msa_dir,
                    msa_server_url,
                    msa_pairing_strategy,
                )

                # Parse MSA
                record.msa = {}
                for chain in record.data:
                    msa_path = msa_dir / f"{chain}.csv"
                    record.msa[chain] = parse_csv(msa_path)
            else:
                # No MSA, create empty
                record.msa = {
                    chain: MSA(
                        keys=[],
                        sequences=[],
                    )
                    for chain in record.data
                }

        # Save the record
        records.append(record)

    # Create a manifest
    manifest = Manifest(
        records=records,
    )

    # Save the manifest
    manifest_path = out_dir / "manifest.pkl"
    with manifest_path.open("wb") as f:
        pickle.dump(manifest, f)

    # Return processed input object
    return BoltzProcessedInput(
        manifest=manifest,
        targets_dir=targets_dir,
        msa_dir=msa_dir,
    )


def init_model_and_state(
    rng: jnp.ndarray,
    config: Dict,
    checkpoint_path: Optional[str] = None,
) -> Tuple[Boltz1, Boltz1TrainState]:
    """Initialize the model and training state.
    
    Args:
        rng: Random number generator seed
        config: Model configuration
        checkpoint_path: Path to checkpoint to load
        
    Returns:
        Tuple of model and state
    """
    # Initialize model
    model = Boltz1(**config)
    
    # Create a dummy input to initialize parameters
    batch_size = 1
    seq_len = 64
    
    dummy_input = {
        "aatype": jnp.zeros((batch_size, seq_len), dtype=jnp.int32),
        "tokens": jnp.zeros((batch_size, seq_len, config["token_s"] + 2 * const.num_tokens + 1 + len(const.pocket_contact_info))),
        "msa_tokens": jnp.zeros((batch_size, 16, seq_len, config["token_s"] + 2 * const.num_tokens + 1 + len(const.pocket_contact_info))),
    }
    
    # Split PRNGKey for parameter initialization and dropout
    init_rng, dropout_rng = jax.random.split(rng)
    
    # Initialize parameters
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        feats=dummy_input,
        train=False,
    )
    
    # Create optimizer
    learning_rate = config.get("learning_rate", 1e-4)
    optimizer = optax.adamw(
        learning_rate=learning_rate,
        weight_decay=config.get("weight_decay", 0.01),
    )
    
    # Create training state
    state = Boltz1TrainState.create(
        apply_fn=model.apply,
        params=variables["params"],
        tx=optimizer,
    )
    
    # Load checkpoint if provided
    if checkpoint_path:
        state = checkpoints.restore_checkpoint(checkpoint_path, state)
        
        # Initialize EMA if using it
        if config.get("use_ema", False):
            state = state.initialize_ema()
    
    return model, state


def predict_structure(
    state: Boltz1TrainState,
    model: Boltz1,
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    config: Dict,
) -> Dict[str, jnp.ndarray]:
    """Perform structure prediction.
    
    Args:
        state: Training state with parameters
        model: Model instance
        batch: Input batch
        rng: RNG key for prediction
        config: Prediction configuration
        
    Returns:
        Dictionary of prediction results
    """
    # Extract prediction parameters
    recycling_steps = config.get("recycling_steps", 3)
    sampling_steps = config.get("sampling_steps", 200)
    diffusion_samples = config.get("diffusion_samples", 1)
    
    # Create PRNGKeys for different model components
    rng_keys = ["dropout", "msa_dropout", "pairformer_dropout", "confidence_dropout", "diffusion"]
    rngs = dict(zip(rng_keys, jax.random.split(rng, len(rng_keys))))
    
    # Use EMA parameters if available, otherwise use regular parameters
    params = state.ema_params if state.ema_params is not None else state.params
    
    # Perform prediction
    outputs = model.apply(
        {"params": params},
        feats=batch,
        recycling_steps=recycling_steps,
        num_sampling_steps=sampling_steps,
        diffusion_samples=diffusion_samples,
        train=False,
        rngs=rngs,
    )
    
    return outputs


@click.group()
def cli() -> None:
    """JAX implementation of the Boltz protein structure prediction."""
    pass


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the predictions.",
    default="./",
)
@click.option(
    "--cache",
    type=click.Path(exists=False),
    help="The directory where to download the data and model. Default is ~/.boltz_jax.",
    default="~/.boltz_jax",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=True),
    help="An optional checkpoint, will use the provided Boltz-1 model by default.",
    default=None,
)
@click.option(
    "--devices",
    type=int,
    help="The number of devices to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--accelerator",
    type=click.Choice(["gpu", "cpu", "tpu"]),
    help="The accelerator to use for prediction. Default is gpu.",
    default="gpu",
)
@click.option(
    "--recycling_steps",
    type=int,
    help="The number of recycling steps to use for prediction. Default is 3.",
    default=3,
)
@click.option(
    "--sampling_steps",
    type=int,
    help="The number of sampling steps to use for prediction. Default is 200.",
    default=200,
)
@click.option(
    "--diffusion_samples",
    type=int,
    help="The number of diffusion samples to use for prediction. Default is 1.",
    default=1,
)
@click.option(
    "--step_scale",
    type=float,
    help="The step size is related to the temperature at which the diffusion process samples the distribution."
    "The lower the higher the diversity among samples (recommended between 1 and 2). Default is 1.638.",
    default=1.638,
)
@click.option(
    "--write_full_pae",
    type=bool,
    is_flag=True,
    help="Whether to dump the pae into a npz file. Default is True.",
)
@click.option(
    "--write_full_pde",
    type=bool,
    is_flag=True,
    help="Whether to dump the pde into a npz file. Default is False.",
)
@click.option(
    "--output_format",
    type=click.Choice(["pdb", "mmcif"]),
    help="The output format to use for the predictions. Default is mmcif.",
    default="mmcif",
)
@click.option(
    "--num_workers",
    type=int,
    help="The number of dataloader workers to use for prediction. Default is 2.",
    default=2,
)
@click.option(
    "--override",
    is_flag=True,
    help="Whether to override existing found predictions. Default is False.",
)
@click.option(
    "--seed",
    type=int,
    help="Seed to use for random number generator. Default is 42.",
    default=42,
)
@click.option(
    "--use_msa_server",
    is_flag=True,
    help="Whether to use the MMSeqs2 server for MSA generation. Default is False.",
)
@click.option(
    "--msa_server_url",
    type=str,
    help="MSA server url. Used only if --use_msa_server is set. ",
    default="https://api.colabfold.com",
)
@click.option(
    "--msa_pairing_strategy",
    type=str,
    help="Pairing strategy to use. Used only if --use_msa_server is set. Options are 'greedy' and 'complete'",
    default="greedy",
)
def predict(
    data: str,
    out_dir: str,
    cache: str = "~/.boltz_jax",
    checkpoint: Optional[str] = None,
    devices: int = 1,
    accelerator: str = "gpu",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    diffusion_samples: int = 1,
    step_scale: float = 1.638,
    write_full_pae: bool = False,
    write_full_pde: bool = False,
    output_format: Literal["pdb", "mmcif"] = "mmcif",
    num_workers: int = 2,
    override: bool = False,
    seed: int = 42,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Predict protein structures using the JAX implementation of Boltz.
    
    Args:
        data: Path to input data
        out_dir: Output directory
        cache: Cache directory
        checkpoint: Checkpoint path
        devices: Number of devices
        accelerator: Accelerator type
        recycling_steps: Number of recycling steps
        sampling_steps: Number of sampling steps
        diffusion_samples: Number of diffusion samples
        step_scale: Step scale for diffusion
        write_full_pae: Whether to write full PAE
        write_full_pde: Whether to write full PDE
        output_format: Output format
        num_workers: Number of workers
        override: Whether to override existing predictions
        seed: Random seed
        use_msa_server: Whether to use MSA server
        msa_server_url: MSA server URL
        msa_pairing_strategy: MSA pairing strategy
    """
    # Set up paths
    data_path = Path(data).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    cache_dir = Path(cache).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Download data and model if needed
    download(cache_dir)
    
    # Check inputs
    input_data = check_inputs(data_path, out_dir, override)
    if not input_data:
        click.echo("No input data to process.")
        return
    
    # Download the CCD dictionary
    ccd_path = cache_dir / "ccd.pkl"
    if not ccd_path.exists():
        click.echo("CCD dictionary not found. Please run the download command first.")
        return
    
    # Set up jax platforms
    if accelerator == "gpu":
        jax.config.update("jax_platform_name", "gpu")
    elif accelerator == "tpu":
        jax.config.update("jax_platform_name", "tpu")
    else:
        jax.config.update("jax_platform_name", "cpu")
    
    # Process inputs
    processed_input = process_inputs(
        input_data,
        out_dir,
        ccd_path,
        msa_server_url,
        msa_pairing_strategy,
        use_msa_server=use_msa_server,
    )
    
    # Initialize model
    click.echo("Initializing model")
    model_config = {
        "atom_s": 384,
        "atom_z": 128,
        "token_s": 384,
        "token_z": 128,
        "num_bins": 64,
        "training_args": {},
        "validation_args": {},
        "embedder_args": {"num_layers": 3},
        "msa_args": {"num_layers": 4},
        "pairformer_args": {"num_layers": 12},
        "score_model_args": {"num_layers": 8},
        "diffusion_process_args": {
            "gamma_0": 0.605,
            "gamma_min": 1.107,
            "noise_scale": 0.901,
            "rho": 8,
            "step_scale": step_scale,
            "sigma_min": 0.0004,
            "sigma_max": 160.0,
            "sigma_data": 16.0,
            "P_mean": -1.2,
            "P_std": 1.5,
        },
        "diffusion_loss_args": {},
        "confidence_model_args": {"compute_pae": True},
        "atom_feature_dim": 128,
        "confidence_prediction": True,
        "confidence_imitate_trunk": False,
        "alpha_pae": 0.5,
        "structure_prediction_training": True,
        "atoms_per_window_queries": 32,
        "atoms_per_window_keys": 128,
        "use_ema": True,
        "ema_decay": 0.999,
    }
    
    # Initialize RNG
    rng = jax.random.PRNGKey(seed)
    
    # Initialize model and state
    model, state = init_model_and_state(
        rng=rng,
        config=model_config,
        checkpoint_path=checkpoint,
    )
    
    # Set up data module for prediction
    data_module = BoltzInferenceDataModule(
        processed_input=processed_input,
        batch_size=1,
        num_workers=num_workers,
    )
    
    # Prediction config
    predict_config = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
    }
    
    # Set up writer
    writer = BoltzWriter(
        out_dir=out_dir,
        processed_input=processed_input,
        write_full_pae=write_full_pae,
        write_full_pde=write_full_pde,
        output_format=output_format,
    )
    
    # Predict structures
    click.echo("Predicting structures")
    for batch in tqdm(data_module):
        # Convert numpy arrays to jax arrays
        jax_batch = {k: jnp.array(v) for k, v in batch.items() if isinstance(v, np.ndarray)}
        
        # Split RNG key
        rng, predict_rng = jax.random.split(rng)
        
        # Predict
        outputs = predict_structure(
            state=state,
            model=model,
            batch=jax_batch,
            rng=predict_rng,
            config=predict_config,
        )
        
        # Convert outputs back to numpy for writing
        numpy_outputs = {k: np.array(v) for k, v in outputs.items()}
        
        # Write outputs
        writer.write(
            batch=batch,
            outputs=numpy_outputs,
        )


if __name__ == "__main__":
    cli() 