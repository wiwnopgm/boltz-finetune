import pickle
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Optional

import click
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from tqdm import tqdm

from boltz.data import const
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.msa.mmseqs2 import run_mmseqs2
from boltz.data.parse.a3m import parse_a3m
from boltz.data.parse.csv import parse_csv
from boltz.data.parse.fasta import parse_fasta
from boltz.data.parse.yaml import parse_yaml
from boltz.data.types import MSA, Manifest, Record
from boltz.data.write.writer import BoltzWriter
from boltz.model.model import Boltz1

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


@rank_zero_only
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


@rank_zero_only
def process_inputs(  # noqa: C901, PLR0912, PLR0915
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
        The input data.
    out_dir : Path
        The output directory.
    ccd_path : Path
        The path to the CCD dictionary.
    max_msa_seqs : int, optional
        Max number of MSA sequences, by default 4096.
    use_msa_server : bool, optional
        Whether to use the MMSeqs2 server for MSA generation, by default False.

    Returns
    -------
    BoltzProcessedInput
        The processed input data.

    """
    click.echo("Processing input data.")
    existing_records = None

    # Check if manifest exists at output path
    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        click.echo(f"Found a manifest file at output directory: {out_dir}")

        manifest: Manifest = Manifest.load(manifest_path)
        input_ids = [d.stem for d in data]
        existing_records, processed_ids = zip(
            *[
                (record, record.id)
                for record in manifest.records
                if record.id in input_ids
            ]
        )

        if isinstance(existing_records, tuple):
            existing_records = list(existing_records)

        # Check how many examples need to be processed
        missing = len(input_ids) - len(processed_ids)
        if not missing:
            click.echo("All examples in data are processed. Updating the manifest")
            # Dump updated manifest
            updated_manifest = Manifest(existing_records)
            updated_manifest.dump(out_dir / "processed" / "manifest.json")
            return

        click.echo(f"{missing} missing ids. Preprocessing these ids")
        missing_ids = list(set(input_ids).difference(set(processed_ids)))
        data = [d for d in data if d.stem in missing_ids]
        assert len(data) == len(missing_ids)

    # Create output directories
    msa_dir = out_dir / "msa"
    structure_dir = out_dir / "processed" / "structures"
    processed_msa_dir = out_dir / "processed" / "msa"
    predictions_dir = out_dir / "predictions"

    out_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)
    structure_dir.mkdir(parents=True, exist_ok=True)
    processed_msa_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    # Load CCD
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    if existing_records is not None:
        click.echo(f"Found {len(existing_records)} records. Adding them to records")

    # Parse input data
    records: list[Record] = existing_records if existing_records is not None else []
    for path in tqdm(data):
        try:
            # Parse data
            if path.suffix in (".fa", ".fas", ".fasta"):
                target = parse_fasta(path, ccd)
            elif path.suffix in (".yml", ".yaml"):
                target = parse_yaml(path, ccd)
            elif path.is_dir():
                msg = f"Found directory {path} instead of .fasta or .yaml, skipping."
                raise RuntimeError(msg)
            else:
                msg = (
                    f"Unable to parse filetype {path.suffix}, "
                    "please provide a .fasta or .yaml file."
                )
                raise RuntimeError(msg)

            # Get target id
            target_id = target.record.id

            # Get all MSA ids and decide whether to generate MSA
            to_generate = {}
            prot_id = const.chain_type_ids["PROTEIN"]
            for chain in target.record.chains:
                # Add to generate list, assigning entity id
                if (chain.mol_type == prot_id) and (chain.msa_id == 0):
                    entity_id = chain.entity_id
                    msa_id = f"{target_id}_{entity_id}"
                    to_generate[msa_id] = target.sequences[entity_id]
                    chain.msa_id = msa_dir / f"{msa_id}.csv"

                # We do not support msa generation for non-protein chains
                elif chain.msa_id == 0:
                    chain.msa_id = -1

            # Generate MSA
            if to_generate and not use_msa_server:
                msg = "Missing MSA's in input and --use_msa_server flag not set."
                raise RuntimeError(msg)

            if to_generate:
                msg = f"Generating MSA for {path} with {len(to_generate)} protein entities."
                click.echo(msg)
                compute_msa(
                    data=to_generate,
                    target_id=target_id,
                    msa_dir=msa_dir,
                    msa_server_url=msa_server_url,
                    msa_pairing_strategy=msa_pairing_strategy,
                )

            # Parse MSA data
            msas = sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})
            msa_id_map = {}
            for msa_idx, msa_id in enumerate(msas):
                # Check that raw MSA exists
                msa_path = Path(msa_id)
                if not msa_path.exists():
                    msg = f"MSA file {msa_path} not found."
                    raise FileNotFoundError(msg)

                # Dump processed MSA
                processed = processed_msa_dir / f"{target_id}_{msa_idx}.npz"
                msa_id_map[msa_id] = f"{target_id}_{msa_idx}"
                if not processed.exists():
                    # Parse A3M
                    if msa_path.suffix == ".a3m":
                        msa: MSA = parse_a3m(
                            msa_path,
                            taxonomy=None,
                            max_seqs=max_msa_seqs,
                        )
                    elif msa_path.suffix == ".csv":
                        msa: MSA = parse_csv(msa_path, max_seqs=max_msa_seqs)
                    else:
                        msg = f"MSA file {msa_path} not supported, only a3m or csv."
                        raise RuntimeError(msg)

                    msa.dump(processed)

            # Modify records to point to processed MSA
            for c in target.record.chains:
                if (c.msa_id != -1) and (c.msa_id in msa_id_map):
                    c.msa_id = msa_id_map[c.msa_id]

            # Keep record
            records.append(target.record)

            # Dump structure
            struct_path = structure_dir / f"{target.record.id}.npz"
            target.structure.dump(struct_path)

        except Exception as e:
            if len(data) > 1:
                print(f"Failed to process {path}. Skipping. Error: {e}.")
            else:
                raise e

    # Dump manifest
    manifest = Manifest(records)
    manifest.dump(out_dir / "processed" / "manifest.json")


@click.group()
def cli() -> None:
    """Boltz1."""
    return


@cli.command()
@click.argument("data", type=click.Path(exists=True))
@click.option(
    "--msa_dir",
    type=click.Path(exists=True),
    help="Directory containing MSA files",
    required=True,
)
@click.option(
    "--out_dir",
    type=click.Path(exists=False),
    help="The path where to save the processed data.",
    default="./boltz_processed",
)
@click.option(
    "--redis_host",
    type=str,
    help="Redis host (default: localhost)",
    default="localhost",
)
@click.option(
    "--ccd_port",
    type=int,
    help="Port for CCD Redis server (default: 7777)",
    default=7777,
)
@click.option(
    "--taxonomy_port",
    type=int,
    help="Port for taxonomy Redis server (default: 7778)",
    default=7778,
)
@click.option(
    "--num_processes",
    type=int,
    help="Number of processes to use (default: 4)",
    default=4,
)
@click.option(
    "--max_seqs",
    type=int,
    help="Maximum number of sequences to process (default: 1000)",
    default=1000,
)
def process_data(
    data: str,
    msa_dir: str,
    out_dir: str = "./boltz_processed",
    redis_host: str = "localhost",
    ccd_port: int = 7777,
    taxonomy_port: int = 7778,
    num_processes: int = 4,
    max_seqs: int = 1000,
) -> None:
    """Process input data for Boltz model training or fine-tuning.
    
    This command:
    1. Processes CIF/PDB files from data_dir
    2. Processes MSA files from msa_dir
    3. Prepares the data in the format required by the model
    4. Saves the processed data to the output directory
    """
    import subprocess
    from pathlib import Path
    
    # Create output directories
    data = Path(data).expanduser()
    msa_dir = Path(msa_dir).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get the absolute path to the scripts directory
    script_dir = Path(__file__).parent.parent.parent / "scripts" / "process"
    
    # Create output directories
    structures_output_dir = out_dir / "processed_structures"
    msa_output_dir = out_dir / "processed_msa"
    structures_output_dir.mkdir(parents=True, exist_ok=True)
    msa_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process structures using CCD Redis server
    subprocess.run(
        ["python", str(script_dir / "rcsb.py"), 
         "--datadir", str(data),
         "--outdir", str(structures_output_dir),
         "--redis-host", redis_host,
         "--redis-port", str(ccd_port)],
        check=True,
    )
    
    # Process MSA files using taxonomy Redis server
    subprocess.run(
        ["python", str(script_dir / "msa.py"), 
         "--msadir", str(msa_dir),
         "--outdir", str(msa_output_dir),
         "--redis-host", redis_host,
         "--redis-port", str(taxonomy_port),
         "--max-seqs", str(max_seqs)],
        check=True,
    )
    
    click.echo(f"Data processing completed. Processed data saved to {out_dir}")
    click.echo(f"Processed structures saved to: {structures_output_dir}")
    click.echo(f"Processed MSA files saved to: {msa_output_dir}")


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
    help="The directory where to download the data and model. Default is ~/.boltz.",
    default="~/.boltz",
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
    help="Seed to use for random number generator. Default is None (no seeding).",
    default=None,
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
    cache: str = "~/.boltz",
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
    seed: Optional[int] = None,
    use_msa_server: bool = False,
    msa_server_url: str = "https://api.colabfold.com",
    msa_pairing_strategy: str = "greedy",
) -> None:
    """Run predictions with Boltz-1."""
    # If cpu, write a friendly warning
    if accelerator == "cpu":
        msg = "Running on CPU, this will be slow. Consider using a GPU."
        click.echo(msg)

    # Set no grad
    torch.set_grad_enabled(False)

    # Ignore matmul precision warning
    torch.set_float32_matmul_precision("highest")

    # Set seed if desired
    if seed is not None:
        seed_everything(seed)

    # Set cache path
    cache = Path(cache).expanduser()
    cache.mkdir(parents=True, exist_ok=True)

    # Create output directories
    data = Path(data).expanduser()
    out_dir = Path(out_dir).expanduser()
    out_dir = out_dir / f"boltz_results_{data.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download necessary data and model
    download(cache)

    # Validate inputs
    data = check_inputs(data, out_dir, override)
    if not data:
        click.echo("No predictions to run, exiting.")
        return

    # Set up trainer
    strategy = "auto"
    if (isinstance(devices, int) and devices > 1) or (
        isinstance(devices, list) and len(devices) > 1
    ):
        strategy = DDPStrategy()
        if len(data) < devices:
            msg = (
                "Number of requested devices is greater "
                "than the number of predictions."
            )
            raise ValueError(msg)

    msg = f"Running predictions for {len(data)} structure"
    msg += "s" if len(data) > 1 else ""
    click.echo(msg)

    # Process inputs
    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        use_msa_server=use_msa_server,
        msa_server_url=msa_server_url,
        msa_pairing_strategy=msa_pairing_strategy,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=Manifest.load(processed_dir / "manifest.json"),
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
    )

    # Create data module
    data_module = BoltzInferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        num_workers=num_workers,
    )

    # Load model
    if checkpoint is None:
        checkpoint = cache / "boltz1_conf.ckpt"

    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "write_confidence_summary": True,
        "write_full_pae": write_full_pae,
        "write_full_pde": write_full_pde,
    }

    # Load model with verification
    try:
        click.echo(f"Loading model from checkpoint: {checkpoint}")
        diffusion_params = BoltzDiffusionParams()
        diffusion_params.step_scale = step_scale
        model_module: Boltz1 = Boltz1.load_from_checkpoint(
            checkpoint,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(diffusion_params),
            ema=False,
        )
        model_module.eval()
        
        # Verify model loaded successfully by checking a few attributes
        if not hasattr(model_module, "forward") or not hasattr(model_module, "predict_step"):
            raise AttributeError("Model loading appears incomplete - missing expected methods")
            
        click.echo("Model loaded successfully!")
    except Exception as e:
        click.echo(f"Error loading model from checkpoint: {e}")
        raise RuntimeError(f"Failed to load model from {checkpoint}") from e

    # Create prediction writer
    pred_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=out_dir / "predictions",
        output_format=output_format,
    )

    trainer = Trainer(
        default_root_dir=out_dir,
        strategy=strategy,
        callbacks=[pred_writer],
        accelerator=accelerator,
        devices=devices,
        precision=32,
    )

    # Compute predictions
    trainer.predict(
        model_module,
        datamodule=data_module,
        return_predictions=False,
    )


@cli.command()
@click.option(
    "--data_dir",
    type=click.Path(exists=True),
    help="Directory containing processed data",
    required=True,
)
@click.option(
    "--output_dir",
    type=click.Path(exists=False),
    help="Directory for model checkpoints and logs",
    default="./boltz_checkpoints",
)
@click.option(
    "--max_epochs",
    type=int,
    help="Maximum number of training epochs",
    default=100,
)
@click.option(
    "--learning_rate",
    type=float,
    help="Learning rate for training",
    default=1e-4,
)
@click.option(
    "--batch_size",
    type=int,
    help="Batch size for training",
    default=32,
)
@click.option(
    "--method",
    type=click.Choice(["lora", "full"]),
    help="Training method: 'lora' for LoRA training or 'full' for full model training",
    default="full",
)
@click.option(
    "--rank",
    type=int,
    help="Rank for LoRA training (only used if method='lora')",
    default=8,
)
@click.option(
    "--alpha",
    type=float,
    help="Alpha parameter for LoRA training (only used if method='lora')",
    default=16.0,
)
def train(
    data_dir: str,
    output_dir: str = "./boltz_checkpoints",
    max_epochs: int = 100,
    learning_rate: float = 1e-4,
    batch_size: int = 32,
    method: str = "full",
    rank: int = 8,
    alpha: float = 16.0,
) -> None:
    """Train the Boltz model.
    
    This command:
    1. Loads the processed data from data_dir
    2. Trains the model using the specified method (full training or LoRA)
    3. Saves checkpoints and logs to output_dir
    """
    from pathlib import Path
    import torch
    from boltz.models import BoltzModel
    from boltz.data import BoltzDataset
    from boltz.train import train_model
    
    # Create output directories
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = BoltzDataset(data_dir)
    
    # Initialize model
    model = BoltzModel()
    
    # Configure training based on method
    if method == "lora":
        from boltz.lora import apply_lora
        model = apply_lora(model, rank=rank, alpha=alpha)
    
    # Train model
    train_model(
        model=model,
        dataset=dataset,
        output_dir=output_path,
        max_epochs=max_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
    )
    
    click.echo(f"Training completed. Model checkpoints saved to {output_path}")


@cli.command()
@click.option(
    "--host",
    type=str,
    help="Hostname or IP address of the remote cluster",
    required=True,
)
@click.option(
    "--user",
    type=str,
    help="Username for SSH connection",
    required=True,
)
@click.option(
    "--identity_file",
    type=click.Path(exists=True),
    help="Path to SSH identity file (private key)",
    required=True,
)
@click.option(
    "--config_file",
    type=click.Path(exists=False),
    help="Path to save SSH config file",
    default="~/.ssh/boltz_config",
)
def ssh_connect(
    host: str,
    user: str,
    identity_file: str,
    config_file: str,
) -> None:
    """Connect to a remote cluster via SSH.
    
    This command:
    1. Creates or updates an SSH config file with the provided host information
    2. Establishes an SSH connection to the remote cluster
    3. Stores the host information for future use
    """
    import os
    import subprocess
    from pathlib import Path
    
    # Expand paths
    identity_file = Path(identity_file).expanduser()
    config_file = Path(config_file).expanduser()
    
    # Create SSH config directory if it doesn't exist
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create or update SSH config file
    config_content = f"""Host {host}
  HostName {host}
  User {user}
  IdentityFile {identity_file}
"""
    
    # Check if config file exists and contains the host
    if config_file.exists():
        with open(config_file, "r") as f:
            existing_config = f.read()
        
        # If host already exists, update it
        if f"Host {host}" in existing_config:
            lines = existing_config.split("\n")
            new_lines = []
            skip_lines = False
            
            for line in lines:
                if line.startswith(f"Host {host}"):
                    skip_lines = True
                    new_lines.append(config_content.strip())
                elif skip_lines and line.startswith("Host "):
                    skip_lines = False
                    new_lines.append(line)
                elif not skip_lines:
                    new_lines.append(line)
            
            if skip_lines:  # If we were skipping at the end of the file
                new_lines.append("")
            
            config_content = "\n".join(new_lines)
    
    # Write config file
    with open(config_file, "w") as f:
        f.write(config_content)
    
    # Store host information for future use
    cache_dir = Path.home() / ".boltz"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cache_dir / "ssh_host.txt", "w") as f:
        f.write(host)
    
    # Establish SSH connection
    click.echo(f"Connecting to {host} as {user}...")
    try:
        subprocess.run(["ssh", "-F", str(config_file), host], check=True)
    except subprocess.CalledProcessError:
        click.echo("SSH connection failed. You can try connecting manually with:")
        click.echo(f"ssh -F {config_file} {host}")
    except KeyboardInterrupt:
        click.echo("SSH connection terminated by user.")


@cli.command()
@click.option(
    "--host",
    type=str,
    help="Hostname or IP address of the remote cluster (if not provided, will use the last connected host)",
)
@click.option(
    "--config_file",
    type=click.Path(exists=True),
    help="Path to SSH config file",
    default="~/.ssh/boltz_config",
)
@click.option(
    "--database_dir",
    type=click.Path(exists=False),
    help="Directory containing Redis database files on the remote host",
    default="/ist-nas/users/bunditb/boltz/scripts/database",
)
@click.option(
    "--local_port_ccd",
    type=int,
    help="Local port to forward for CCD Redis server",
    default=7777,
)
@click.option(
    "--local_port_taxonomy",
    type=int,
    help="Local port to forward for taxonomy Redis server",
    default=7778,
)
@click.option(
    "--remote_port_ccd",
    type=int,
    help="Remote port for CCD Redis server",
    default=7777,
)
@click.option(
    "--remote_port_taxonomy",
    type=int,
    help="Remote port for taxonomy Redis server",
    default=7778,
)
def start_redis_servers(
    host: str = None,
    config_file: str = "~/.ssh/boltz_config",
    database_dir: str = "/ist-nas/users/bunditb/boltz/scripts/database",
    local_port_ccd: int = 7777,
    local_port_taxonomy: int = 7778,
    remote_port_ccd: int = 7777,
    remote_port_taxonomy: int = 7778,
) -> None:
    """Start Redis servers on a remote cluster and wait for connections.
    
    This command:
    1. Connects to the remote cluster via SSH
    2. Starts Redis servers for CCD and taxonomy databases
    3. Sets up port forwarding to access the Redis servers locally
    4. Waits for the Redis servers to accept connections
    """
    import os
    import subprocess
    import time
    from pathlib import Path
    
    # Expand paths
    config_file = Path(config_file).expanduser()
    
    # Get host if not provided
    if host is None:
        cache_dir = Path.home() / ".boltz"
        if (cache_dir / "ssh_host.txt").exists():
            with open(cache_dir / "ssh_host.txt", "r") as f:
                host = f.read().strip()
        else:
            click.echo("Error: No host specified and no previous host found.")
            return
    
    # Create a script to start Redis servers
    redis_script = f"""#!/bin/bash
cd {database_dir}
redis-server --dbfilename ccd.rdb --port {remote_port_ccd} &
redis-server --dbfilename taxonomy.rdb --port {remote_port_taxonomy} &
echo "Redis servers started"
"""
    
    # Create a temporary script file
    script_path = Path.home() / ".boltz" / "start_redis.sh"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(redis_script)
    
    # Make the script executable
    os.chmod(script_path, 0o755)
    
    # Start Redis servers on the remote host
    click.echo(f"Starting Redis servers on {host}...")
    try:
        subprocess.run(
            ["ssh", "-F", str(config_file), host, f"bash -s < {script_path}"],
            check=True,
        )
    except subprocess.CalledProcessError as e:
        click.echo(f"Error starting Redis servers: {e}")
        return
    
    # Set up port forwarding
    click.echo("Setting up port forwarding...")
    try:
        # Start port forwarding in the background
        ccd_forward = subprocess.Popen(
            [
                "ssh", "-F", str(config_file), "-L", 
                f"{local_port_ccd}:localhost:{remote_port_ccd}", 
                "-N", host
            ],
        )
        
        taxonomy_forward = subprocess.Popen(
            [
                "ssh", "-F", str(config_file), "-L", 
                f"{local_port_taxonomy}:localhost:{remote_port_taxonomy}", 
                "-N", host
            ],
        )
        
        # Wait for Redis servers to accept connections
        click.echo("Waiting for Redis servers to accept connections...")
        
        # Function to check if Redis server is accepting connections
        def check_redis_connection(port):
            try:
                result = subprocess.run(
                    ["redis-cli", "-h", "localhost", "-p", str(port), "ping"],
                    capture_output=True,
                    text=True,
                    timeout=1,
                )
                return result.stdout.strip() == "PONG"
            except (subprocess.SubprocessError, subprocess.TimeoutExpired):
                return False
        
        # Wait for both Redis servers
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            ccd_ready = check_redis_connection(local_port_ccd)
            taxonomy_ready = check_redis_connection(local_port_taxonomy)
            
            if ccd_ready and taxonomy_ready:
                click.echo("Both Redis servers are accepting connections!")
                break
            
            attempt += 1
            click.echo(f"Waiting for Redis servers... (attempt {attempt}/{max_attempts})")
            time.sleep(2)
        
        if attempt >= max_attempts:
            click.echo("Timeout waiting for Redis servers to accept connections.")
            click.echo("You may need to check the Redis server logs on the remote host.")
        
        # Keep the port forwarding running
        click.echo("\nRedis servers are running with port forwarding:")
        click.echo(f"CCD Redis server: localhost:{local_port_ccd} -> {host}:{remote_port_ccd}")
        click.echo(f"Taxonomy Redis server: localhost:{local_port_taxonomy} -> {host}:{remote_port_taxonomy}")
        click.echo("\nPress Ctrl+C to stop the port forwarding and exit.")
        
        # Wait for user to press Ctrl+C
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            click.echo("\nStopping port forwarding...")
            ccd_forward.terminate()
            taxonomy_forward.terminate()
            click.echo("Port forwarding stopped.")
    
    except Exception as e:
        click.echo(f"Error setting up port forwarding: {e}")
        return