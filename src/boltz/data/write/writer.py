from dataclasses import asdict, replace
from pathlib import Path
from typing import Literal

import numpy as np
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import BasePredictionWriter
from torch import Tensor

from boltz.data.types import (
    Interface,
    Record,
    Structure,
)
from boltz.data.write.mmcif import to_mmcif
from boltz.data.write.pdb import to_pdb


class BoltzWriter(BasePredictionWriter):
    """Custom writer for predictions."""

    def __init__(
        self,
        data_dir: str,
        output_dir: str,
        output_format: Literal["pdb", "mmcif"] = "mmcif",
    ) -> None:
        """Initialize the writer.

        Parameters
        ----------
        output_dir : str
            The directory to save the predictions.

        """
        super().__init__(write_interval="batch")
        if output_format not in ["pdb", "mmcif"]:
            msg = f"Invalid output format: {output_format}"
            raise ValueError(msg)

        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_format = output_format
        self.failed = 0

        # Create the output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(
        self,
        trainer: Trainer,  # noqa: ARG002
        pl_module: LightningModule,  # noqa: ARG002
        prediction: dict[str, Tensor],
        batch_indices: list[int],  # noqa: ARG002
        batch: dict[str, Tensor],
        batch_idx: int,  # noqa: ARG002
        dataloader_idx: int,  # noqa: ARG002
    ) -> None:
        """Write the predictions to disk."""
        if prediction["exception"]:
            self.failed += 1
            return

        # Get the records
        records: list[Record] = batch["record"]

        # Get the predictions
        coords = prediction["coords"]
        coords = coords.unsqueeze(0)

        pad_masks = prediction["masks"]
        if prediction.get("confidence") is not None:
            confidences = prediction["confidence"]
            confidences = confidences.reshape(len(records), -1).tolist()
        else:
            confidences = [0.0 for _ in range(len(records))]

        # Iterate over the records
        for record, coord, pad_mask, _confidence in zip(
            records, coords, pad_masks, confidences
        ):
            # Load the structure
            path = self.data_dir / f"{record.id}.npz"
            structure: Structure = Structure.load(path)

            # Compute chain map with masked removed, to be used later
            chain_map = {}
            for i, mask in enumerate(structure.mask):
                if mask:
                    chain_map[len(chain_map)] = i

            # Remove masked chains completely
            structure = structure.remove_invalid_chains()

            for model_idx in range(coord.shape[0]):
                # Get model coord
                model_coord = coord[model_idx]
                # Unpad
                coord_unpad = model_coord[pad_mask.bool()]
                coord_unpad = coord_unpad.cpu().numpy()

                # New atom table
                atoms = structure.atoms
                atoms["coords"] = coord_unpad
                atoms["is_present"] = True

                # Mew residue table
                residues = structure.residues
                residues["is_present"] = True

                # Update the structure
                interfaces = np.array([], dtype=Interface)
                new_structure: Structure = replace(
                    structure,
                    atoms=atoms,
                    residues=residues,
                    interfaces=interfaces,
                )

                # Update chain info
                chain_info = []
                for chain in new_structure.chains:
                    old_chain_idx = chain_map[chain["asym_id"]]
                    old_chain_info = record.chains[old_chain_idx]
                    new_chain_info = replace(
                        old_chain_info,
                        chain_id=int(chain["asym_id"]),
                        valid=True,
                    )
                    chain_info.append(new_chain_info)

                # Save the structure
                struct_dir = self.output_dir / record.id
                struct_dir.mkdir(exist_ok=True)

                if self.output_format == "pdb":
                    path = struct_dir / f"{record.id}_model_{model_idx}.pdb"
                    with path.open("w") as f:
                        f.write(to_pdb(new_structure))
                elif self.output_format == "mmcif":
                    path = struct_dir / f"{record.id}_model_{model_idx}.cif"
                    with path.open("w") as f:
                        f.write(to_mmcif(new_structure))
                else:
                    path = struct_dir / f"{record.id}_model_{model_idx}.npz"
                    np.savez_compressed(path, **asdict(new_structure))

    def on_predict_epoch_end(
        self,
        trainer: Trainer,  # noqa: ARG00s2
        pl_module: LightningModule,  # noqa: ARG002
    ) -> None:
        """Print the number of failed examples."""
        # Print number of failed examples
        print(f"Number of failed examples: {self.failed}")  # noqa: T201
