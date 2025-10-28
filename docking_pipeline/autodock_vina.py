import logging
from pathlib import Path
from typing import Tuple
import subprocess
import numpy as np
from rdkit import Chem
import argparse
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_box_parameters(
    reference_ligand_sdf: Path, size: float = 20.0
) -> Tuple[float, float, float, float, float, float]:
    """
    Calculate box center and size from reference ligand coordinates.

    Parameters
    ----------
    reference_ligand_sdf : Path
        Path to reference ligand SDF file
    size : float, optional
        Padding around ligand in Angstroms (default: 10.0)

    Returns
    -------
    tuple of float
        Box parameters as
            (center_x, center_y, center_z, size_x, size_y, size_z)
    """
    logger.info(f"Calculating box parameters from {reference_ligand_sdf}")

    # Read SDF file
    supplier = Chem.SDMolSupplier(str(reference_ligand_sdf))
    mol = next(supplier)

    if mol is None:
        raise ValueError(
            f"Could not read molecule from {reference_ligand_sdf}"
        )

    # Extract coordinates
    conf = mol.GetConformer()
    coords = np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
    )

    # Calculate center
    center = coords.mean(axis=0)

    # Calculate size with padding
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    box_size = max(size, (max_coords - min_coords + 5).max())

    logger.info(
        f"Box center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
    )
    logger.info(f"Box size: ({box_size:.2f}, {box_size:.2f}, {box_size:.2f})")

    return center[0], center[1], center[2], box_size, box_size, box_size


def prepare_receptor(
    protein_pdb: Path, output_pdbqt: Path, box: tuple[float]
) -> None:
    """
    Prepare receptor for docking using Meeko's mk_prepare_receptor.py.

    Parameters
    ----------
    protein_pdb : Path
        Path to input protein PDB file
    output_pdbqt : Path
        Path to output PDBQT file
    box : tuple of float
        Box parameters (center_x, center_y, center_z, size_x, size_y, size_z)
    """
    logger.info(f"Preparing receptor from {protein_pdb}")

    cmd = [
        "mk_prepare_receptor.py",
        "-i",
        str(protein_pdb),
        "-o",
        str(output_pdbqt),
        "-p",
        "-v",
        "--box_size",
        f"{box[3]:.0f}",
        f"{box[4]:.0f}",
        f"{box[5]:.0f}",
        "--box_center",
        f"{box[0]:.3f}",
        f"{box[1]:.3f}",
        f"{box[2]:.3f}",
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.debug(f"STDERR: {result.stderr}")

    logger.info(f"Receptor prepared successfully: {output_pdbqt}")


def prepare_ligand(ligand_sdf: Path, output_pdbqt: Path) -> None:
    """
    Prepare ligand for docking using Meeko's mk_prepare_ligand.py.

    Parameters
    ----------
    ligand_sdf : Path
        Path to input ligand SDF file
    output_pdbqt : Path
        Path to output PDBQT file
    """
    logger.info(f"Preparing ligand from {ligand_sdf}")

    cmd = [
        "mk_prepare_ligand.py",
        "-i",
        str(ligand_sdf),
        "-o",
        str(output_pdbqt),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.debug(f"STDERR: {result.stderr}")

    logger.info(f"Ligand prepared successfully: {output_pdbqt}")


# flake8: noqa: C901
def run_vina_docking(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    box_file: tuple[float],
    output_folder: Path,
    num_poses: int = 5,
    exhaustiveness: int = 8,
) -> None:
    """
    Run AutoDock Vina docking.

    Parameters
    ----------
    receptor_pdbqt : Path
        Path to receptor PDBQT file
    ligand_pdbqt : Path
        Path to ligand PDBQT file
    box_file : tuple of float
        Config file for box
    output_folder : Path
        Path to output folder
    num_poses : int, optional
        Number of poses to generate (default: 5)
    exhaustiveness : int, optional
        Exhaustiveness of the search (default: 8)
    """
    logger.info(f"Running Vina docking for {ligand_pdbqt}")

    # Create output path by combining output folder with ligand filename
    ligand_name = ligand_pdbqt.stem  # Get filename without extension
    output_pdbqt = output_folder / f"{ligand_name}_docked.pdbqt"

    cmd = [
        "vina",
        "--receptor",
        str(receptor_pdbqt),
        "--ligand",
        str(ligand_pdbqt),
        "--config",
        str(box_file),
        "--out",
        str(output_pdbqt),
        "--num_modes",
        str(num_poses),
        "--exhaustiveness",
        str(exhaustiveness),
    ]

    logger.info(f"Running command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"Vina failed with return code {result.returncode}")
        if result.stderr:
            logger.error(f"STDERR: {result.stderr}")
        raise RuntimeError(f"Vina docking failed for {ligand_pdbqt}")

    if result.stdout:
        logger.info(f"STDOUT: {result.stdout}")
    if result.stderr:
        logger.debug(f"STDERR: {result.stderr}")

    logger.info(f"Docking completed: {output_pdbqt}")

    # Convert docked PDBQT to SDF using mk_export.py
    output_sdf = output_pdbqt.with_suffix(".sdf")

    cmd_export = [
        "mk_export.py",
        str(output_pdbqt),
        "-s",
        str(output_sdf),
    ]

    logger.info(f"Converting to SDF: {' '.join(cmd_export)}")

    result_export = subprocess.run(cmd_export, capture_output=True, text=True)

    if result_export.returncode != 0:
        logger.error(
            f"mk_export.py failed with return code {result_export.returncode}"
        )
        if result_export.stderr:
            logger.error(f"STDERR: {result_export.stderr}")
        raise RuntimeError(f"SDF conversion failed for {output_pdbqt}")

    if result_export.stdout:
        logger.info(f"STDOUT: {result_export.stdout}")
    if result_export.stderr:
        logger.debug(f"STDERR: {result_export.stderr}")

    logger.info(f"SDF file created: {output_sdf}")


def main():

    parser = argparse.ArgumentParser(
        description="AutoDock Vina docking pipeline"
    )
    parser.add_argument(
        "-p", "--protein", type=Path, required=True, help="Protein PDB file"
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type=Path,
        required=True,
        help="Reference ligand SDF file",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=Path,
        required=True,
        help="Ligand data CSV file with similar_to_original column",
    )
    parser.add_argument(
        "-c",
        "--conformers",
        type=Path,
        required=True,
        help="File with paths to ligand conformers",
    )
    parser.add_argument(
        "-o", "--outdir", type=Path, required=True, help="Output directory"
    )

    args = parser.parse_args()

    # Setup logging
    args.outdir.mkdir(parents=True, exist_ok=True)
    log_file = args.outdir / "autodock-vina.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger.info("Starting AutoDock Vina pipeline")

    # Calculate box parameters
    box_params = calculate_box_parameters(args.ligand)
    logger.info(f"Box parameters calculated: {box_params}")

    # Prepare receptor
    receptor_folder = args.outdir / "receptor"
    prepare_receptor(args.protein, receptor_folder, box_params)
    receptor_pdbqt = Path(str(receptor_folder) + ".pdbqt")

    # Read conformers file

    df = pd.read_csv(args.conformers)
    logger.info(f"Read {len(df)} molecules from {args.conformers}")

    # Create output directory for PDBQT files
    pdbqt_dir = args.outdir / "pdbqt_files"
    pdbqt_dir.mkdir(parents=True, exist_ok=True)

    # Prepare ligands and store paths
    pdbqt_columns = {}
    for col in df.columns:
        if col.startswith("conf_"):
            pdbqt_col = f"pdbqt_{col}"
            pdbqt_columns[pdbqt_col] = []

            for sdf_path in df[col]:
                # Extract filename without extension
                sdf_file = Path(sdf_path)
                base_name = sdf_file.stem  # molecule_<idx>_conformer_<n>

                # Create output PDBQT path
                pdbqt_path = pdbqt_dir / f"{base_name}.pdbqt"

                # Prepare ligand
                prepare_ligand(sdf_file, pdbqt_path)

                pdbqt_columns[pdbqt_col].append(str(pdbqt_path))

    # Add PDBQT paths to dataframe
    for col, paths in pdbqt_columns.items():
        df[col] = paths

    # Save updated dataframe
    output_csv = args.outdir / "conformers_with_pdbqt.csv"
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved conformers with PDBQT paths to {output_csv}")

    # Run docking for all conformers
    logger.info("Running docking for all conformers")

    # Create config file for vina
    config_file = Path(str(receptor_folder) + ".box.txt")

    # Create docking output folder
    docking_dir = args.outdir / "docking_structures"
    docking_dir.mkdir(parents=True, exist_ok=True)

    # Run docking for all conformers and collect results
    docking_columns = {}
    for pdbqt_col in pdbqt_columns.keys():
        # Extract index from pdbqt column name (pdbqt_conf_0 -> 0)
        conf_idx = pdbqt_col.split("_")[-1]
        dock_col = f"dock_{conf_idx}"
        docking_columns[dock_col] = []

        for pdbqt_path in df[pdbqt_col]:
            ligand_pdbqt = Path(pdbqt_path)

            # Run docking
            run_vina_docking(
                receptor_pdbqt,
                ligand_pdbqt,
                config_file,
                docking_dir,
                num_poses=5,
                exhaustiveness=8,
            )

            # Get output SDF path
            ligand_name = ligand_pdbqt.stem
            output_sdf = docking_dir / f"{ligand_name}_docked.sdf"
            docking_columns[dock_col].append(str(output_sdf))

    # Create output dataframe with docked columns, smiles, and similar flag
    output_df = pd.DataFrame()
    output_df["smiles"] = df["smiles"]

    output_df["similar_to_original"] = pd.read_csv(args.data)[
        "similar_to_original"
    ]

    for col, paths in docking_columns.items():
        output_df[col] = paths

    # Save docked results
    output_csv = args.outdir / "docked.csv"
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Saved docking results to {output_csv}")
    logger.info(f"Docking completed for all {len(df)} molecules")


if __name__ == "__main__":
    main()
