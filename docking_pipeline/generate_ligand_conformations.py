import logging
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import argparse


def setup_logging(output_folder: Path) -> None:
    """Setup logging configuration."""
    log_file = output_folder / "conformer_generation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def generate_conformers(smiles: str, n_conformers: int) -> list[Chem.Mol]:
    """
    Generate N conformers from a SMILES string.

    Parameters
    ----------
    smiles : str
        SMILES string representation of the molecule
    n_conformers : int
        Number of conformers to generate

    Returns
    -------
    list[Chem.Mol]

    Raises
    ------
    Exception
        If conformer generation fails

    Notes
    -----
    This function performs the following steps:
    1. Converts SMILES string to RDKit molecule object
    2. Adds hydrogen atoms to the molecule
    3. Generates multiple 3D conformers using random coordinates
    4. Optimizes each conformer using UFF force field

    The function uses a fixed random seed (42) for reproducibility.

    Examples
    --------
    >>> mols, conf_ids = generate_conformers("CCO", 10)
    >>> len(mols)
    10
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logging.error(f"Invalid SMILES: {smiles}")
        return []

    # Add hydrogens
    mol = Chem.AddHs(mol)

    # Generate conformers
    try:
        confIds = AllChem.EmbedMultipleConfs(
            mol, numConfs=n_conformers, randomSeed=42, useRandomCoords=True
        )

        # Optimize conformers
        for confId in confIds:
            AllChem.UFFOptimizeMolecule(mol, confId=confId)

        logging.info(
            f"Generated {len(confIds)} conformers for SMILES: {smiles}"
        )
        return [mol] * len(confIds), list(confIds)

    except Exception as e:
        logging.error(f"Error generating conformers for {smiles}: {str(e)}")
        return [], []


def save_conformers(
    mol: Chem.Mol, conf_ids: list[int], idx: int, conformations_folder: Path
) -> list[str]:
    """Save conformers to SDF files.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule object with conformers
    conf_ids : list[int]
        List of conformer IDs
    idx : int
        Index of the molecule
    conformations_folder : Path
        Path to conformations folder

    Returns
    -------
    list[str]
        List of paths to saved conformer files
    """
    conformer_paths = []

    for n, conf_id in enumerate(conf_ids, 1):
        filename = f"ligand_{idx}_conformer_{n}.sdf"
        filepath = conformations_folder / filename

        writer = Chem.SDWriter(str(filepath))
        writer.write(mol, confId=conf_id)
        writer.close()

        conformer_paths.append(str(filepath))

    return conformer_paths


def main() -> None:
    """Main function to generate conformers from SMILES."""
    parser = argparse.ArgumentParser(
        description="Generate conformers from SMILES"
    )
    parser.add_argument(
        "-c",
        "--csv",
        dest="csv_path",
        type=str,
        help="Path to CSV file with 'smiles' column",
    )
    parser.add_argument(
        "-n",
        "--n-conformerms",
        dest="n_conformers",
        type=int,
        help="Number of conformers to generate",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_folder",
        type=str,
        help="Path to output folder",
    )

    args = parser.parse_args()

    # Setup output directories
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    conformations_folder = output_path / "conformations"
    conformations_folder.mkdir(exist_ok=True)

    # Setup logging
    setup_logging(output_path)
    logging.info(f"Starting conformer generation with N={args.n_conformers}")

    # Read input CSV
    df = pd.read_csv(args.csv_path)
    if "smiles" not in df.columns:
        raise ValueError("CSV file must contain 'smiles' column")

    logging.info(f"Loaded {len(df)} SMILES from {args.csv_path}")

    # Process each SMILES
    results = []

    for idx, row in df.iterrows():
        smiles = row["smiles"]
        logging.info(f"Processing molecule {idx + 1}/{len(df)}: {smiles}")

        mols, conf_ids = generate_conformers(smiles, args.n_conformers)

        if not conf_ids:
            logging.warning(f"No conformers generated for molecule {idx}")
            result_row = {"smiles": smiles}
            for i in range(1, args.n_conformers + 1):
                result_row[f"conf_{i}"] = np.nan
            results.append(result_row)
            continue

        # Save conformers
        conformer_paths = save_conformers(
            mols[0], conf_ids, idx, conformations_folder
        )

        # Prepare result row
        result_row = {"smiles": smiles}
        for i, path in enumerate(conformer_paths, 1):
            result_row[f"conf_{i}"] = path

        # Fill missing conformers with NaN if fewer than N were generated
        for i in range(len(conformer_paths) + 1, args.n_conformers + 1):
            result_row[f"conf_{i}"] = np.nan

        results.append(result_row)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    output_csv = output_path / "conformers.csv"
    results_df.to_csv(output_csv, index=False)

    logging.info(f"Results saved to {output_csv}")
    logging.info("Conformer generation completed")


if __name__ == "__main__":
    main()
