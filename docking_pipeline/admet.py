from admet_ai import ADMETModel
from admet_ai.plot import plot_radial_summary, plot_molecule_svg
from cairosvg import svg2png

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
import logging
import argparse


def setup_logging(output_dir: Path) -> logging.Logger:
    """
    Set up logging configuration for the script.

    Parameters
    ----------
    output_dir : str or Path
        Path to output directory where the log file will be created.

    Returns
    -------
    logging.Logger
        Configured root logger instance with file and console handlers.
    """
    log_file = Path(output_dir) / "admet.log"

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def predict_admet_properties(
    smiles: str, output_folder: str
) -> tuple[dict, str, str]:
    """
    Predict ADMET properties for a molecule and generate visualizations.

    Parameters
    ----------
    smiles : str
        SMILES string of the molecule
    output_folder : str, optional
        Prefix for output SVG files

    Returns
    -------
    dict
        Predicted ADMET properties
    str
        Path to molecule png image
    str
        Path to molecule properties png image
    """
    logging.info(f"Predicting ADMET properties for SMILES: {smiles}")

    # Load ADMET model
    logging.debug("Loading ADMET model")
    model = ADMETModel()

    # Predict properties
    logging.debug("Running predictions")
    predictions = model.predict(smiles=smiles)

    # Generate molecule structure SVG
    logging.debug("Generating molecule structure visualization")
    mol_image = plot_molecule_svg(smiles)

    logging.debug("Generating radial summary visualization")
    radial_image = plot_radial_summary(
        predictions, "drugbank_approved_percentile"
    )

    structure_path = str(Path(output_folder) / "structure.png")
    radial_path = str(Path(output_folder) / "radial.png")

    logging.debug(f"Saving structure image to {structure_path}")
    svg2png(mol_image, write_to=structure_path, dpi=300, scale=8)

    logging.debug(f"Saving radial image to {radial_path}")
    svg2png(radial_image, write_to=radial_path, dpi=300, scale=8)

    important_properties = {
        "BBB": predictions["BBB_Martins_drugbank_approved_percentile"],
        "Non-toxic": predictions["ClinTox_drugbank_approved_percentile"],
        "Solubilty": predictions[
            "Solubility_AqSolDB_drugbank_approved_percentile"
        ],
        "Bioavailability": predictions[
            "Bioavailability_Ma_drugbank_approved_percentile"
        ],
        "hErgSafe": predictions["hERG_drugbank_approved_percentile"],
    }

    logging.info("ADMET prediction completed successfully")
    return (
        important_properties,
        structure_path,
        radial_path,
    )


def analyze_docking_results(
    smiles_list: list[str],
    free_energies: list[float],
    known_binders: list[bool],
    indices: list[int],
    output_folder: str,
) -> tuple[pd.DataFrame, str]:
    """
    Analyze multiple molecules with ADMET predictions and docking results.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings for molecules
    free_energies : list[float]
        List of free energy values for each molecule
    known_binders : list[bool]
        List of flags indicating if molecule is a known binder
    indices : list[int]
        List of molecule indices
    output_folder : str
        Path to output folder for saving results

    Returns
    -------
    pd.DataFrame
        DataFrame with SMILES, known binder flag, index, ADMET properties,
        and free energies
    str
        Path to the generated summary image
    """
    logging.info(f"Analyzing {len(smiles_list)} molecules")

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    results = []
    n_molecules = len(smiles_list)

    # Create figure with subplots
    # Calculate grid dimensions - 3 pairs per row
    pairs_per_row = 3
    n_rows = (
        n_molecules + pairs_per_row - 1
    ) // pairs_per_row  # Ceiling division

    fig, axes = plt.subplots(
        n_rows, pairs_per_row * 2, figsize=(12, 4 * n_rows)
    )

    # Handle edge cases for axes array shape
    if n_rows == 1 and n_molecules == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for i, (smiles, dg, known, idx) in enumerate(
        zip(smiles_list, free_energies, known_binders, indices)
    ):
        logging.debug(f"Processing molecule {i + 1}/{n_molecules}")

        # Calculate position in grid
        row = i // pairs_per_row
        col_offset = (i % pairs_per_row) * 2

        # Create temporary folder for this molecule
        temp_folder = output_path / f"temp_{i}"
        temp_folder.mkdir(exist_ok=True)

        # Predict ADMET properties
        admet_props, structure_path, radial_path = predict_admet_properties(
            smiles, str(temp_folder)
        )

        # Create title
        title = f"Molecule {idx}"
        if known:
            title += " (Known Binder)"
        title += f"\ndG = {dg:.2f}"

        # Plot structure
        structure_img = mpimg.imread(structure_path)
        axes[row, col_offset].imshow(structure_img)
        axes[row, col_offset].axis("off")
        axes[row, col_offset].set_title(title, fontsize=12, fontweight="bold")

        # Plot radial summary
        radial_img = mpimg.imread(radial_path)
        axes[row, col_offset + 1].imshow(radial_img)
        axes[row, col_offset + 1].axis("off")

        # Collect results
        result_row = {
            "smiles": smiles,
            "known_binder": known,
            "index": idx,
            "free_energy": dg,
            **admet_props,
        }
        results.append(result_row)

    # Hide unused subplots
    for i in range(n_molecules, n_rows * pairs_per_row):
        row = i // pairs_per_row
        col_offset = (i % pairs_per_row) * 2
        axes[row, col_offset].axis("off")
        axes[row, col_offset + 1].axis("off")

    plt.tight_layout()
    summary_path = str(output_path / "docking_summary.png")
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Summary image saved to {summary_path}")

    # Create DataFrame
    df = pd.DataFrame(results)

    return df, summary_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze docking results with ADMET predictions"
    )
    parser.add_argument(
        "-c",
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file with ranked compounds",
    )
    parser.add_argument(
        "-n",
        "--n-molecules",
        type=int,
        default=5,
        help="Number of top molecules to analyze (default: 5)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output folder for results",
    )

    args = parser.parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    # Setup logging
    logger = setup_logging(Path(args.output))

    # Read CSV file
    df = pd.read_csv(args.csv)

    # Take top N molecules
    df_top = df.head(args.n_molecules)
    # Add molecule with index 0 if not already in top N
    if 0 not in df_top["idx"].values:
        mol_0 = df[df["idx"] == 0]
        df_top = pd.concat([df_top, mol_0], ignore_index=True)

    # Add known_binder column based on idx
    df_top["known_binder"] = df_top["idx"] == 0
    # Analyze results
    results_df, summary_path = analyze_docking_results(
        smiles_list=df_top["smiles"].tolist(),
        free_energies=df_top["best_free_energy"].tolist(),
        known_binders=df_top["known_binder"].tolist(),
        indices=df_top["idx"].tolist(),
        output_folder=args.output,
    )

    # Save results
    output_csv = str(Path(args.output) / "admet_analysis.csv")
    results_df.to_csv(output_csv, index=False)
    logger.info(f"Results saved to {output_csv}")
