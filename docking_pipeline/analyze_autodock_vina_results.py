import argparse
import pandas as pd
from pathlib import Path
import json
from rdkit import Chem
import logging

"""
Script to analyze AutoDock Vina docking results and generate plots.
"""

import matplotlib.pyplot as plt


def setup_logging(output_dir):
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
    log_file = Path(output_dir) / "docking_analysis.log"

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


def extract_free_energy_from_sdf(sdf_path):
    """
    Extract the lowest free energy from an SDF file containing meeko
    properties.

    Parameters
    ----------
    sdf_path : str or Path
        Path to the SDF file

    Returns
    -------
    tuple of (float or None, Mol or None)
        Lowest free energy and corresponding molecule, or (None, None)
        if parsing fails
    """
    if not Path(sdf_path).exists():
        logging.warning(f"SDF file not found: {sdf_path}")
        return None, None

    supplier = Chem.SDMolSupplier(str(sdf_path))
    best_energy = 100000000
    best_mol = None

    for mol in supplier:
        if mol is None:
            continue

        if mol.HasProp("meeko"):
            meeko_data = mol.GetProp("meeko")
            try:
                meeko_dict = json.loads(meeko_data)
                free_energy = meeko_dict.get("free_energy")

                if free_energy is not None and free_energy < best_energy:
                    best_energy = free_energy
                    best_mol = mol
            except Exception:
                logging.warning(f"Failed to parse meeko data from {sdf_path}")
                continue

    return (best_energy, best_mol) if best_mol is not None else (None, None)


def process_docking_results(input_csv, output_dir) -> pd.DataFrame:
    """
    Process docking results and extract best poses.

    Parameters
    ----------
    input_csv : str or Path
        Path to input CSV file
    output_dir : str or Path
        Path to output directory

    Returns
    -------
    pd.DataFrame
        DataFrame with best results
    """
    logging.info(f"Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    logging.info(f"Processing {len(df)} molecules")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    best_poses_dir = output_path / "best_poses"
    best_poses_dir.mkdir(exist_ok=True)

    results = []

    for idx, row in df.iterrows():
        best_energy = 100000000000
        best_mol = None
        best_pose_col = None

        # Check all dock columns
        for col in ["dock_1", "dock_2", "dock_3", "dock_4", "dock_5"]:
            if pd.notna(row[col]):
                energy, mol = extract_free_energy_from_sdf(row[col])
                if energy is not None and energy < best_energy:
                    best_energy = energy
                    best_mol = mol
                    best_pose_col = col

        if best_mol is not None:
            # Save best pose
            output_file = best_poses_dir / f"molecule_{idx}_best.sdf"
            writer = Chem.SDWriter(str(output_file))
            writer.write(best_mol)
            writer.close()

            logging.debug(
                f"Molecule {idx}: best energy = {best_energy:.2f} kcal/mol "
                f"from {best_pose_col}"
            )

            results.append(
                {
                    "idx": idx,
                    "smiles": row["smiles"],
                    "similar_to_original": row["similar_to_original"],
                    "best_free_energy": best_energy,
                    "best_pose_path": str(output_file),
                }
            )
        else:
            logging.warning(f"No valid poses found for molecule {idx}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("best_free_energy").reset_index(
        drop=True
    )
    logging.info(f"Successfully processed {len(results_df)} molecules")
    return results_df


def create_ranking_plot(results_df, output_dir):
    """
    This function generates a scatter plot showing the relationship between
    molecule rank and free energy, with color coding to distinguish between
    the original molecule, similar molecules, and different molecules. The plot
    includes statistics about similar molecules in the top 5 and top 10
    positions.

    Parameters
    ----------
    results_df : pandas.DataFrame
        DataFrame containing docking results with columns:
        - 'best_free_energy': float, binding free energy values
        - 'idx': int, molecule index (0 for original molecule)
        - 'similar_to_original': bool, similarity flag
    output_dir : str or pathlib.Path
        Path to the directory where the plot will be saved

    Returns
    -------
    None
        The function saves the plot to 'ranking_plot.png' in the output
        directory and logs statistics about the ranking results.

    Notes
    -----
    The plot uses the following color scheme:
    - Black: Original molecule (idx == 0)
    - Magenta: Molecules similar to the original
    - Cyan: Molecules different from the original

    The function logs the following information:
    - Rank of the original molecule
    - Number of similar molecules in the top 5
    - Number of similar molecules in the top 10

    Examples
    --------
    >>> create_ranking_plot(results_df, '/path/to/output')
    INFO:root:Original molecule rank: 3
    INFO:root:Similar molecules in top 5: 2
    INFO:root:Similar molecules in top 10: 5
    INFO:root:Plot saved to /path/to/output/ranking_plot.png
    """
    # Sort by free energy
    sorted_df = results_df.sort_values("best_free_energy").reset_index(
        drop=True
    )
    sorted_df["rank"] = range(1, len(sorted_df) + 1)

    # Determine colors
    colors = []
    for idx, similar in zip(
        sorted_df["idx"], sorted_df["similar_to_original"]
    ):
        if idx == 0:
            colors.append("black")
        elif similar:
            colors.append("magenta")
        else:
            colors.append("cyan")

    # Find original molecule rank
    original_rank = (
        sorted_df[sorted_df["idx"] == 0]["rank"].values[0]
        if 0 in sorted_df["idx"].values
        else None
    )

    # Count similar molecules in top 5 and top 10
    top5 = sorted_df.head(5)
    top10 = sorted_df.head(10)
    similar_top5 = top5[top5["similar_to_original"]].shape[0]
    similar_top10 = top10[top10["similar_to_original"]].shape[0]

    logging.info(f"Original molecule rank: {original_rank}")
    logging.info(f"Similar molecules in top 5: {similar_top5}")
    logging.info(f"Similar molecules in top 10: {similar_top10}")

    # Create plot
    plt.figure(figsize=(12, 6))

    # Plot by category for legend
    for color, label in [
        ("black", "Original"),
        ("magenta", "Similar"),
        ("cyan", "Different"),
    ]:
        mask = [c == color for c in colors]
        plt.scatter(
            sorted_df["rank"][mask],
            sorted_df["best_free_energy"][mask],
            c=color,
            label=label,
            s=50,
            alpha=0.7,
        )

    plt.xlabel("Molecule Rank", fontsize=12)
    plt.ylabel("Free Energy (kcal/mol)", fontsize=12)

    title = "Docking Results Ranking"
    if original_rank is not None:
        title += f"\nOriginal Molecule Rank: {original_rank}"
    title += (
        f" | Similar in Top 5: {similar_top5} | "
        f"Similar in Top 10: {similar_top10}"
    )
    plt.title(title, fontsize=14)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = Path(output_dir) / "ranking_plot.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    logging.info(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AutoDock Vina docking results"
    )
    parser.add_argument(
        "-d",
        "--docking_poses",
        required=True,
        help="Path to CSV file with docked structures",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to output directory"
    )

    args = parser.parse_args()

    # Create output directory first
    Path(args.output).mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logging(args.output)

    logging.info("Starting docking results analysis")
    logging.info(f"Input file: {args.docking_poses}")
    logging.info(f"Output directory: {args.output}")

    logging.info("Processing docking results...")
    results_df = process_docking_results(args.docking_poses, args.output)

    # Save results dataframe
    output_csv = Path(args.output) / "best_results.csv"
    results_df.to_csv(output_csv, index=False)
    logging.info(f"Results saved to {output_csv}")

    logging.info("Creating ranking plot...")
    create_ranking_plot(results_df, args.output)

    logging.info("Analysis complete!")


if __name__ == "__main__":
    main()
