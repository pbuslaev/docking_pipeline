import argparse
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDConfig
from rdkit.Chem import (
    AllChem,
    ChemicalFeatures,
    Descriptors,
    Lipinski,
    rdFingerprintGenerator,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.NullHandler()],
)
logger = logging.getLogger(__name__)


def compute_pharmacophore_fingerprint(mol: Chem.Mol):
    """
    Compute pharmacophore fingerprint for a molecule using RDKit.

    Args:
        mol: RDKit molecule object

    Returns:
        Pharmacophore fingerprint
    """

    fdefName = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
    feats = factory.GetFeaturesForMol(mol)

    # Create a set of feature types for comparison
    feature_types = {}
    for f in feats:
        family = f.GetFamily()
        feature_types[family] = feature_types.get(family, 0) + 1
    return feature_types


def compute_pharmacophore_similarity(
    ligand_smiles: list[str], ref_ligand_smiles: str
) -> list[float]:
    """
    Compute pharmacophore similarity between ligands and a reference ligand.

    Args:
        ligand_smiles: List of SMILES strings for ligands
        ref_ligand_smiles: SMILES string for reference ligand

    Returns:
        List of similarity scores (Jaccard similarity of pharmacophore feature
        sets)
    """
    logger.info("Starting pharmacophore similarity computation...")
    ref_mol = Chem.MolFromSmiles(ref_ligand_smiles)
    ref_features = compute_pharmacophore_fingerprint(ref_mol)

    similarities = []
    for i, smiles in enumerate(ligand_smiles):
        if i % 1000 == 0:
            logger.info(f"Processed {i} ligands for pharmacophore similarity")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            similarities.append(np.nan)
            continue

        mol_features = compute_pharmacophore_fingerprint(mol)

        # Jaccard similarity: intersection / union of feature types
        # Taking into account the count of each feature type
        all_features = set(ref_features.keys()) | set(mol_features.keys())
        if len(all_features) == 0:
            similarity = np.nan
        else:
            intersection = sum(
                min(ref_features.get(f, 0), mol_features.get(f, 0))
                for f in all_features
            )
            union = sum(
                max(ref_features.get(f, 0), mol_features.get(f, 0))
                for f in all_features
            )
            similarity = 1.0 - (intersection / union) if union > 0 else 1.0

        similarities.append(similarity)

    return similarities


def compute_tanimoto_distance(
    mol1: Chem.Mol, mol2: Chem.Mol, fingerprint_type: str = "morgan"
) -> float:
    """
    Compute Tanimoto similarity between two molecules using specified
    fingerprint.

    Parameters
    ----------
    mol1 : Chem.Mol
        First RDKit molecule object
    mol2 : Chem.Mol
        Second RDKit molecule object
    fingerprint_type : str, optional
        Type of molecular fingerprint to use for similarity calculation.
        Options are 'morgan' (default) or 'maccs'
    Returns
    -------
    float
        Tanimoto distance coefficient between the two molecules, ranging
        from 0 to 1, where 1 indicates no similarity and 0 indicates identical
        fingerprints
    Notes
    -----
    - Morgan fingerprints are generated with radius 2 and 2048 bits
    - MACCS keys are 166-bit structural key descriptors

    """
    if fingerprint_type == "maccs":
        fp1 = AllChem.GetMACCSKeysFingerprint(mol1)
        fp2 = AllChem.GetMACCSKeysFingerprint(mol2)
    else:  # morgan
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )

        fp1 = mfpgen.GetFingerprint(mol1)
        fp2 = mfpgen.GetFingerprint(mol2)

    return 1 - DataStructs.TanimotoSimilarity(fp1, fp2)


def compute_ligand_similarities(
    ligand_smiles: list[str], ref_ligand_smiles: str
) -> list[tuple[float, float]]:
    """
    Compute similarity between ligands and a reference ligand using both
    MACCS and Morgan fingerprints.

    Parameters
    ----------
    ligand_smiles : list[str]
        List of SMILES strings for ligands to compare
    ref_ligand_smiles : str
        SMILES string for reference ligand

    Returns
    -------
    list[tuple[float, float]]
        List of tuples containing (MACCS similarity, Morgan similarity) for
        each ligand. Returns (0.0, 0.0) for invalid SMILES strings.

    """
    logger.info("Starting fingerprint similarity computation...")
    ref_mol = Chem.MolFromSmiles(ref_ligand_smiles)
    if ref_mol is None:
        return [(np.nan, np.nan)] * len(ligand_smiles)

    similarities = []
    for i, smiles in enumerate(ligand_smiles):
        if i % 1000 == 0:
            logger.info(f"Processed {i} ligands for fingerprint similarity")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            similarities.append((np.nan, np.nan))
            continue

        maccs_sim = compute_tanimoto_distance(mol, ref_mol, "maccs")
        morgan_sim = compute_tanimoto_distance(mol, ref_mol, "morgan")
        similarities.append((maccs_sim, morgan_sim))

    return similarities


def compute_lipinski_distance(mol1: Chem.Mol, mol2: Chem.Mol) -> float:
    """
    Compute Lipinski-based distance between two molecules.

    Two molecules are considered identical (distance = 0) if they match on all
    Lipinski criteria:
    - Same number of hydrogen bond donors
    - Same number of hydrogen bond acceptors
    - Molecular mass within 10% of each other
    - cLogP within 10% of each other
    - Same number of rotatable bonds

    Parameters
    ----------
    mol1 : Chem.Mol
        First RDKit molecule object
    mol2 : Chem.Mol
        Second RDKit molecule object

    Returns
    -------
    float
        Distance between molecules: 0.0 if all criteria match, 1.0 otherwise
    """

    # Hydrogen bond donors
    hbd1 = Lipinski.NumHDonors(mol1)
    hbd2 = Lipinski.NumHDonors(mol2)

    # Hydrogen bond acceptors
    hba1 = Lipinski.NumHAcceptors(mol1)
    hba2 = Lipinski.NumHAcceptors(mol2)

    # Molecular weight
    mw1 = Descriptors.MolWt(mol1)
    mw2 = Descriptors.MolWt(mol2)

    # cLogP
    logp1 = Descriptors.MolLogP(mol1)
    logp2 = Descriptors.MolLogP(mol2)

    # Rotatable bonds
    rot1 = Lipinski.NumRotatableBonds(mol1)
    rot2 = Lipinski.NumRotatableBonds(mol2)

    # Check all criteria
    if (
        abs(hbd1 - hbd2) < 2
        and abs(hba1 - hba2) < 2
        and abs(mw1 - mw2) <= 0.2 * max(mw1, mw2)
        and (
            abs(logp1 - logp2) <= 0.1 * max(abs(logp1), abs(logp2))
            if max(abs(logp1), abs(logp2)) > 0
            else logp1 == logp2
        )
        and abs(rot1 - rot2) < 3
    ):
        return 0.0
    else:
        return 1.0


def compute_lipinski_similarities(
    ligand_smiles: list[str], ref_ligand_smiles: str
) -> list[float]:
    """
    Compute Lipinski-based similarity between ligands and a reference ligand.

    Parameters
    ----------
    ligand_smiles : list[str]
        List of SMILES strings for ligands to compare
    ref_ligand_smiles : str
        SMILES string for reference ligand

    Returns
    -------
    list[float]
        List of distances (0.0 for identical, 1.0 for different) for each
        ligand. Returns 1.0 for invalid SMILES strings.
    """
    logger.info("Starting Lipinski similarity computation...")
    ref_mol = Chem.MolFromSmiles(ref_ligand_smiles)
    if ref_mol is None:
        return [1.0] * len(ligand_smiles)

    similarities = []
    for i, smiles in enumerate(ligand_smiles):
        if i % 1000 == 0:
            logger.info(f"Processed {i} ligands for Lipinski similarity")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            similarities.append(1.0)
            continue

        distance = compute_lipinski_distance(mol, ref_mol)
        similarities.append(distance)

    return similarities


def select_diverse_ligand_set(
    df: pd.DataFrame,
    ref_ligand_smiles: str,
    n: int = 100,
    m: int = 20,
) -> pd.DataFrame:
    """
    Select a diverse set of ligands including similar and dissimilar compounds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing similarity scores with columns:
        'smiles', 'pharmacophore_sim', 'maccs_sim', 'morgan_sim',
        'lipinski_sim'
    ref_ligand_smiles : str
        SMILES string of the reference ligand
    n : int, optional
        Total number of molecules to select (default: 100)
    m : int, optional
        Minimum number of similar molecules to include (default: 20)

    Returns
    -------
    pd.DataFrame
        Selected ligands with an additional 'similar_to_original' column
        indicating if the compound is similar (True) or dissimilar (False)
        to the reference ligand
    """
    logger.info(f"Selecting diverse ligand set: n={n}, m={m}")

    if m < 10:
        raise ValueError("m must be at least 10")
    if n <= m:
        raise ValueError("n must be greater than m")

    # Create a copy to avoid modifying the original dataframe
    df_work = df.copy()

    selected_similar = []

    # Select top 3 by pharmacophore similarity
    top_pharma = df_work.nsmallest(3, "pharmacophore_sim")
    selected_similar.append(top_pharma)
    df_work = df_work.drop(top_pharma.index)

    # Select top 3 by MACCS similarity
    top_maccs = df_work.nsmallest(3, "maccs_sim")
    selected_similar.append(top_maccs)
    df_work = df_work.drop(top_maccs.index)

    # Select top 3 by Morgan similarity
    top_morgan = df_work.nsmallest(3, "morgan_sim")
    selected_similar.append(top_morgan)
    df_work = df_work.drop(top_morgan.index)

    # Combine selected similar compounds
    similar_df = pd.concat(selected_similar, ignore_index=True)

    # Calculate number of remaining similar compounds needed
    remaining_similar = m - len(similar_df)

    if remaining_similar > 0:
        # Get 10% best from each column
        n_top_10pct = max(1, int(0.1 * len(df_work)))

        top_pharma_10pct = set(
            df_work.nsmallest(n_top_10pct, "pharmacophore_sim").index
        )
        top_maccs_10pct = set(
            df_work.nsmallest(n_top_10pct, "maccs_sim").index
        )
        top_morgan_10pct = set(
            df_work.nsmallest(n_top_10pct, "morgan_sim").index
        )
        top_lipinski_10pct = set(df_work[df_work["lipinski_sim"] == 0].index)

        # Intersection of top 5% across all metrics
        intersection_indices = (
            top_pharma_10pct
            & top_maccs_10pct
            & top_morgan_10pct
            & top_lipinski_10pct
        )

        if len(intersection_indices) >= remaining_similar:
            additional_similar = df_work.loc[
                list(intersection_indices)
            ].sample(n=remaining_similar, random_state=42)
        else:
            # If not enough in intersection, take all and sample from union
            additional_similar = df_work.loc[list(intersection_indices)]
            still_needed = remaining_similar - len(additional_similar)
            union_indices = (
                top_pharma_10pct
                | top_maccs_10pct
                | top_morgan_10pct
                | top_lipinski_10pct
            )
            union_indices = union_indices - intersection_indices
            if len(union_indices) >= still_needed:
                extra_similar = df_work.loc[list(union_indices)].sample(
                    n=still_needed, random_state=42
                )
                additional_similar = pd.concat(
                    [additional_similar, extra_similar]
                )
            else:
                # Take all from union and sample remaining from df_work
                extra_similar = df_work.loc[list(union_indices)]
                additional_similar = pd.concat(
                    [additional_similar, extra_similar]
                )
                still_needed = remaining_similar - len(additional_similar)
                remaining_indices = (
                    set(df_work.index)
                    - top_pharma_10pct
                    - top_maccs_10pct
                    - top_morgan_10pct
                    - top_lipinski_10pct
                )
                if len(remaining_indices) >= still_needed:
                    final_similar = df_work.loc[
                        list(remaining_indices)
                    ].sample(n=still_needed, random_state=42)
                    additional_similar = pd.concat(
                        [additional_similar, final_similar]
                    )

        similar_df = pd.concat(
            [similar_df, additional_similar], ignore_index=True
        )
        df_work = df_work.drop(additional_similar.index)

    # Select dissimilar compounds from worst 90%
    n_dissimilar = n - m
    n_worst_90pct = max(1, int(0.9 * len(df_work)))

    worst_pharma_90pct = set(
        df_work.nlargest(n_worst_90pct, "pharmacophore_sim").index
    )
    worst_maccs_90pct = set(df_work.nlargest(n_worst_90pct, "maccs_sim").index)
    worst_morgan_90pct = set(
        df_work.nlargest(n_worst_90pct, "morgan_sim").index
    )
    worst_lipinski_90pct = set(
        df_work.nlargest(n_worst_90pct, "lipinski_sim").index
    )

    # Union of worst 90% across all metrics
    worst_union = (
        worst_pharma_90pct
        | worst_maccs_90pct
        | worst_morgan_90pct
        | worst_lipinski_90pct
    )

    if len(worst_union) >= n_dissimilar:
        dissimilar_df = df_work.loc[list(worst_union)].sample(
            n=n_dissimilar, random_state=42
        )
    else:
        # Take all from worst union
        dissimilar_df = df_work.loc[list(worst_union)]

    # Add similarity flags
    similar_df["similar_to_original"] = True
    dissimilar_df["similar_to_original"] = False

    # Add reference ligand with zero similarities
    ref_row = pd.DataFrame(
        {
            "smiles": [ref_ligand_smiles],
            "pharmacophore_sim": [0.0],
            "maccs_sim": [0.0],
            "morgan_sim": [0.0],
            "lipinski_sim": [0.0],
            "similar_to_original": [True],
        }
    )

    # Combine all selected ligands
    final_df = pd.concat(
        [ref_row, similar_df, dissimilar_df], ignore_index=True
    )

    logger.info(
        f"Selected {len(final_df)} ligands: {len(similar_df) + 1} similar,"
        f" {len(dissimilar_df)} dissimilar"
    )

    return final_df


def main():
    """
    Main function to compute similarity scores between a library of ligands
    and a reference ligand.
    """
    parser = argparse.ArgumentParser(
        description="Compute ligand similarity scores"
    )
    parser.add_argument(
        "-s",
        "--smiles",
        required=True,
        help="Path to a file containing ligand library with SMILES column",
    )
    parser.add_argument(
        "-l",
        "--ligand",
        required=True,
        help="Path to SDF file containing reference ligand",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Path to output folder",
    )

    args = parser.parse_args()

    log_file = Path(args.output) / "ligand_selection.log"
    # Create output directory if it doesn't exist
    Path(args.output).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )

    # Read library
    logger.addHandler(file_handler)
    logger.info("Starting command-line processing for ligand selection")
    logger.info(f"Reading library SMILES file from {args.smiles}")
    library_df = pd.read_csv(args.smiles, sep="\t")
    ligand_smiles = library_df["SMILES"].tolist()

    # Read reference ligand from SDF
    logger.info(f"Reading reference ligand from {args.ligand}")
    suppl = Chem.SDMolSupplier(args.ligand)
    ref_mol = next(suppl)
    ref_ligand_smiles = Chem.MolToSmiles(ref_mol)

    # Compute similarities
    logger.info("Computing pharmacophore similarities...")
    pharmacophore_sims = compute_pharmacophore_similarity(
        ligand_smiles, ref_ligand_smiles
    )
    logger.info("Done computing pharmacophore similarities.")

    logger.info("Computing fingerprint similarities...")
    fingerprint_sims = compute_ligand_similarities(
        ligand_smiles, ref_ligand_smiles
    )
    maccs_sims = [sim[0] for sim in fingerprint_sims]
    morgan_sims = [sim[1] for sim in fingerprint_sims]
    logger.info("Done computing fingerprint similarities.")

    logger.info("Computing Lipinski similarities...")
    lipinski_sims = compute_lipinski_similarities(
        ligand_smiles, ref_ligand_smiles
    )
    logger.info("Done computing Lipinski similarities.")

    # Create results dataframe
    results_df = pd.DataFrame(
        {
            "smiles": ligand_smiles,
            "pharmacophore_sim": pharmacophore_sims,
            "maccs_sim": maccs_sims,
            "morgan_sim": morgan_sims,
            "lipinski_sim": lipinski_sims,
        }
    )
    results_df.dropna(inplace=True)

    # Save results
    output_file = Path(args.output) / "similarity_scores.csv"
    results_df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Subplot 1: Pharmacophore similarity distribution
    axes[0, 0].hist(
        [x for x in pharmacophore_sims if not np.isnan(x)],
        bins=30,
        edgecolor="black",
    )
    axes[0, 0].set_xlabel("Pharmacophore Similarity")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of Pharmacophore Similarity")

    # Subplot 2: MACCS similarity distribution
    axes[0, 1].hist(
        [x for x in maccs_sims if not np.isnan(x)],
        bins=30,
        edgecolor="black",
    )
    axes[0, 1].set_xlabel("MACCS Similarity")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of MACCS Similarity")

    # Subplot 3: Morgan similarity distribution
    axes[1, 0].hist(
        [x for x in morgan_sims if not np.isnan(x)],
        bins=30,
        edgecolor="black",
    )
    axes[1, 0].set_xlabel("Morgan Similarity")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Distribution of Morgan Similarity")

    # Subplot 4: Lipinski similarity pie chart
    lipinski_counts = pd.Series(lipinski_sims).value_counts()
    axes[1, 1].pie(
        lipinski_counts.values,
        labels=[f"Distance: {k}" for k in lipinski_counts.index],
        autopct="%1.1f%%",
    )
    axes[1, 1].set_title("Lipinski Similarity Distribution")

    plt.tight_layout()
    plot_file = Path(args.output) / "similarity_distributions.png"
    plt.savefig(plot_file, dpi=300)
    logger.info(f"Plot saved to {plot_file}")

    # Select diverse ligand set
    logger.info("Selecting diverse ligand set...")
    selected_df = select_diverse_ligand_set(
        results_df, ref_ligand_smiles, n=100, m=20
    )

    # Save selected ligands
    selected_output = Path(args.output) / "selected_ligands.csv"
    selected_df.to_csv(selected_output, index=False)
    logger.info(f"Selected ligands saved to {selected_output}")

    # Create figure for selected ligands with density plots
    fig_selected, axes_selected = plt.subplots(2, 2, figsize=(12, 10))

    # Separate similar and dissimilar ligands
    similar_ligands = selected_df[selected_df["similar_to_original"]]
    dissimilar_ligands = selected_df[~selected_df["similar_to_original"]]

    # Subplot 1: Pharmacophore similarity distribution
    axes_selected[0, 0].hist(
        results_df["pharmacophore_sim"].dropna(),
        bins=40,
        density=True,
        alpha=0.2,
        edgecolor="black",
        label="Full dataset",
    )
    axes_selected[0, 0].hist(
        similar_ligands["pharmacophore_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Similar",
    )
    axes_selected[0, 0].hist(
        dissimilar_ligands["pharmacophore_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Dissimilar",
    )
    axes_selected[0, 0].set_xlabel("Pharmacophore Similarity")
    axes_selected[0, 0].set_ylabel("Density")
    axes_selected[0, 0].set_title(
        "Distribution of Pharmacophore Similarity (Selected)"
    )
    axes_selected[0, 0].legend()

    # Subplot 2: MACCS similarity distribution
    axes_selected[0, 1].hist(
        results_df["maccs_sim"].dropna(),
        bins=30,
        density=True,
        alpha=0.2,
        edgecolor="black",
        label="Full dataset",
    )
    axes_selected[0, 1].hist(
        similar_ligands["maccs_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Similar",
    )
    axes_selected[0, 1].hist(
        dissimilar_ligands["maccs_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Dissimilar",
    )
    axes_selected[0, 1].set_xlabel("MACCS Similarity")
    axes_selected[0, 1].set_ylabel("Density")
    axes_selected[0, 1].set_title(
        "Distribution of MACCS Similarity (Selected)"
    )
    axes_selected[0, 1].legend()

    # Subplot 3: Morgan similarity distribution
    axes_selected[1, 0].hist(
        results_df["morgan_sim"].dropna(),
        bins=30,
        density=True,
        alpha=0.2,
        edgecolor="black",
        label="Full dataset",
    )
    axes_selected[1, 0].hist(
        similar_ligands["morgan_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Similar",
    )
    axes_selected[1, 0].hist(
        dissimilar_ligands["morgan_sim"].dropna(),
        bins=20,
        density=True,
        alpha=0.5,
        edgecolor="black",
        label="Dissimilar",
    )
    axes_selected[1, 0].set_xlabel("Morgan Similarity")
    axes_selected[1, 0].set_ylabel("Density")
    axes_selected[1, 0].set_title(
        "Distribution of Morgan Similarity (Selected)"
    )
    axes_selected[1, 0].legend()

    # Subplot 4: Lipinski similarity pie chart
    lipinski_selected_counts = selected_df["lipinski_sim"].value_counts()
    axes_selected[1, 1].pie(
        lipinski_selected_counts.values,
        labels=[f"Distance: {k}" for k in lipinski_selected_counts.index],
        autopct="%1.1f%%",
    )
    axes_selected[1, 1].set_title(
        "Lipinski Similarity Distribution (Selected)"
    )

    plt.tight_layout()
    plot_selected_file = (
        Path(args.output) / "selected_similarity_distributions.png"
    )
    plt.savefig(plot_selected_file, dpi=300)
    logger.info(f"Selected ligands plot saved to {plot_selected_file}")


if __name__ == "__main__":
    main()
