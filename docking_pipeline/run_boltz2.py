import logging
import yaml
from pathlib import Path
from Bio import PDB
import numpy as np
from rdkit import Chem
import argparse

"""Functions for running Boltz model with affinity prediction."""


class FlowListDumper(yaml.SafeDumper):
    pass


def represent_inline_list(dumper, data):
    # If it's a list of lists (like contacts), write in flow style:
    # [[A, 1], [A, 2]]
    if all(isinstance(i, (list, tuple)) for i in data):
        return dumper.represent_sequence(
            "tag:yaml.org,2002:seq", data, flow_style=True
        )
    # Otherwise, use normal representation
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data)


FlowListDumper.add_representer(list, represent_inline_list)


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
    log_file = Path(output_dir) / "boltz2.log"

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


def extract_protein_sequence(pdb_file: Path) -> tuple[str, dict[int, int]]:
    """
    Extract protein sequence from PDB file.

    Parameters
    ----------
    pdb_file : Path
        Path to the PDB file.

    Returns
    -------
    str
        Protein sequence in single-letter code.
    dict[int, int]
        Map between structure residue IDs and sequence indices.

    Raises
    ------
    ValueError
        If missing residues are detected.
    """
    logging.info(f"Extracting sequence from {pdb_file}...")

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    # Check for single chain
    chains = list(structure.get_chains())
    if len(chains) != 1:
        raise ValueError(
            f"Expected single chain, but found {len(chains)} chains in"
            f" {pdb_file}"
        )

    chain = chains[0]

    # Standard amino acid three-letter to one-letter mapping
    aa_dict = {
        "ALA": "A",
        "CYS": "C",
        "ASP": "D",
        "GLU": "E",
        "PHE": "F",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LYS": "K",
        "LEU": "L",
        "MET": "M",
        "ASN": "N",
        "PRO": "P",
        "GLN": "Q",
        "ARG": "R",
        "SER": "S",
        "THR": "T",
        "VAL": "V",
        "TRP": "W",
        "TYR": "Y",
    }
    # Extract sequence with missing residue detection
    sequence = []
    prev_resid = None
    resid_dict = {}

    for residue in chain:
        # Skip heteroatoms and water
        if residue.id[0] != " ":
            continue

        resname = residue.get_resname()
        resid = residue.id[1]

        # Check for missing residues
        if prev_resid is not None:
            gap = resid - prev_resid - 1
            if gap > 0:
                logging.warning(
                    f"Detected gap of {gap} residues between {prev_resid} and "
                    f"{resid}. "
                    f"Inserting {gap} ALA residues."
                )
                sequence.extend(["A"] * gap)

        # Convert three-letter code to one-letter
        if resname not in aa_dict:
            raise ValueError(
                f"Non-standard residue {resname} at position {resid}"
            )

        sequence.append(aa_dict[resname])
        resid_dict[resid] = len(sequence)
        prev_resid = resid

    sequence = "".join(sequence)
    logging.info(f"Extracted sequence of length {len(sequence)}")
    return sequence, resid_dict


def get_binding_site_residues(
    protein_file: Path,
    ligand_file: Path,
    resid_dict: dict[int, int],
    distance_cutoff: float = 3.2,
) -> list[int]:
    """
    Get residue numbers within distance cutoff of reference ligand.

    Parameters
    ----------
    protein_file : Path
        Path to protein PDB file.
    ligand_file : Path
        Path to ligand SDF file.
    resid_dict: dict[int, int],
        Map between structure residue IDs and sequence indices.
    distance_cutoff : float, optional
        Distance cutoff in Angstroms (default: 6.0).

    Returns
    -------
    list[int]
        List of residue numbers within cutoff distance.
    """
    logging.info(
        f"Finding binding site residues within {distance_cutoff} Å..."
    )

    parser = PDB.PDBParser(QUIET=True)
    protein_structure = parser.get_structure("protein", protein_file)
    # Load ligand from SDF file
    suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
    ligand_mol = next(suppl)
    if ligand_mol is None:
        raise ValueError(f"Failed to parse ligand file: {ligand_file}")

    # Get conformer coordinates
    conf = ligand_mol.GetConformer()
    ligand_coords = np.array(
        [conf.GetAtomPosition(i) for i in range(ligand_mol.GetNumAtoms())]
    )

    # Find close residues
    binding_residues = set()

    for model in protein_structure:
        for chain in model:
            for residue in chain:
                if residue.id[0] != " ":  # Skip heteroatoms
                    continue

                for atom in residue:
                    atom_coord = atom.coord
                    distances = np.linalg.norm(
                        ligand_coords - atom_coord, axis=1
                    )

                    if np.any(distances <= distance_cutoff):
                        # NOTE: We expect single chain and no breaks.
                        binding_residues.add(resid_dict[residue.id[1]])
                        break

    binding_residues_list = sorted(list(binding_residues))

    logging.info(
        f"Found {len(binding_residues_list)} binding site residues: "
        f"{binding_residues_list}"
    )

    return binding_residues_list


def prepare_boltz_yaml(
    ligand_smiles: str,
    output_dir: Path,
    sequence: str,
    binding_residues: list[int],
) -> Path:
    """
    Prepare YAML configuration file for Boltz model.

    Parameters
    ----------
    ligand_file : Path
        Path to ligand PDB file.
    output_dir : Path
        Output directory for YAML file.
    sequence : str
        Protein sequence.
    binding_residues : List[int]
        List of binding site residue numbers.

    Returns
    -------
    Path
        Path to created YAML file.
    """
    logging.info("Preparing Boltz YAML configuration...")

    config = {
        "sequences": [
            {
                "protein": {
                    "id": ["A"],
                    "sequence": sequence,
                }
            },
            {
                "ligand": {"id": ["B"], "smiles": ligand_smiles},
            },
        ],
        "constraints": [
            {
                "pocket": {
                    "binder": "B",
                    "contacts": [["A", x] for x in binding_residues],
                    "force": True,
                }
            }
        ],
        "properties": [{"affinity": {"binder": "B"}}],
    }

    yaml_file = output_dir / "boltz_config.yaml"

    with open(yaml_file, "w") as f:
        yaml.dump(
            config,
            f,
            sort_keys=False,  # preserve key order
            default_flow_style=None,
            Dumper=FlowListDumper,
        )

    logging.info(f"YAML configuration saved to {yaml_file}")
    return yaml_file


def generate_smiles_from_sdf(ligand_file: Path) -> str:
    """
    Generate SMILES string from ligand SDF file.

    Parameters
    ----------
    ligand_file : Path
        Path to ligand SDF file.

    Returns
    -------
    str
        SMILES string representation of the ligand.
    """

    logging.info(f"Generating SMILES from {ligand_file}...")

    suppl = Chem.SDMolSupplier(str(ligand_file), removeHs=False)
    mol = next(suppl)
    if mol is None:
        raise ValueError(f"Failed to parse ligand file: {ligand_file}")

    smiles = Chem.MolToSmiles(mol)
    logging.info(f"Generated SMILES: {smiles}")

    return smiles


def main():
    """
    Main function to prepare Boltz configuration from protein and ligand files.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Prepare Boltz2 configuration from protein and ligand "
            "structures"
        )
    )
    parser.add_argument(
        "-p",
        "--protein",
        type=Path,
        required=True,
        help="Path to protein PDB file",
    )
    parser.add_argument(
        "-l",
        "--ligand",
        type=Path,
        required=True,
        help="Path to ligand PDB file",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("boltz_output"),
        help=(
            "Output directory for configuration files (default: boltz_output)"
        ),
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)

    try:
        # Extract protein sequence
        sequence, resid_map = extract_protein_sequence(args.protein)

        # Generate SMILES from ligand
        ligand_smiles = generate_smiles_from_sdf(args.ligand)

        # Get binding site residues
        binding_residues = get_binding_site_residues(
            args.protein,
            args.ligand,
            resid_map,
        )

        # Prepare Boltz YAML configuration
        yaml_file = prepare_boltz_yaml(
            ligand_smiles, args.output_dir, sequence, binding_residues
        )

        logging.info("Configuration preparation completed successfully!")
        logging.info(f"YAML file created at: {yaml_file}")

    except Exception as e:
        logging.error(f"Error during configuration preparation: {e}")
        raise


if __name__ == "__main__":
    main()
