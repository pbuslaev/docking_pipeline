import argparse
import logging
import os
from pathlib import Path

from Bio import PDB
from Bio.PDB import PDBIO, Select
from rdkit import Chem
from rdkit.Geometry import Point3D

"""
Structure downloading and processing utilities.

This module provides functions to download protein structures from the PDB,
extract protein and ligand components, and save them in different formats.
"""

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.NullHandler()],
)
logger = logging.getLogger(__name__)


class ProteinSelect(Select):
    """Select only protein atoms (no water, ligands, or ions)."""

    def accept_residue(self, residue):
        """
        Accept only standard amino acid residues.

        Parameters
        ----------
        residue : Bio.PDB.Residue
            Residue to evaluate

        Returns
        -------
        bool
            True if residue is a standard amino acid
        """
        return residue.get_id()[0] == " "


class LigandSelect(Select):
    """Select only ligand atoms (heteroatoms excluding water)."""

    def accept_residue(self, residue):
        """
        Accept only heteroatoms that are not water.

        Parameters
        ----------
        residue : Bio.PDB.Residue
            Residue to evaluate

        Returns
        -------
        bool
            True if residue is a ligand (heteroatom, not water)
        """
        hetero_flag = residue.get_id()[0]
        resname = residue.get_resname()
        return hetero_flag[0] == "H" and resname != "HOH"


def download_structure(pdb_code: str, output_dir: str) -> str:
    """
    Download structure from PDB in mmCIF format.

    Parameters
    ----------
    pdb_code : str
        PDB code of the structure to download
    output_dir : str
        Directory to save the downloaded structure

    Returns
    -------
    str
        Path to the downloaded CIF file

    Raises
    ------
    Exception
        If download fails
    """
    logger.info(f"Starting download of structure {pdb_code}")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory created/verified: {output_path}")

    pdbl = PDB.PDBList()
    logger.info(f"Retrieving structure {pdb_code} from PDB")
    file_path = pdbl.retrieve_pdb_file(
        pdb_code, pdir=str(output_path), file_format="mmCif", overwrite=True
    )

    # Rename to standard format
    final_path = output_path / f"{pdb_code.lower()}.cif"
    if os.path.exists(file_path) and file_path != str(final_path):
        os.rename(file_path, final_path)
        logger.info(f"Renamed structure file to: {final_path}")

    logger.info(f"Successfully downloaded structure to: {final_path}")
    return str(final_path)


def extract_protein(cif_path: str, output_pdb: str) -> list[str]:
    """
    Extract protein atoms from structure and save as PDB and optionally CIF.

    Parameters
    ----------
    cif_path : str
        Path to input CIF file
    output_pdb : str
        Path to output PDB file

    Returns
    -------
    list[str]
        Paths to saved files (pdb_path, cif_path or None)
    """
    logger.info(f"Starting protein extraction from: {cif_path}")
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_path)
    logger.info("Structure parsed successfully")

    # Collect all protein chains
    protein_chains = []
    for model in structure:
        for chain in model:
            # Check if chain has any protein residues
            has_protein = any(
                ProteinSelect().accept_residue(res) for res in chain
            )
            if has_protein:
                protein_chains.append(chain.get_id())

    logger.info(
        f"Found {len(protein_chains)} protein chain(s): {protein_chains}"
    )

    if len(protein_chains) > 1:
        logger.warning(
            f"Multiple protein chains ({len(protein_chains)}) detected in structure."
        )
        print(
            f"WARNING: Multiple protein chains ({len(protein_chains)}) found: {protein_chains}"
        )
        print("Each chain will be saved separately.")

    # Save each chain separately
    pdb_outputs = []
    for chain_id in protein_chains:
        # Create chain-specific filename
        chain_output_pdb = output_pdb.replace(".pdb", f"_chain_{chain_id}.pdb")

        io = PDBIO()
        io.set_structure(structure)

        # Create a select class for specific chain
        class ChainSelect(ProteinSelect):
            def accept_chain(self, chain):
                return chain.get_id() == chain_id

        io.save(chain_output_pdb, ChainSelect())
        pdb_outputs.append(chain_output_pdb)
        logger.info(f"Chain {chain_id} saved as PDB: {chain_output_pdb}")
        print(f"Chain {chain_id} -> {chain_output_pdb}")

    logger.info("Protein extraction completed")
    return pdb_outputs


# flake8: noqa: C901
def extract_ligand(cif_path: str, output_sdf: str) -> list[str]:
    """
    Extract ligand from structure and save as SDF.

    Parameters
    ----------
    cif_path : str
        Path to input CIF file
    output_sdf : str
        Path to output SDF file

    Returns
    -------
    list[str]
        Path to saved SDF files

    Notes
    -----
    This function extracts all non-water heteroatoms as ligands.
    If multiple ligands are present, they will all be saved in the SDF file.
    """
    logger.info(f"Starting ligand extraction from: {cif_path}")
    parser = PDB.MMCIFParser(QUIET=True)
    structure = parser.get_structure("ligand", cif_path)
    logger.info("Structure parsed successfully")
    output_files = []

    # Extract ligand residues
    ligands = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if LigandSelect().accept_residue(residue):
                    ligands.append(residue)

    logger.info(f"Found {len(ligands)} ligand residue(s)")

    if not ligands:
        # Create empty SDF file
        writer = Chem.SDWriter(output_sdf)
        writer.close()
        logger.info(f"No ligands found, created empty SDF: {output_sdf}")
        return [output_sdf]

    # Write ligands to separate SDF files
    if len(ligands) > 1:
        logger.warning(
            f"Found {len(ligands)} ligands in the structure. Each will be saved separately."
        )
        print(
            f"WARNING: Multiple ligands ({len(ligands)}) found in structure!"
        )
        print("Manual selection will be required for downstream processing.")

    for i, ligand in enumerate(ligands):
        resname = ligand.get_resname()
        # Create separate SDF file for each ligand
        ligand_output = output_sdf.replace(".sdf", f"_{resname}_{i+1}.sdf")
        output_files.append(ligand_output)
        writer = Chem.SDWriter(ligand_output)
        mol = Chem.RWMol()
        atom_map = {}

        for atom in ligand.get_atoms():
            element = atom.element.strip()
            if not element:
                element = atom.get_name()[0]

            rd_atom = Chem.Atom(element)
            idx = mol.AddAtom(rd_atom)
            atom_map[atom.get_serial_number()] = idx

        # Add bonds from CIF file
        try:
            # Get the MMCIF dictionary
            mmcif_dict = PDB.MMCIF2Dict.MMCIF2Dict(cif_path)

            # Check if bond information exists
            if "_chem_comp_bond.atom_id_1" in mmcif_dict:
                comp_id = ligand.get_resname()
                atom1_list = mmcif_dict["_chem_comp_bond.atom_id_1"]
                atom2_list = mmcif_dict["_chem_comp_bond.atom_id_2"]
                bond_order_list = mmcif_dict["_chem_comp_bond.value_order"]
                comp_id_list = mmcif_dict["_chem_comp_bond.comp_id"]

                bond_type_map = {
                    "sing": Chem.BondType.SINGLE,
                    "doub": Chem.BondType.DOUBLE,
                    "trip": Chem.BondType.TRIPLE,
                    "arom": Chem.BondType.AROMATIC,
                }

                # Create atom name to index mapping
                atom_name_map = {
                    atom.get_name().strip(): atom_map[atom.get_serial_number()]
                    for atom in ligand.get_atoms()
                }

                for atom1, atom2, bond_order, comp in zip(
                    atom1_list, atom2_list, bond_order_list, comp_id_list
                ):
                    if (
                        comp == comp_id
                        and atom1.strip() in atom_name_map
                        and atom2.strip() in atom_name_map
                    ):
                        idx1 = atom_name_map[atom1.strip()]
                        idx2 = atom_name_map[atom2.strip()]
                        bond_type = bond_type_map.get(
                            bond_order, Chem.BondType.SINGLE
                        )
                        mol.AddBond(idx1, idx2, bond_type)
            else:
                logger.warning("No bond information found in CIF file.")

            logger.info(f"Added bonds from CIF file for ligand {resname}")
        except Exception as e:
            logger.warning(
                f"Could not add bonds from CIF file: {e}. Creating bonds based on distance."
            )

        # Set 3D coordinates from the CIF file
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom in ligand.get_atoms():
            idx = atom_map[atom.get_serial_number()]
            coord = atom.get_coord()
            conf.SetAtomPosition(idx, Point3D(*[float(x) for x in coord]))
        mol.AddConformer(conf)

        # Basic molecule - without bond information from PDB
        mol = mol.GetMol()
        writer.write(mol)
        writer.close()

        logger.info(
            f"Ligand {i+1}/{len(ligands)}: {resname} saved to {ligand_output}"
        )
        print(f"Ligand {i+1}: {resname} -> {ligand_output}")

    logger.info(f"Ligand extraction completed")
    return output_files


def process_pdb_structure(pdb_code: str, output_dir: str) -> dict:
    """
    Complete pipeline to download and process PDB structure.

    Parameters
    ----------
    pdb_code : str
        PDB code of the structure to process
    output_dir : str
        Directory to save all output files

    Returns
    -------
    dict
        Dictionary with paths to all generated files:
        - 'cif': original structure
        - 'protein_pdb': list of protein chains in PDB format
        - 'ligand_sdf': ligand in SDF format
    """
    logger.info(
        f"Starting structure processing pipeline for PDB code: {pdb_code}"
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdb_lower = pdb_code.lower()

    # Download structure
    cif_file = download_structure(pdb_code, output_dir)

    # Extract protein
    protein_pdb = str(output_path / f"{pdb_lower}_protein.pdb")
    protein_pdbs = extract_protein(cif_file, protein_pdb)

    # Extract ligand
    ligand_sdf = str(output_path / f"{pdb_lower}_ligand.sdf")
    ligand_sdfs = extract_ligand(cif_file, ligand_sdf)

    logger.info(f"Structure processing pipeline completed for {pdb_code}")

    return {
        "cif": cif_file,
        "protein_pdbs": protein_pdbs,
        "ligand_sdfs": ligand_sdfs,
    }


def main():
    """
    Command-line interface for structure downloading and processing.
    """
    parser = argparse.ArgumentParser(
        description="Download and process PDB structures"
    )
    parser.add_argument(
        "-p",
        "--pdb",
        dest="pdb_code",
        type=str,
        required=True,
        help="PDB code of the structure to download",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        type=str,
        required=True,
        help="Output directory for processed files",
    )

    args = parser.parse_args()
    # Configure file logging for this run
    log_file = Path(args.output_dir) / "structure_loading.log"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    )
    logger.addHandler(file_handler)
    logger.info(f"Starting command-line processing for PDB: {args.pdb_code}")
    results = process_pdb_structure(args.pdb_code, args.output_dir)

    print(f"Structure processing complete for {args.pdb_code}")
    print(f"Original structure: {results['cif']}")
    print(f"Protein (PDB): {results['protein_pdbs']}")
    print(f"Ligand (SDF): {results['ligand_sdfs']}")


if __name__ == "__main__":
    main()
