"""
PDB / CIF structure parser.

Extracts Cα atom coordinates and amino acid sequences from PDB or mmCIF files.
"""

from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1 as three_to_one


def parse_structure(
    path: str,
    chain_ids: Optional[list] = None,
) -> tuple:
    """
    Parse a PDB or CIF file and extract per-residue Cα coordinates and sequence.

    Only standard amino acid residues that contain a Cα atom are included.
    Heteroatom residues (HETATM) and water molecules are skipped.

    Args:
        path: Path to a .pdb or .cif / .mmcif file
        chain_ids: Chain IDs to include. If None, all chains are included.

    Returns:
        coords:       np.ndarray, shape (N, 3), float32 — Cα positions in Å
        sequence:     str of length N — one-letter amino acid codes
        chain_labels: np.ndarray, shape (N,), dtype '<U1' — chain ID per residue
    """
    path = Path(path)
    suffix = path.suffix.lower()

    # Detect actual format by content: mmCIF files start with "data_"
    is_cif = suffix in (".cif", ".mmcif")
    if not is_cif:
        with open(path, "r") as fh:
            first_line = fh.readline().strip()
        if first_line.startswith("data_"):
            is_cif = True

    if is_cif:
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    structure = parser.get_structure(path.stem, str(path))

    coords, sequence, chain_labels = [], [], []

    # Use only the first model (index 0)
    model = next(structure.get_models())
    for chain in model.get_chains():
        if chain_ids is not None and chain.id not in chain_ids:
            continue
        for residue in chain.get_residues():
            # Skip heteroatoms (water, ligands) and disordered residues
            het_flag = residue.get_id()[0]
            if het_flag.strip():
                continue
            if "CA" not in residue:
                continue
            try:
                aa = three_to_one(residue.resname)
            except KeyError:
                aa = "X"
            coords.append(residue["CA"].get_vector().get_array())
            sequence.append(aa)
            chain_labels.append(chain.id)

    if not coords:
        raise ValueError(f"No Cα atoms found in {path}")

    return (
        np.array(coords, dtype=np.float32),
        "".join(sequence),
        np.array(chain_labels),
    )
