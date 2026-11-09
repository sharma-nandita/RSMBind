import os
import json
import numpy as np
import csv
from Bio.PDB import PDBParser
import ast


# -----------------------------
# Helpers for annotation fields
# -----------------------------
def normalize_interaction(x):
    if isinstance(x, str):
        return ast.literal_eval(x)
    return x


def normalize_motif(x):
    if isinstance(x, str):
        return [int(c) for c in x.strip()]
    return x


# -----------------------------
# SASA loader
# -----------------------------
def load_sasa_csv(sasa_csv_path):
    """
    Returns dict:
    (pdb_chain, chain, pos) -> sasa
    Example key: ('1BYJ_A', 'A', 0)
    """
    sasa_dict = {}

    with open(sasa_csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pdb_chain = row["pdb"].strip()   # e.g. 1BYJ_A
            chain = row["chain"].strip()     # e.g. A
            pos = int(row["pos"])
            sasa = float(row["sasa"])

            sasa_dict[(pdb_chain, chain, pos)] = sasa

    print(f"[INFO] Loaded SASA for {len(sasa_dict)} nucleotides")
    return sasa_dict


# -----------------------------
# Coordinate helpers
# -----------------------------
def get_c3_prime_coord(residue):
    if "C3'" in residue:
        return residue["C3'"].get_coord().tolist()
    elif "C3*" in residue:
        return residue["C3*"].get_coord().tolist()
    elif "C3" in residue:
        return residue["C3"].get_coord().tolist()
    return None


def get_ligand_centroids(structure):
    ligands = []
    coords = []
    for model in structure:
        for chain in model:
            for res in chain:
                if res.id[0] != ' ' and res.resname != 'HOH':
                    atoms = [a.get_coord() for a in res.get_atoms()]
                    if atoms:
                        ligands.append(res.resname)
                        coords.append(np.mean(atoms, axis=0).tolist())
    return ligands, coords

def pdb_to_final_json(
    pdb_path,
    annotation,
    out_json,
    sasa_lookup,
    max_len=500
):
    seq = annotation["SEQUENCE"]
    interaction = normalize_interaction(annotation["Interaction"])
    sec = annotation["SecondaryStructure"]
    motif = normalize_motif(annotation["Motif_encode"])
    chain_id = annotation["CHAIN"]

    # ---- length checks ----
    if len(seq) > max_len:
        print(f"[SKIP] {os.path.basename(pdb_path)} length > {max_len}")
        return False

    if not (len(seq) == len(interaction) == len(sec) == len(motif)):
        print(
            f"[SKIP] Annotation length mismatch in {os.path.basename(pdb_path)} "
            f"(seq={len(seq)}, int={len(interaction)}, sec={len(sec)}, motif={len(motif)})"
        )
        return False

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("X", pdb_path)

    model = structure[0]
    if chain_id not in model:
        print(f"[SKIP] Chain {chain_id} not found in {pdb_path}")
        return False

    chain = model[chain_id]

    residues = [r for r in chain if r.id[0] == ' ']
    if len(residues) != len(seq):
        print(f"[SKIP] Residue/sequence mismatch in {pdb_path}")
        return False

    # ---- identifiers ----
    pdb_chain = os.path.basename(pdb_path).replace(".pdb", "")
    pdb_id = pdb_chain  # keep meta consistent

    # ---- RNA nodes ----
    nodes = []
    coords = []
    missing_sasa = 0

    for i, res in enumerate(residues):
        try:
            c3 = get_c3_prime_coord(res)
            # if c3 is None:
            #     raise ValueError("Missing C3'")
            if c3 is None:
                raise ValueError(
                    f"Missing C3' | "
                    f"sequence_index={i} | "
                    f"PDB_residue_id={res.id} | "
                    f"resname={res.resname} | "
                    f"atoms={[a.get_name() for a in res]}"
                )

            key = (pdb_chain, chain_id, i)
            sasa = sasa_lookup.get(key, 0.0)
            if key not in sasa_lookup:
                missing_sasa += 1

            nodes.append({
                "base": res.get_resname().strip(),
                "pos": i,
                "motif": int(motif[i]),
                "dotbracket": sec[i],
                "label": int(interaction[i]),
                "coord": c3,
                "sasa": float(sasa)
            })

            coords.append(c3)

        except Exception as e:
            print(f"[SKIP] {os.path.basename(pdb_path)} residue {i}: {e}")
            return False

    if missing_sasa > 0:
        print(f"[WARN] {pdb_chain}: missing SASA for {missing_sasa}/{len(nodes)} residues")

    # ---- Ligands ----
    lig_names, lig_coords = get_ligand_centroids(structure)

    # ---- Final JSON ----
    final_json = {
        "rna": {
            "sequence": "".join(seq),
            "coords": coords,
            "nodes": nodes
        },
        "ligand": {
            "names": lig_names,
            "coords": lig_coords
        },
        "meta": {
            "pdb": pdb_id,
            "chain": chain_id
        }
    }

    with open(out_json, "w") as f:
        json.dump(final_json, f, indent=2)

    print(f"[OK] Saved {out_json}")
    return True
