import os
import argparse
import pandas as pd
# from pdb_to_final_json import pdb_to_final_json
from pdb_to_final_json_updated import pdb_to_final_json, load_sasa_csv


def convert_all(pdb_dir, csv_file, output_dir, max_len=500):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_file)

    total = len(df)
    success_count = 0
    skipped_count = 0
    missing_count = 0

    sasa_lookup = load_sasa_csv("Hariboss_TR60_sasa_seqindex.csv")

    for idx, row in df.iterrows():
        pdb_file = row['PDB_FILE']
        # pdb_file = row['Filename']
        chain = row['CHAIN']

        pdb_path = os.path.join(pdb_dir, pdb_file)
        if not os.path.isfile(pdb_path):
            print(f"[WARN] PDB file missing: {pdb_path} - skipping")
            missing_count += 1
            continue

        output_name = pdb_file.replace('.pdb', f"_{chain}.json")
        output_path = os.path.join(output_dir, output_name)

        # Prepare annotation dict
        # Convert Interaction string '[0,1,0,...]' to list if needed
        interaction = row['Interaction']
        if isinstance(interaction, str):
            interaction = eval(interaction)

        annotation = {
            "SEQUENCE": row['SEQUENCE'],
            "Interaction": interaction,
            "SecondaryStructure": row['SecondaryStructure'],
            "Motif_encode": row['Motif_encode'],
            "CHAIN": chain
        }

        # success = pdb_to_json(pdb_path, annotation, output_path, max_len=max_len)
        try:
            # success = pdb_to_final_json(pdb_path, annotation, output_path, max_len=max_len)
            success = pdb_to_final_json(
                pdb_path,
                annotation,
                output_path,
                sasa_lookup,
                max_len=max_len
            )

        except Exception as e:
            print(f"[ERROR] {pdb_file} chain {chain}: {e}")
            success = False

        if success:
            success_count += 1
        else:
            skipped_count += 1

        if (idx+1) % 10 == 0 or (idx+1) == total:
            print(f"Processed {idx+1}/{total} chains. Success: {success_count}, Skipped: {skipped_count}, Missing: {missing_count}")

    print(f"\nConversion completed.")
    print(f"Total chains: {total}")
    print(f"Successfully converted: {success_count}")
    print(f"Skipped (length or errors): {skipped_count}")
    print(f"Missing PDB files: {missing_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch convert PDB + CSV annotations to JSON dataset")
    parser.add_argument("--pdb_dir", required=True, help="Folder with raw PDB files")
    parser.add_argument("--csv", required=True, help="CSV file with annotations")
    parser.add_argument("--out_dir", required=True, help="Output folder for JSON files")
    parser.add_argument("--max_len", type=int, default=500, help="Max RNA length to process")

    args = parser.parse_args()

    convert_all(args.pdb_dir, args.csv, args.out_dir, max_len=args.max_len)
