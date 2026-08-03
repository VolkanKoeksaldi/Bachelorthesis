from pathlib import Path
from itertools import combinations

import pandas as pd

INPUT_PATH = Path("prototype/output/processed/mesh_fragments_sample.csv")

OUTPUT_PATH = Path("prototype/output/processed/mesh_overlaps_sample.csv")

UPDATED_INPUT_PATH = Path("prototype/output/reoptimization/mesh_fragments_sample_updates.csv")
UPDATED_OUTPUT_PATH = Path("prototype/output/reoptimization/mesh_overlaps_sample_updates.csv")

MODES = {
    "baseline": (INPUT_PATH, OUTPUT_PATH),
    "updates": (UPDATED_INPUT_PATH, UPDATED_OUTPUT_PATH),
}

MODE = "baseline"

def parse_item_ids(item_ids_string):
    """
    Wandelt gespeicherte Descriptor-IDs aus CSV wieder in ein Python Set
    
    CSV enthält IDS als String "D00001,D00002" usw
    
    Für die Overlap Berechnung bauen wir daraus dann ein Set.
    """

    if pd.isna(item_ids_string) or item_ids_string=="":
        return set()
    
    return set(item_ids_string.split(","))

def compute_overlaps(fragments_df, item_ids_column, overlap_ids_column):
    """
    Berechnet Überschneidung zwischen Fragmenten
    Hierbei werden Fragmente aus untersch. Schemata verglichen.
    """

    overlap_rows = []

    # Zeilen werden in Dictionaries umgewandelt
    fragments = []

    for _, row in fragments_df.iterrows():
        fragments.append({
            "fragment_id": row["fragment_id"],
            "scheme": row["scheme"],
            "value": row["value"],
            "item_ids": parse_item_ids(row[item_ids_column])
        })
    
    # Alle möglichen Fragmentpaare werden hier durchgegangen.
    for f1, f2 in combinations(fragments, 2):
        if f1["scheme"] == f2["scheme"]:
            continue
        
        # Schnittmenge berechnen.
        overlap = f1["item_ids"].intersection(f2["item_ids"])

        if overlap:
            overlap_rows.append({
                "fragment_1": f1["fragment_id"],
                "scheme_1": f1["scheme"],
                "value_1": f1["value"],
                "fragment_2": f2["fragment_id"],
                "scheme_2": f2["scheme"],
                "value_2": f2["value"],
                "overlap_size": len(overlap),
                overlap_ids_column: ",".join(sorted(overlap))
            })

    return pd.DataFrame(overlap_rows)

def process_overlaps(input_path: Path, output_path: Path, item_ids_column: str, overlap_ids_column: str):
    fragments_df = pd.read_csv(input_path)
    overlaps_df = compute_overlaps(fragments_df, item_ids_column, overlap_ids_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlaps_df.to_csv(output_path, index=False)
    print("Input fragments:", len(fragments_df))
    print("Computed overlaps:", len(overlaps_df))
    print(f"Saved to; {output_path}")

    return overlaps_df

def main():

    if MODE not in MODES:
        raise ValueError(f"Unbekannter Modus wurde gestartet: {MODE}")

    input_path, output_path = MODES[MODE]
    print("\nModus: ", MODE)
    overlaps_df = process_overlaps(input_path, output_path, item_ids_column="descriptor_ids", overlap_ids_column="overlap_descriptor_ids")

    

if __name__ == "__main__":
    main()