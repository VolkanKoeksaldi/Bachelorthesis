# Generates imdb_overlaps.csv

from pathlib import Path
from itertools import combinations

import pandas as pd

INPUT_PATH = Path(
    "prototype/output/processed/imdb_fragments.csv"
)

OUTPUT_PATH = Path(
    "prototype/output/processed/imdb_overlaps.csv"
)

UPDATED_INPUT_PATH = Path(
    "prototype/output/reoptimization/imdb_fragments_updates.csv"
)

UPDATED_OUTPUT_PATH = Path(
    "prototype/output/reoptimization/imdb_overlaps_updates.csv"
)

MODES = {
    "baseline": (INPUT_PATH, OUTPUT_PATH),
    "updates": (UPDATED_INPUT_PATH, UPDATED_OUTPUT_PATH),
}

MODE = "baseline"

item_ids_column = "title_ids"

# Wenn True, dann werden auch Fragmente desselben schemas verglichen.
# Es sollten hierbei bei disjunkten fragmentierungen keine zusätzlichen Overlaps entstehen.
# Wenn False, dann werden nur Fragmente aus unterschiedlichen Schemes verglichen
compare_same_scheme = False

def parse_title_ids(value):
    """
    Wandelt String title_ids in ein Set um.
    """

    if pd.isna(value) or value == "":
        return set()

    return {
        title_id.strip() for title_id in str(value).split(",") if title_id.strip()
    }

def prepare_fragments(fragments_df):
    """
    Um Fragmente für Overlaps vorzubereiten.
    """

    prepared = []

    for row in fragments_df.itertuples(index=False):
        title_ids = parse_title_ids(getattr(row, item_ids_column))

        prepared.append({
            "fragment_id": row.fragment_id,
            "scheme": row.scheme,
            "value": row.value,
            "fragment_size": row.fragment_size,
            "title_ids": title_ids
        })

    return prepared

def compute_overlaps(prepared, compare_same_scheme):
    """
    Vergleicht alle Fragmentpaare nach Overlaps.
    """

    overlap_rows = []
    
    # Alle möglichen Fragmentpaare werden hier durchgegangen.
    for f1, f2 in combinations(prepared, 2):
        if (not compare_same_scheme and f1["scheme"] == f2["scheme"]):
            continue
        
        # Schnittmenge berechnen.
        overlap = f1["title_ids"].intersection(f2["title_ids"])

        if overlap:
            overlap_rows.append({
                "fragment_1": f1["fragment_id"],
                "scheme_1": f1["scheme"],
                "value_1": f1["value"],
                "fragment_2": f2["fragment_id"],
                "scheme_2": f2["scheme"],
                "value_2": f2["value"],
                "overlap_size": len(overlap),
                "overlap_title_ids": ",".join(sorted(overlap))
            })

    return pd.DataFrame(overlap_rows)


def process_imdb_overlaps(input_path, output_path, compare_same_scheme):
    fragments_df = pd.read_csv(input_path)

    required_columns = {"fragment_id", "scheme", "value", "fragment_size", "title_ids"}

    missing = (required_columns-set(fragments_df.columns))

    if missing:
        raise ValueError(f"Es fehlen folgende Spalten in dem Fragment: {sorted(missing)}")

    prepared_fragments = prepare_fragments(fragments_df)

    overlaps_df = compute_overlaps(prepared_fragments, compare_same_scheme)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    overlaps_df.to_csv(output_path, index=False)

    print(f"Fragments Anzahl: {len(fragments_df)}")
    print(f"Anzahl Overlap Paare: {len(overlaps_df)}")

    if not overlaps_df.empty:
        print(f"Summe der paarweisen Overlap-Größen: {overlaps_df['overlap_size'].sum()}")

        print("Largest overlap:")
        print(overlaps_df.sort_values("overlap_size", ascending=False).head(10))

    print(f"Datei gespeichert unter {output_path}")

    return overlaps_df

def main():
    if MODE not in MODES:
        raise ValueError(f"Unbekannter Modus: {MODE}")

    selected_input_path, selected_output_path = MODES[MODE]

    print(f"Modus: {MODE}")

    process_imdb_overlaps(
        selected_input_path,
        selected_output_path,
        compare_same_scheme
    )

if __name__ == "__main__":
    main()