import math
from pathlib import Path
import pandas as pd

def parse_item_count(value):
    if pd.isna(value) or value == "":
        return 0
    return len({item_id.strip()
                for item_id in str(value).split(",")
                if item_id.strip()})

def calculate_node_capacity(reference_fragments_path, item_ids_column, num_nodes, capacity_buffer):
    """
    Berechnet W anhand der Baseline Fragment Datei.
    """

    fragments_df = pd.read_csv(reference_fragments_path)

    if item_ids_column not in fragments_df.columns:
        raise ValueError(f"Spalte {item_ids_column} fehlt in der Fragment Datei {reference_fragments_path}")

    fragment_weights = (fragments_df[item_ids_column].apply(parse_item_count))

    if fragment_weights.empty:
        raise ValueError("Es gibt keine Fragmente in der Datei.")

    max_fragment_weight = int(fragment_weights.max())
    total_fragment_weight = int(fragment_weights.sum())

    capacity_lower_bound = max(max_fragment_weight, math.ceil(total_fragment_weight/num_nodes))

    return math.ceil((1 + capacity_buffer) * capacity_lower_bound)