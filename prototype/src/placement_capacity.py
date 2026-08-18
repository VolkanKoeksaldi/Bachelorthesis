import math
import pandas as pd

def parse_item_count(value):
    """
    Counts number of distinct item ids in a comma-separated CSV field.
    """
    if pd.isna(value) or value == "":
        return 0

    return len({item_id.strip() for item_id in str(value).split(",") if item_id.strip()})

def calculate_node_capacity(reference_fragments_path, item_ids_column, 
                            num_nodes, capacity_buffer):
    """
    Calculates node capacity from a baseline fragments file.
    The lower bound is calculated as the maximum of the largest fragment weight
    and the average fragment load across available nodes. The configured capacity buffer 
    is then applied to that lower bound.
    """

    fragments_df = pd.read_csv(reference_fragments_path)

    if item_ids_column not in fragments_df.columns:
        raise ValueError(f"Column {item_ids_column} is missing from the "
                         f"fragment file {reference_fragments_path}")

    fragment_weights = fragments_df[item_ids_column].apply(parse_item_count)

    if fragment_weights.empty:
        raise ValueError("There are no fragments in the file.")

    max_fragment_weight = int(fragment_weights.max())
    total_fragment_weight = int(fragment_weights.sum())
    average_node_weight = math.ceil(total_fragment_weight / num_nodes)

    # Every fragment must fit on one node, and the overall fragment weight 
    # should also fit across the available nodes.
    capacity_lower_bound = max(max_fragment_weight, average_node_weight)

    return math.ceil((1 + capacity_buffer) * capacity_lower_bound)