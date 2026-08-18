from experiment_config import MESH_FRAGMENTATION_SCHEMES, target_rows, experiment_path
import pandas as pd
from clustering_utils import safe_fragment_component, validate_fragmentation_memberships


INPUT_PATH = experiment_path("processed/mesh_terms.csv")
FRAGMENT_OUTPUT_PATH = experiment_path("processed/mesh_fragments.csv")
CLUSTER_OUTPUT_PATH = experiment_path("processed/mesh_clusters.csv")

SCHEME_METADATA = {
    "top_category": {"relaxation_attribute": "mesh_term",
                    "cluster_method": "mesh_taxonomy_prefix_level_1"},

    "branch_code": {"relaxation_attribute": "mesh_term",
                    "cluster_method": "mesh_taxonomy_prefix_level_2"},

    "subbranch_code": {"relaxation_attribute": "mesh_term", 
                       "cluster_method": "mesh_taxonomy_prefix_level_3"}
}

def choose_cluster_head(group):
    """
    Chooses a representative cluster head for MeSH fragments.
    Terms with shallower canonical Tree Numbers are preferred.
    If there is a tie it is resolved lexicographically using Tree Number, TermUI, and term text.

    Parameters:
        group: DataFrame that contains all items assigned to one fragment
    
    Returns:
        head: The selected fragment cluster head
    """

    candidates = group.copy()

    # Calculates the tree depth by counting the separator dots of a tree number.
    candidates["tree_depth"] = candidates["tree_number"].astype(str).str.count(r"\.")

    # Sorts the candidates and takes the first shallowest tree depth as the cluster head.
    head = candidates.sort_values(["tree_depth", "tree_number", "term_ui", "mesh_term"]).iloc[0]

    return head


def create_fragments(df, fragmentation_schemes):
    """
    Creates a complete horizontal fragmentation for each scheme.
    Generates fragments from different fragmentations.
    A fragment contains all tuples with the same value within the scheme.
    Fragments within the same scheme are disjoint, whereas fragments from different schemes
    may overlap.

    Parameters:
        df: DataFrame that contains the processed MeSH items
        fragmentation_schemes: The different columns used to construct fragmentations
    
    Returns:
        A DataFrame that contains one row per generated fragment.
    """

    fragment_rows = []

    # Creates a complete fragmentation for each scheme.
    for scheme in fragmentation_schemes:
        metadata = SCHEME_METADATA[scheme]

        # Groups terms by value in the current scheme.
        for value, group in df.groupby(scheme, sort=True, dropna=False):

            # Missing values are given the String "UNKNOWN".
            value = "UNKNOWN" if pd.isna(value) else str(value)

            # Ensures that every tuple occurs only once in the fragment.
            unique_items = group.drop_duplicates("tuple_id").sort_values("tuple_id")

            head = choose_cluster_head(group)

            tuple_ids = unique_items["tuple_id"].astype(str).tolist()

            # Adds all unique tree numbers that are associated with terms.
            all_tree_numbers = sorted({tree_number.strip() 
                                       for stored_values in group["all_tree_numbers"].dropna()
                                       for tree_number in str(stored_values).split("|") 
                                       if tree_number.strip()})

            fragment_rows.append({
                "fragment_id": (f"{scheme}_{safe_fragment_component(value)}"),
                "scheme": scheme,
                "relaxation_attribute": metadata["relaxation_attribute"],
                "value": value,
                "cluster_head": head["mesh_term"],
                "cluster_head_source_id": head["term_ui"],
                "cluster_method": metadata["cluster_method"],
                "fragment_size": len(tuple_ids),
                "tuple_ids": ",".join(tuple_ids),
                "item_names": "|".join(unique_items["mesh_term"].astype(str)),
                "all_tree_numbers": "|".join(all_tree_numbers)
            })

    return pd.DataFrame(fragment_rows)

def process(input_path, output_path, cluster_output_path):
    """
    Loads processed terms and validates required columns and tuple IDs.
    Generates one fragmentation for every configured scheme.
    Validates their memberships.
    Stores the complete fragment table and reduced cluster table as CSV files.

    Parameters:
        input_path: Processed MeSH term CSV file path
        output_path: Output fragment CSV file path
        cluster_output_path: Path to the reduced cluster CSV file
    
    Returns:
        fragments_df: DataFrame that contains all generated fragments 
    """

    df = pd.read_csv(input_path, dtype={"tuple_id": "string"})

    # Sets the required columns needed for fragments and the cluster-head selection.
    required_columns = {"tuple_id", "mesh_term", "term_ui", "tree_number", "all_tree_numbers",
                        *MESH_FRAGMENTATION_SCHEMES}

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"MeSH file is missing columns: {sorted(missing_columns)}")

    # Verifies that processed input contains exactly one row per unique tuple.
    if len(df) != target_rows or not df["tuple_id"].is_unique:
        raise ValueError(f"Expected {target_rows} rows with unique tuple_id values.")

    # Identifies rows that contain a missing scheme value.
    missing_scheme_values = df[list(MESH_FRAGMENTATION_SCHEMES)].isna().any(axis=1)

    if missing_scheme_values.any():
        raise ValueError(f"{int(missing_scheme_values.sum())} rows are missing at least "
                         "one value in the fragmentation schemes.")

    fragments_df = create_fragments(df, MESH_FRAGMENTATION_SCHEMES)

    # Verifies the completeness of all memberships and the uniqueness of tuple_ids.
    validate_fragmentation_memberships(fragments_df, df["tuple_id"], 
                                       MESH_FRAGMENTATION_SCHEMES, "tuple_ids")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_output_path.parent.mkdir(parents=True, exist_ok=True)

    fragments_df.to_csv(output_path, index=False)

    cluster_columns = [
        "fragment_id",
        "scheme",
        "relaxation_attribute",
        "value",
        "cluster_head",
        "cluster_head_source_id",
        "cluster_method",
        "fragment_size"
    ]

    # Stores the reduced cluster file as a CSV.
    fragments_df[cluster_columns].to_csv(cluster_output_path, index=False)

    print(f"MeSH rows: {len(df)}")
    print(f"Cluster fragments: {len(fragments_df)}")
    print(f"Memberships: {fragments_df['fragment_size'].sum()}")
    print(f"Fragments saved to: {output_path}")
    print(f"Cluster metadata saved to: {cluster_output_path}")

    return fragments_df


def main():
    process(INPUT_PATH, FRAGMENT_OUTPUT_PATH, CLUSTER_OUTPUT_PATH)
    

if __name__ == "__main__":
    main()