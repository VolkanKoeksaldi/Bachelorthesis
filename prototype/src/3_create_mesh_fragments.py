from pathlib import Path
from experiment_config import MESH_FRAGMENTATION_SCHEMES, target_rows, experiment_path
import pandas as pd
from clustering_utils import (safe_fragment_component, 
                              validate_fragmentation_memberships)


INPUT_PATH = experiment_path("processed/mesh_terms.csv")
FRAGMENT_OUTPUT_PATH = experiment_path("processed/mesh_fragments.csv")
CLUSTER_OUTPUT_PATH = experiment_path("processed/mesh_clusters.csv")

SCHEME_METADATA = {
    "top_category": {
        "relaxation_attribute": "mesh_term",
        "cluster_method": "mesh_taxonomy_prefix_level_1",
    },

    "branch_code": {
        "relaxation_attribute": "mesh_term",
        "cluster_method": "mesh_taxonomy_prefix_level_2",
    },

    "subbranch_code": {
        "relaxation_attribute": "mesh_term",
        "cluster_method": "mesh_taxonomy_prefix_level_3",
    }
}

def choose_cluster_head(group):
    """
    Chooses a cluster head for MeSH term cluster.
    Shallow canonical Tree Number is preferred.
    If there is a tie it is resolved by Tree Number, TermUI, and term text.
    """

    # copies the group
    candidates = group.copy()

    # writes the tree depth by counting dots of a tree number
    candidates["tree_depth"] = candidates["tree_number"].astype(str).str.count(r"\.")

    # sorts the values and takes the first (smallest) tree depth as the cluster head
    head = candidates.sort_values(["tree_depth", "tree_number", "term_ui", "mesh_term"]).iloc[0]

    return head


def create_fragments(df, fragmentation_schemes):
    """
    Generates fragments from different schemes.
    A fragment contains all terms with the same value within the scheme.
    The fragments within one scheme are disjoint.
    Fragments from different schemes may overlap.
    """

    fragment_rows = []

    for scheme in fragmentation_schemes:
        metadata = SCHEME_METADATA[scheme]

        # Groups terms by value in the scheme
        for value, group in df.groupby(scheme, sort=True, dropna=False):

            value = "UNKNOWN" if pd.isna(value) else str(value)

            # filters unique items in the tuple ids
            unique_items = group.drop_duplicates("tuple_id").sort_values("tuple_id")

            # chooses cluster head
            head = choose_cluster_head(group)

            # creates a list of unique tuple ids
            tuple_ids = unique_items["tuple_id"].astype(str).tolist()

            # adds all tree numbers of a term
            all_tree_numbers = sorted({tree_number.strip() 
                                       for stored_values in group["all_tree_numbers"].dropna()
                                       for tree_number in str(stored_values).split("|") 
                                       if tree_number.strip()})

            # adds information to fragments
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
    Loads processed terms.
    Selects one canonical Tree Number per term.
    Generates disjoint fragmentations.
    Validates their memberships.
    Writes a CSV file for resulting fragment table.
    """

    # reads the DataFrame and changes tuple_ids into string
    df = pd.read_csv(input_path, dtype={"tuple_id": "string"})

    # sets the required columns needed for fragments
    required_columns = {
        "tuple_id",
        "mesh_term",
        "term_ui",
        "tree_number",
        "all_tree_numbers",
        *MESH_FRAGMENTATION_SCHEMES,
    }

    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"MeSH file is missing columns: {sorted(missing_columns)}")

    # checks the DataFrame
    if len(df) != target_rows or not df["tuple_id"].is_unique:
        raise ValueError(f"Expected {target_rows} rows with unique tuple_id values.")

    # lists the missing scheme values, axis=1 applies the insa() function to every row, 
    # detecting any missing values
    missing_scheme_values = df[list(MESH_FRAGMENTATION_SCHEMES)].isna().any(axis=1)

    if missing_scheme_values.any():
        raise ValueError(f"{int(missing_scheme_values.sum())} rows are not in a cluster.")

    fragments_df = create_fragments(df, MESH_FRAGMENTATION_SCHEMES)

    # Verifies the completeness of all memberships and the uniqueness of tuple_ids
    validate_fragmentation_memberships(fragments_df, df["tuple_id"], 
                                       MESH_FRAGMENTATION_SCHEMES, "tuple_ids")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fragments_df.to_csv(output_path, index=False)

    cluster_columns = [
        "fragment_id",
        "scheme",
        "relaxation_attribute",
        "value",
        "cluster_head",
        "cluster_head_source_id",
        "cluster_method",
        "fragment_size",
    ]

    # the specific cluster columns are stored as a csv in the cluster output path
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