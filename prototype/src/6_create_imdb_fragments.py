from pathlib import Path
import pandas as pd
from experiment_config import IMDB_FRAGMENTATION_SCHEMES, target_rows, experiment_path
from clustering_utils import safe_fragment_component, validate_fragmentation_memberships


input_path = experiment_path("processed/imdb_titles.csv")
membership_output_path = experiment_path("processed/imdb_fragment_memberships.csv")

fragment_output_path = experiment_path("processed/imdb_fragments.csv")

cluster_output_path = experiment_path("processed/imdb_clusters.csv")


# Describes how the clusters of each fragmentation scheme are generated.
SCHEME_METADATA = {
    "title_type": {
        "relaxation_attribute": "title_type",
        "cluster_method": "categorical_value_cluster",
    },
    "decade": {
        "relaxation_attribute": "start_year",
        "cluster_method": "numeric_decade_cluster",
    },
    "primary_genre": {
        "relaxation_attribute": "primary_genre",
        "cluster_method": "categorical_value_cluster",
    }
}

def create_memberships(titles_df):
    """
    Creates fragment memberships for each title based on the title type, 
    decade, and primary genre.
    Each title then belongs to exactly one fragment in each scheme.
    """

    membership_rows = []

    # Iterates over all titles without including the DataFrame index.
    for row in titles_df.itertuples(index=False):

        # Creates one membership for every fragmentation scheme.
        for scheme in IMDB_FRAGMENTATION_SCHEMES:
            # gets the value of scheme for every row
            # and compares with defined scheme
            value = getattr(row, scheme)

            # checks the extracted values and assigns missing fragmentation values to
            # scheme "UNKNOWN"
            value = "UNKNOWN" if pd.isna(value) else str(value)

            # Makes the value safe for use as part of a fragment ID.
            safe_value = safe_fragment_component(value)

            # appends information for the membership information and file
            membership_rows.append({
                "fragment_id": f"{scheme}_{safe_value}",
                "scheme": scheme,
                "relaxation_attribute": (SCHEME_METADATA[scheme]["relaxation_attribute"]),
                "value": value,
                "cluster_head": value,
                "cluster_method": (SCHEME_METADATA[scheme]["cluster_method"]),
                "title_id": row.title_id,
                "primary_title": row.primary_title
            })

    return pd.DataFrame(membership_rows)


def create_fragments(memberships):
    """
    Title memberships are grouped into fragments.
    """

    fragment_rows = []

    # All cluster metadata is included in the grouping so that it remains
    # explicitly associated with the resulting fragment.
    group_columns = [
        "fragment_id",
        "scheme",
        "relaxation_attribute",
        "value",
        "cluster_head",
        "cluster_method"
    ]

    # Each group represents here one fragment.
    for keys, group in memberships.groupby(group_columns, sort=True):
        # extracts all information from fragment keys
        fragment_id, scheme, relaxation_attribute, value, cluster_head, cluster_method = keys

        # Ensures that each title occurs only once within a fragment and
        # orders the titles by their IDs.
        unique_titles = group.drop_duplicates(subset=["title_id"]).sort_values("title_id")

        # Transforms the title_id column into a Python list of string values.
        title_ids = unique_titles["title_id"].astype(str).tolist()

        # Stores the title names of the primary titles as a list of string values.
        item_names = unique_titles["primary_title"].astype(str).tolist()

        # adds fragmentation rows
        fragment_rows.append({
            "fragment_id": fragment_id,
            "scheme": scheme,
            "relaxation_attribute": relaxation_attribute,
            "value": value,
            "cluster_head": cluster_head,

            # IMDb does not have separate source items that represent the cluster head
            "cluster_head_source_id": "",
            "cluster_method": cluster_method,
            "fragment_size": len(title_ids),
            "title_ids": ",".join(title_ids),
            "item_names": "|".join(item_names)
        })

    return pd.DataFrame(fragment_rows)
          

def process_imdb_fragments(input_path, membership_output_path, 
                           fragment_output_path, cluster_output_path):
    """
    Loads the prepared titles and creates their fragment memberships.
    Afterwards the memberships are grouped into fragments and both result tables are stored.
    """

    # Reads title_id as a string
    titles_df = pd.read_csv(input_path, dtype={"title_id": "string"})

    # Ensures that the input contains the expected number of rows.
    if len(titles_df) != target_rows:
        raise ValueError(f"Expected {target_rows} IMDb rows, but found {len(titles_df)}.")

    # Each row must have a unique title_id
    if not titles_df["title_id"].is_unique:
        raise ValueError("The prepared IMDb data contains duplicate title_id values.")

    required_columns = {
        "title_id",
        "primary_title",
        *IMDB_FRAGMENTATION_SCHEMES,
    }

    missing_columns = required_columns - set(titles_df.columns)

    if missing_columns:
        raise ValueError(f"The prepared IMDb file is missing the following columns: "
                         f"{sorted(missing_columns)}")

    memberships_df = create_memberships(titles_df)
    fragments_df = create_fragments(memberships_df)

    # Validates that every title belongs to exactly one fragment in each fragmentation scheme
    validate_fragmentation_memberships(fragments_df, expected_item_ids=titles_df["title_id"],
                                       expected_schemes=IMDB_FRAGMENTATION_SCHEMES, 
                                       item_ids_column="title_ids",)

    membership_output_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_output_path.parent.mkdir(parents=True, exist_ok=True)

    memberships_df.to_csv(membership_output_path, index=False)
    fragments_df.to_csv(fragment_output_path, index=False)

    # Stores a separate table containing only the cluster data
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

    fragments_df[cluster_columns].to_csv(cluster_output_path, index=False)

    print(f"IMDb rows: {len(titles_df)}")
    print(f"Unique title IDs: {titles_df['title_id'].nunique()}")
    print(f"Number of memberships: {len(memberships_df)}")
    print(f"Number of cluster fragments: {len(fragments_df)}")
    print(f"Fragments saved to: {fragment_output_path}")
    print(f"Cluster metadata saved to: {cluster_output_path}")

    return memberships_df, fragments_df

def main():
    process_imdb_fragments(input_path, membership_output_path, 
                           fragment_output_path, cluster_output_path)


if __name__ == "__main__":
    main()