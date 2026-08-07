from pathlib import Path
from experiment_config import experiment_path
import pandas as pd

INPUT_PATH = experiment_path("processed/mesh_descriptors_sample.csv")

OUTPUT_PATH = experiment_path("processed/mesh_fragments_sample.csv")

FRAGMENTATION_SCHEMES = ["top_category", "branch_code", "subbranch_code"]

def validate_fragmentation_memberships(fragments_df, expected_desc_ids, expected_schemes):
    """
    Verify that every descriptor belongs to exactly one fragment in each fragmentation scheme.
    """

    # Store all fragment memberships for each descriptor and scheme
    memberships = {
        descriptor_id: {
            scheme: set()
            for scheme in expected_schemes
        }
        for descriptor_id in expected_desc_ids
    }

    # Reconstructs the descriptor to fragment memberships from fragment table
    for row in fragments_df.itertuples(index=False):
        if row.scheme not in expected_schemes:
            continue

        if pd.isna(row.descriptor_ids):
            descriptor_ids = []
        else:
            descriptor_ids = [
                descriptor_id.strip()
                for descriptor_id in str(row.descriptor_ids).split(",")
                if descriptor_id.strip()
            ]

        # records the fragment membership of each descriptor
        for descriptor_id in descriptor_ids:
            # if descriptor id is not already recorded into the memberships array, 
            # then it is added for the scheme.
            if descriptor_id not in memberships:
                memberships[descriptor_id] = {
                    scheme: set()
                    for scheme in expected_schemes
                }

            memberships[descriptor_id][row.scheme].add(row.fragment_id)

    violation_array = []

    # Each descriptor must occur in exactly one fragment per scheme otherwise it is a violation of rules.
    for descriptor_id, scheme_memberships in memberships.items():
        for scheme in expected_schemes:
            fragment_ids = scheme_memberships[scheme]

            if len(fragment_ids) != 1:
                violation_array.append({
                    "descriptor_id": descriptor_id,
                    "scheme": scheme,
                    "fragment_ids": sorted(fragment_ids)
                })

    if violation_array:
        raise ValueError(f"{len(violation_array)} violated assignments.")

    print(f"Each descriptor belongs to exactly one fragment in each "
          f"of the {len(expected_schemes)} schemes.")

def select_canonical_tree_number(df):
    """
    Selects one deterministic Tree Number per descriptor.

    A MeSH descriptor can have multiple Tree Numbers because it can occur at several positions 
    in the hierarchy.
    If all of these Numbers were used for fragment generation, one descriptor could belong to several
    fragments within the same fragmentation scheme.

    The Conflict-Locality-based ILP used in this project assumes that each fragmentation is disjoint.
    Therefore, every descriptor must belong to exactly one fragment in each scheme.
    With the three schemes top_category, branch_code, and subbranch_code, each descriptor belongs to
    exactly three fragments in total.

    In order to satisfy this assumption reproducibly, the lexicographically smallest Tree Number is selected
    as the canonical Tree Number. The remaining Tree Numbers are retained in all_tree_numbers as metadata,
    but not used to construct placement fragments.
    """

    # sorting by descriptor id and tree number ensures that the first row of every descriptor always
    # contains the lexicographically smallest Tree Number. Thus it makes the selection deterministic
    # and independent of original row order
    df = df.sort_values(["descriptor_ui", "tree_number"]).copy()

    def collect_tree_numbers(group):
        """
        Combines preserved tree-number metadata from all rows of a descriptor.
        """

        values = set(group["tree_number"].dropna().astype(str))

        if "all_tree_numbers" in group.columns:
            values.update(tree_number.strip() for stored_values in group["all_tree_numbers"].dropna()
                          for tree_number in str(stored_values).split("|")
                          if tree_number.strip())
        return "|".join(sorted(values))

    tree_number_columns = ["tree_number"]

    if "all_tree_numbers" in df.columns:
        tree_number_columns.append("all_tree_numbers")

    all_tree_numbers = (df.groupby("descriptor_ui", sort=False)[tree_number_columns]
                        .apply(collect_tree_numbers).rename("all_tree_numbers"))

    # Keeps the first row for each descriptor. First rows Tree Number becomes canonical Tree Number
    # to derive the fragmentation schemes.
    canonical_df = (df.drop_duplicates(subset="descriptor_ui",
                                      keep="first")

                                      # Avoids duplicate metadata column when file parse_descriptors
                                      # already preserved all Tree numbers.
                                      .drop(columns=["all_tree_numbers"], errors="ignore")
                                    
                                      # Adds complete Tree Number list to selected row
                                      # validate="one_to_one" verifies that both tables
                                      # contain exactly one row for each descriptor
                                      .merge(
                                          all_tree_numbers,
                                          on="descriptor_ui",
                                          validate="one_to_one"
                                      )
                                      .reset_index(drop=True)
    )

    # Return contains exactly one row per descriptor
    # -> every descriptor has exactly one value for each fragmentation scheme
    return canonical_df

def create_fragments(df, fragmentation_schemes):
    """
    Generates fragments from different schemes.
    A fragment contains all descriptors with the same value within the scheme.
    The fragments within one scheme are disjoint.
    Fragments from different schemes may overlap.
    """

    fragment_rows = []

    for scheme in fragmentation_schemes:
        # Groups descriptors by value in the scheme.
        for value, group in df.groupby(scheme):
            # Collects unique descriptor ids in the fragment
            descriptor_ids = sorted(set(group["descriptor_ui"]))

            # Collects unique descriptor names
            descriptor_names = sorted(set(group["descriptor_name"]))

            # Collects union of all original tree numbers of descriptors contained in this fragment.
            # Metadata value is first split at "|" and the resulting Tree Numbers are deduplicated.
            all_tree_numbers = sorted({tree_number.strip()
                                      for descriptor_tree_numbers in group["all_tree_numbers"].dropna()
                                      for tree_number in str(descriptor_tree_numbers).split("|")
                                      if tree_number.strip()})

            # includes the scheme name to obtain unique fragment_ids
            fragment_id = f"{scheme}_{value}"

            fragment_rows.append({
                "fragment_id": fragment_id,
                "scheme": scheme,
                "value": value,
                "fragment_size": len(descriptor_ids),
                "descriptor_ids": ",".join(descriptor_ids),
                "descriptor_names": "|".join(descriptor_names),
                "all_tree_numbers": "|".join(all_tree_numbers)
            })

    return pd.DataFrame(fragment_rows)

def process(input_path: Path, output_path: Path, fragmentation_schemes):
    """
    Loads processed descriptors.
    Selects one canonical Tree Number per descriptor.
    Generates disjoint fragmentations.
    Validates their memberships.
    Writes a CSV file for resulting fragment table.
    """

    df = pd.read_csv(input_path)

    # Reduces multiple tree number rows to one deterministic placement row for each descriptor.
    df = select_canonical_tree_number(df)

    # Canonical selection must produce exactly one row per descriptor.
    # If there are still duplicate descriptor ids, then fragments within a scheme could overlap.
    # This violates required partitioning property
    if not df["descriptor_ui"].is_unique:
        raise ValueError(
            "Each descriptor must occur exactly once after selecting its canonical Tree Number."
        )

    # Every canonical tree number must provide a valid value for all of the fragmentation schemes.
    # A missing value would prevent the descriptor from being assigned to one fragment in every scheme.
    missing_values = df[fragmentation_schemes].isna().any(axis=1)

    if missing_values.any():
        raise ValueError(f"A total of {int(missing_values.sum())} descriptors do not have a value for each scheme.")
    
    fragments_df = create_fragments(df, fragmentation_schemes)

    # Verifies the completeness of all memberships and the uniqueness
    validate_fragmentation_memberships(fragments_df, set(df["descriptor_ui"]), fragmentation_schemes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fragments_df.to_csv(output_path, index=False)

    print("Canonical descriptor rows:", len(df))
    print("Unique descriptors:", df["descriptor_ui"].nunique())
    print("Created fragments:", len(fragments_df))
    print("\n", fragments_df.head(20))
    print("\n", f"Saved to: {output_path}")


def main():
    process(INPUT_PATH, OUTPUT_PATH, FRAGMENTATION_SCHEMES)
    

if __name__ == "__main__":
    main()