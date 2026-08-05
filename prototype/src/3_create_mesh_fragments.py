from pathlib import Path
import pandas as pd

INPUT_PATH = Path("prototype/output/processed/mesh_descriptors_sample.csv")

OUTPUT_PATH = Path("prototype/output/processed/mesh_fragments_sample.csv")

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
            # if descriptor id is not already recorded into the memberships array, then it is added for the scheme.
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

    print(f"Each descriptor belongs to {len(expected_schemes)} schemes in exactly one fragment.")

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

            # includes the scheme name to obtain unique fragment_ids
            fragment_id = f"{scheme}_{value}"

            fragment_rows.append({
                "fragment_id": fragment_id,
                "scheme": scheme,
                "value": value,
                "fragment_size": len(descriptor_ids),
                "descriptor_ids": ",".join(descriptor_ids),
                "descriptor_names": "|".join(descriptor_names),
            })

    return pd.DataFrame(fragment_rows)

def process(input_path: Path, output_path: Path, fragmentation_schemes):
    """
    Loads processed descriptors, generates fragments, validates their memberships and then writes a CSV file.
    """

    df = pd.read_csv(input_path)

    # processing stage in parse_descriptors must produce one row per descriptor (validate data)
    if not df["descriptor_ui"].is_unique:
        raise ValueError(
            "The descriptor file contains multiple rows per descriptor."
        )

    # Checks whether every descriptor has a value for every selected scheme
    missing_values = df[fragmentation_schemes].isna().any(axis=1)

    if missing_values.any():
        raise ValueError(f"A total of {int(missing_values.sum())} descriptors do not have a value for each scheme.")
    
    fragments_df = create_fragments(df, fragmentation_schemes)

    # Verifies the completeness of all memberships and the uniqueness
    validate_fragmentation_memberships(fragments_df, set(df["descriptor_ui"]), fragmentation_schemes)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fragments_df.to_csv(output_path, index=False)

    print("Input rows:", len(df))
    print("Unique descriptors:", df["descriptor_ui"].nunique())
    print("Created fragments:", len(fragments_df))
    print("\n", fragments_df.head(20))
    print("\n", f"Saved to: {output_path}")


def main():
    process(INPUT_PATH, OUTPUT_PATH, FRAGMENTATION_SCHEMES)
    

if __name__ == "__main__":
    main()