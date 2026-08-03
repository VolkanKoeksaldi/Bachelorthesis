from pathlib import Path
import pandas as pd

INPUT_PATH = Path("prototype/output/processed/mesh_descriptors_sample.csv")

OUTPUT_PATH = Path("prototype/output/processed/mesh_fragments_sample.csv")

FRAGMENTATION_SCHEMES = ["top_category", "branch_code", "subbranch_code"]

def validate_fragmentation_memberships(fragments_df, expected_desc_ids, expected_schemes):
    """
    Prüft, ob jeder Descriptor in jedem Schema genau einem Fragment angehört.
    """

    memberships = {
        descriptor_id: {
            scheme: set()
            for scheme in expected_schemes
        }
        for descriptor_id in expected_desc_ids
    }

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

        for descriptor_id in descriptor_ids:
            if descriptor_id not in memberships:
                memberships[descriptor_id] = {
                    scheme: set()
                    for scheme in expected_schemes
                }

            memberships[descriptor_id][row.scheme].add(row.fragment_id)

    violation_array = []

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
        raise ValueError(f"{len(violation_array)} ungültige Zuordnungen.")

    print(f"Jeder Descriptor gehört in jedem der {len(expected_schemes)} Schemas genau einem Fragment an.")

def create_fragments(df, fragmentation_schemes):
    """
    Erzeugt Fragmente aus verschiedenen Fragmentierungsschemata
    Idee: Ein Fragment ist eine Menge von Descriptor-IDs, nach Kriterium angeordnet
    
    Beispiel:
    - Descriptors mit top_category = C bilden ein Fragment
    - Descriptors mit branch_code = C23 bilden ein Fragment
    - Descriptors mit subbranch_code = C23.888 bilden ein Fragment
    """

    fragment_rows = []

    for scheme in fragmentation_schemes:
        # Gruppiere Descriptors nach schema.
        for value, group in df.groupby(scheme):
            # Eindeutige Descriptor-IDs im aktuellen Fragment.
            descriptor_ids = sorted(set(group["descriptor_ui"]))

            # Eindeutige Namen
            descriptor_names = sorted(set(group["descriptor_name"]))

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
    df = pd.read_csv(input_path)
    if not df["descriptor_ui"].is_unique:
        raise ValueError(
            "Die Descriptor-Datei enthält mehrere Zeilen pro Descriptor. "
        )

    missing_values = df[fragmentation_schemes].isna().any(axis=1)

    if missing_values.any():
        raise ValueError(f"Es gibt {int(missing_values.sum())} Descriptoren, die nicht für jedes Schema einen Wert haben.")
    
    fragments_df = create_fragments(df, fragmentation_schemes)

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

    