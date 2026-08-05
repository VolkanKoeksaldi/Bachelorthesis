from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import json

INPUT_PATH = Path("prototype/data/raw/mesh/desc2026_sample.xml")

OUTPUT_PATH = Path("prototype/output/processed/mesh_descriptors_sample.csv")

def extract_tree_levels(tree_number: str):
    """
    Cuts the MeSH tree number into multiple hierarchie layers.
    
    Example:
    Tree Number: C14.280.647
    
    Results into:
    top_category = C
    branch_code = C14
    subbranch_code = C14.280
    
    This is afterwards used for multiple fragmentation schemes.
    For example:
    - Fragmentation after top_category
    - Fragmentation after branch_code
    - Fragmentation after subbranch_code
    """

    parts = tree_number.split(".")

    top_category = tree_number[0]

    branch_code = parts[0]

    if len(parts) >= 2:
        subbranch_code = ".".join(parts[:2])
    else:
        subbranch_code = parts[0]

    return top_category, branch_code, subbranch_code

def extract_metadata(descriptor_record):
    """
    Extracts more information from descriptor item.
    -> Scope Notes
    -> Entry Terms and Synonyms
    -> AllowableQualifiers
    -> Annotation
    """

    scope_notes = []

    # extracts all information on ScopeNote
    for scope_note in descriptor_record.findall("ConceptList/Concept/ScopeNote"):
        if scope_note.text:
            scope_notes.append(scope_note.text)

    entry_terms = []

    # extracts all entry terms
    for term in descriptor_record.findall("ConceptList/Concept/TermList/Term/String"):
        if term.text:
            entry_terms.append(term.text)

    allowable_qualifiers = []

    # extracts all allowable qualifiers
    for allowable_qualifier in descriptor_record.findall("AllowableQualifiersList/AllowableQualifier"):
        qualifier_ui = allowable_qualifier.findtext("QualifierReferredTo/QualifierUI")

        qualifier_name = allowable_qualifier.findtext("QualifierReferredTo/QualifierName/String")

        abbreviation = allowable_qualifier.findtext("Abbreviation")

        allowable_qualifiers.append({"qualifier_ui": qualifier_ui,
                                     "qualifier_name": qualifier_name,
                                     "abbreviation": abbreviation})

    annotation = descriptor_record.findtext("Annotation")

    metadata = {"scope_notes": sorted(set(scope_notes)),
                "entry_terms": sorted(set(entry_terms)),
                "allowable_qualifiers": allowable_qualifiers,
                "annotation": annotation if annotation else None
                }

    return metadata


def parse_mesh_descriptors(xml_path: Path):
    """
    Parses a XML file and extracts the information required for the processing stage.

    For each descriptor, the following information is extracted:
    - DescriptorUI: the unique identifier of the descriptor
    - DescriptorName: the name of that descriptor
    - TreeNumber: the position of that descriptor in the hierarchy

    A descriptor may have multiple Tree Numbers. However, only the lexicographically smallest tree number is selected.
    Therefore, each descriptor occurs only max once in the output.
    """

    rows = []

    # iterparse reads the data step by step
    for event, elem in ET.iterparse(xml_path, events=("end",)):

        
        if elem.tag == "DescriptorRecord":

            descriptor_ui = elem.findtext("DescriptorUI")

            descriptor_name = elem.findtext("DescriptorName/String")

            # extracts additional descriptor information and serializes it as a compact JSON
            metadata = extract_metadata(elem)
            metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

            # Approximates the item size as the uTF-8 size of its ID, name and metadata
            item_size_bytes = len(((descriptor_ui or "") + (descriptor_name or "") + metadata_json).encode("utf-8"))

            tree_numbers = sorted({
                tree_elem.text.strip()
                for tree_elem in elem.findall("TreeNumberList/TreeNumber")
                if tree_elem.text and tree_elem.text.strip()
            })

            # descriptors without a tree number are removed from memory.
            if not tree_numbers:
                elem.clear()
                continue

            # deterministic choice:
            # sets the first tree_number as the primary tree_number
            tree_number = tree_numbers[0]

            top_category, branch_code, subbranch_code = extract_tree_levels(
                tree_number
            )

            rows.append({
                "descriptor_ui": descriptor_ui,
                "descriptor_name": descriptor_name,
                "metadata_json": metadata_json,
                "item_size_bytes": item_size_bytes,
                "tree_number": tree_number,
                "tree_number_count": len(tree_numbers),
                "top_category": top_category,
                "branch_code": branch_code,
                "subbranch_code": subbranch_code,
            })
            
            # clears current xml element from memory
            elem.clear()

    return pd.DataFrame(rows)

def process_mesh_descriptors(input_path: Path, output_path: Path):
    """
    Creates the output directory, parses the XML file, validates the result and stores it as a CSV file.
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # parses the xml file and converts the descriptor into tabular form
    df = parse_mesh_descriptors(input_path)

    # Verifies that the selected tree number produced exactly one row per descriptor
    if not df["descriptor_ui"].is_unique:
        duplicate_ids = (
            df.loc[df["descriptor_ui"].duplicated(keep=False), "descriptor_ui"]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"The deterministic tree number selection has generated a duplicated descriptor with {duplicate_ids[:10]}"
        )

    # statistics for validation:
    print("Rows:", len(df))
    print("Unique descriptors:", df["descriptor_ui"].nunique())
    print("Unique tree numbers:", df["tree_number"].nunique())
    print()
    
    # Stores the processed descriptors
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    return df

def main():
    process_mesh_descriptors(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()