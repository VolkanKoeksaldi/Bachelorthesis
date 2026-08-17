import json
import random
import xml.etree.ElementTree as ET

import pandas as pd

from clustering_utils import expand_base_table
from experiment_config import (copy_factor, MESH_PATIENT_COUNT, 
                               MESH_PATIENT_SEED, SOURCE_ROWS, target_rows, experiment_path)


INPUT_PATH = experiment_path("prepared/mesh/desc2026_sample.xml")
OUTPUT_PATH = experiment_path("processed/mesh_terms.csv")


def extract_tree_levels(tree_number):
    """
    Derives the three hierarchical fragmentation keys from a Tree Number.

    Parameters:
        tree_number: MeSH Tree Number
    
    Returns:
        A tuple that contains the top_category, branch_code, and subbranch_code of a Tree Number
    """

    # Splits the Tree Number into its components.
    tree_part = tree_number.split(".")
    top_category = tree_number[0]
    branch_code = tree_part[0]
    subbranch_code = ".".join(tree_part[:2])

    return top_category, branch_code, subbranch_code


def extract_metadata(descriptor_record):
    """
    Extracts descriptor information as metadata.
    The metadata is preserved in the generated dataset for completeness.

    Parameters:
        descriptor_record: A Parsed MeSH DescriptorRecord element
    
    Returns:
        A dictionary that contains scope notes, entry terms, allowable qualifiers,
        and the annotation.
    """

    # Deduplicates and sorts all information on ScopeNote.
    scope_notes = sorted({element.text for element 
                          in descriptor_record.findall("ConceptList/Concept/ScopeNote") 
                          if element.text})

    # Deduplicates and sorts all information on Term Strings for DescriptorRecords as entry_terms.
    entry_terms = sorted({element.text for element 
                          in descriptor_record.findall("ConceptList/Concept/TermList/Term/String")
                          if element.text})

    # Extracts all allowable qualifiers and their information.
    allowable_qualifiers = []
    for allowable_qualifier in (
    descriptor_record.findall("AllowableQualifiersList/AllowableQualifier")):
        
        qualifier_ui = allowable_qualifier.findtext("QualifierReferredTo/QualifierUI")

        qualifier_name = allowable_qualifier.findtext("QualifierReferredTo/QualifierName/String")

        abbreviation = allowable_qualifier.findtext("Abbreviation")

        allowable_qualifiers.append({"qualifier_ui": qualifier_ui,
                                     "qualifier_name": qualifier_name,
                                     "abbreviation": abbreviation})

    annotation = descriptor_record.findtext("Annotation")

    return {"scope_notes": scope_notes, "entry_terms": entry_terms,
            "allowable_qualifiers": allowable_qualifiers,
            "annotation": annotation if annotation else None}


def parse_mesh_terms(xml_path):
    """
    Parses the reduced XML file into one row per Term.

    Tree Numbers are attached to records in the MeSH source XML.
    Every term therefore inherits the lexicographically first Tree Number of its descriptor.
    All alternative Tree Numbers are added as metadata in the column 'all_tree_numbers'.

    Parameters:
        xml_path: The path to the reduced XML file
    
    Returns:
        pd.DataFrame(rows): A DataFrame that contains the parsed terms and their
                            fragmentation attributes
    """

    rows = []

    # Iterparse reads the reduced XML file step by step.
    for _, elem in ET.iterparse(xml_path, events=("end",)):

        # Skips if the element does not have the tag DescriptorRecord.
        if elem.tag != "DescriptorRecord":
            continue

        descriptor_ui = elem.findtext("DescriptorUI")
        descriptor_name = elem.findtext("DescriptorName/String")

        # Deduplicates and sorts the Tree Numbers lexicographically.
        tree_numbers = sorted({tree_element.text.strip() 
                               for tree_element in elem.findall("TreeNumberList/TreeNumber")
                               if tree_element.text and tree_element.text.strip()})

        # DescriptorRecords without a Tree Number are skipped.
        if not tree_numbers:
            elem.clear()
            continue

        # Chooses the first tree number as the canonical tree number representation
        # of the descriptor hierarchy.
        canonical_tree_number = tree_numbers[0]

        top_category, branch_code, subbranch_code = extract_tree_levels(canonical_tree_number)

        # Writes context information as a json for the descriptor record element.
        # This is reused for every term that belongs to this record element.
        metadata_json = json.dumps(extract_metadata(elem), 
                                   ensure_ascii=False, separators=(",", ":"))

        for concept in elem.findall("ConceptList/Concept"):
            concept_ui = concept.findtext("ConceptUI")

            for term in concept.findall("TermList/Term"):
                term_ui = term.findtext("TermUI")
                mesh_term = term.findtext("String")

                # Skips incomplete Terms.
                if not term_ui or not mesh_term:
                    continue

                rows.append({
                    "term_ui": term_ui,
                    "mesh_term": mesh_term,
                    "concept_ui": concept_ui,
                    "descriptor_ui": descriptor_ui,
                    "descriptor_name": descriptor_name,
                    "tree_number": canonical_tree_number,
                    "all_tree_numbers": "|".join(tree_numbers),
                    "top_category": top_category,
                    "branch_code": branch_code,
                    "subbranch_code": subbranch_code,
                    "metadata_json": metadata_json
                })

        elem.clear()

    return pd.DataFrame(rows)


def calculate_item_size(row):
    """
    Calculates the UTF-8 byte size of the fields from the item row.
    Missing or false fields are represented by an empty string before calculating the size.

    Parameters:
        row: DataFrame row that containts the item fields
    
    Returns:
        Size of the item in UTF-8 bytes.
    """

    values = (row["tuple_id"], row["patient_id"], row["mesh_term"], 
              row["term_ui"], row["concept_ui"],
              row["descriptor_ui"], row["metadata_json"])
    
    return len("".join(str(value or "") for value in values).encode("utf-8"))


def process_mesh_terms(input_path, output_path):
    """
    Creates the output directory, parses the XML file, creates patient ids for terms,
    validates the result and stores it as a CSV file.

    Parameters:
        input_path: The path to the reduced XML file
        output_path: The path to the output CSV file

    Returns:
        result: Processed and expanded MeSH DataFrame
    """

    df = parse_mesh_terms(input_path)

    # Ensures that the parsed mesh terms contain enough terms for the base table.
    if len(df) < SOURCE_ROWS:
        raise ValueError(f"Only {len(df)} eligible MeSH terms were parsed, "
                         f"but {SOURCE_ROWS} are required.")

    # Copies SOURCE_ROWS amount of terms from the parsed mesh term dataframe.
    base_df = df.iloc[:SOURCE_ROWS].copy().reset_index(drop=True)

    # Checks whether term_ui is unique for every row.
    if not base_df["term_ui"].is_unique:
        raise ValueError("The selected MeSH source rows contain duplicate TermUI values.")

    # Uses Patient Seed to create patient ids for every row.
    # Multiple terms can belong to the same patient.
    rng = random.Random(MESH_PATIENT_SEED)
    base_df["patient_id"] = [f"P{rng.randint(1, MESH_PATIENT_COUNT):08d}" 
                             for _ in range(len(base_df))]

    # Duplicates the base table using the copy factor.
    # Assigns unique tuple_id to every resulting row.
    result = expand_base_table(base_df, copy_factor, id_column="tuple_id", id_prefix="MT")

    # adds the item size as the weight of each physical item.
    result["item_size_bytes"] = result.apply(calculate_item_size, axis=1)

    # Defines dataframe columns for the generated CSV file.
    column_order = [
        "tuple_id",
        "patient_id",
        "mesh_term",
        "term_ui",
        "concept_ui",
        "descriptor_ui",
        "descriptor_name",
        "tree_number",
        "all_tree_numbers",
        "top_category",
        "branch_code",
        "subbranch_code",
        "metadata_json",
        "item_size_bytes",
        "copy_number"
    ]

    result = result[column_order]

    # Validates the expanded table with expected target rows.
    if len(result) != target_rows:
        raise ValueError(f"Expected {target_rows} rows, but created {len(result)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Distinct terms in base DataFrame: {len(base_df)}")
    print(f"Copy factor: {copy_factor}")
    print(f"Created rows: {len(result)}")
    print(f"Unique tuple IDs: {result['tuple_id'].nunique()}")
    print(f"Saved to: {output_path}")

    return result


if __name__ == "__main__":
    process_mesh_terms(INPUT_PATH, OUTPUT_PATH)