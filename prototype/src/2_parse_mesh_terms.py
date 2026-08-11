from pathlib import Path
import json
import random
import xml.etree.ElementTree as ET

import pandas as pd

from clustering_utils import expand_base_table
from experiment_config import copy_factor, MESH_PATIENT_COUNT, MESH_PATIENT_SEED, SOURCE_ROWS, target_rows, experiment_path


INPUT_PATH = experiment_path("prepared/mesh/desc2026_sample.xml")
OUTPUT_PATH = experiment_path("processed/mesh_terms.csv")


def extract_tree_levels(tree_number):
    """
    Derives cluster keys used for MeSH.
    """

    # splits the tree number according to different identifier codes
    tree_part = tree_number.split(".")
    top_category = tree_number[0]
    branch_code = tree_part[0]
    subbranch_code = ".".join(tree_part[:2])

    return top_category, branch_code, subbranch_code


def extract_metadata(descriptor_record):
    """
    Extracts descriptor information not used for placement as metadata.
    """

    # extracts all information on ScopeNote
    scope_notes = sorted({element.text for element in descriptor_record.findall("ConceptList/Concept/ScopeNote") if element.text})

    # extracts all information on Entry Terms
    entry_terms = sorted({element.text for element in descriptor_record.findall("ConceptList/Concept/TermList/Term/String") 
                          if element.text})

    # extracts all allowable qualifiers and their information
    allowable_qualifiers = []
    for allowable_qualifier in descriptor_record.findall("AllowableQualifiersList/AllowableQualifier"):
        qualifier_ui = allowable_qualifier.findtext("QualifierReferredTo/QualifierUI")

        qualifier_name = allowable_qualifier.findtext("QualifierReferredTo/QualifierName/String")

        abbreviation = allowable_qualifier.findtext("Abbreviation")

        allowable_qualifiers.append({
            "qualifier_ui": qualifier_ui,
            "qualifier_name": qualifier_name,
            "abbreviation": abbreviation})

    annotation = descriptor_record.findtext("Annotation")

    return {
        "scope_notes": scope_notes,
        "entry_terms": entry_terms,
        "allowable_qualifiers": allowable_qualifiers,
        "annotation": annotation if annotation else None
    }


def parse_mesh_terms(xml_path):
    """
    Creates one row per MeSH Term element.

    Tree Numbers are attached to records in the original MeSH XML.
    A deterministic canonical Tree Number is inherited by every term of their descriptor.
    All Tree Numbers are added as metadata
    """

    rows = []

    # iterparse reads the data step by step
    for _, elem in ET.iterparse(xml_path, events=("end",)):

        # skips if element does not have the tag DescriptorRecord
        if elem.tag != "DescriptorRecord":
            continue

        descriptor_ui = elem.findtext("DescriptorUI")
        descriptor_name = elem.findtext("DescriptorName/String")

        # extracts the tree numbers of a descriptor and sorts them according to size in ascending order
        tree_numbers = sorted({tree_element.text.strip() for tree_element in elem.findall("TreeNumberList/TreeNumber")
                               if tree_element.text and tree_element.text.strip()})

        # if the descriptor does not have any tree numbers, it is removed
        if not tree_numbers:
            elem.clear()
            continue

        # chooses the first tree number as the canonical tree number
        canonical_tree_number = tree_numbers[0]

        # extracts different levels of the tree
        top_category, branch_code, subbranch_code = extract_tree_levels(canonical_tree_number)

        # writes the context information as a json
        metadata_json = json.dumps(extract_metadata(elem), ensure_ascii=False, separators=(",", ":"))

        # extracts concepts and for each concept the terms
        for concept in elem.findall("ConceptList/Concept"):
            concept_ui = concept.findtext("ConceptUI")

            for term in concept.findall("TermList/Term"):
                term_ui = term.findtext("TermUI")
                mesh_term = term.findtext("String")

                # if TermUI or MeSH Term do not exist skipped.
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
                    "metadata_json": metadata_json,
                })

        elem.clear()

    return pd.DataFrame(rows)


def calculate_item_size(row):
    """
    Calculates the item size by bytes in UTF-8 for a MeSH row.
    """

    values = (row["tuple_id"], row["patient_id"], row["mesh_term"], row["term_ui"], row["concept_ui"],
              row["descriptor_ui"], row["metadata_json"])
    
    return len("".join(str(value or "") for value in values).encode("utf-8"))


def process_mesh_terms(input_path, output_path):
    """
    Creates the output directory, parses the XML file, creates patient ids for terms,
    validates the result and stores it as a CSV file.
    """

    df = parse_mesh_terms(input_path)

    # if the DataFrame of parsed mesh terms is smaller than the maximum Source Rows allowed, error
    if len(df) < SOURCE_ROWS:
        raise ValueError(f"Only {len(df)} eligible MeSH terms were parsed, but {SOURCE_ROWS} are required.")

    # copies Source Rows amount of terms from the parsed mesh term dataframe and resets the index
    base_df = df.iloc[:SOURCE_ROWS].copy().reset_index(drop=True)

    # checks whether term_ui is unique for every row
    if not base_df["term_ui"].is_unique:
        raise ValueError("The selected MeSH source rows contain duplicate TermUI values.")

    # uses Patient Seed to create patient ids for every row in the Base Dataframe
    rng = random.Random(MESH_PATIENT_SEED)
    base_df["patient_id"] = [f"P{rng.randint(1, MESH_PATIENT_COUNT):08d}" for _ in range(len(base_df))]

    # duplicates base table using the copy facotr
    result = expand_base_table(base_df, copy_factor, id_column="tuple_id", id_prefix="MT")
    # calculates item size and adds it to the result
    result["item_size_bytes"] = result.apply(calculate_item_size, axis=1)

    # reorders dataframe columns for the output
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
        "copy_number",
    ]

    result = result[column_order]

    if len(result) != target_rows:
        raise ValueError(f"Expected {target_rows} rows, but created {len(result)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Distinct terms in base DataFrame: {len(base_df)}")
    print(f"Copy factor: {copy_factor}")
    print(f"Created MeSH rows: {len(result)}")
    print(f"Unique tuple IDs: {result['tuple_id'].nunique()}")
    print(f"Saved to: {output_path}")

    return result


if __name__ == "__main__":
    process_mesh_terms(INPUT_PATH, OUTPUT_PATH)