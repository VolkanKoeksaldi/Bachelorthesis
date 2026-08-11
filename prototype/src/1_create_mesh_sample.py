from pathlib import Path
import copy
import random
import xml.etree.ElementTree as ET
from experiment_config import MESH_SAMPLE_SEED, SOURCE_ROWS, experiment_path

source_dir = Path(__file__).resolve().parent
prototype_dir = source_dir.parent

# Input and output paths for MeSH dataset
INPUT_PATH = prototype_dir / "data" / "raw"/ "mesh" / "desc2026.xml"
OUTPUT_PATH = experiment_path("prepared/mesh/desc2026_sample.xml")


def iter_eligible_terms(input_path):
    """
    Iterates through XML and returns unique eligible TermUI values.
    """

    seen_term_ids = set()

    # iterates through every record
    for _, record in ET.iterparse(input_path, events=("end",)):
        if record.tag != "DescriptorRecord":
            continue

        # checks after the tree number and extacts elements that have Tree Numbers
        has_tree_number = any(
            element.text and element.text.strip()
            for element in record.findall("TreeNumberList/TreeNumber")
        )

        # for terms that have Tree Numbers following information is extracted from the records
        if has_tree_number:
            for term in record.findall("ConceptList/Concept/TermList/Term"):
                term_ui = term.findtext("TermUI")
                term_text = term.findtext("String")

                # if one of the information is missing or TermUI is already extracted, skip
                if not term_ui or not term_text or term_ui in seen_term_ids:
                    continue

                seen_term_ids.add(term_ui)

                # Pauses execution and makes the function a generator where it can resume here later on.
                yield term_ui

        # clears the records for storage saving.
        record.clear()

def select_term_ids(input_path, required_terms, seed):
    """
    Selects a deterministic uniform sample of TermUI values.
    """

    rng = random.Random(seed)
    array = []
    # counts the eligible item amount
    eligible_count = 0

    # iterates through every term_ui from the eligible terms in input file
    for term_ui in iter_eligible_terms(input_path):
        eligible_count += 1

        # the amount of samples data needs to be as big as the required terms
        if len(array) < required_terms:
            array.append(term_ui)
            continue

        # chooses a random replacement index from 0 to eligible_count - 1
        replacement_index = rng.randrange(eligible_count)

        # if the index is in the amount of required_terms, then term_ui is placed in the array of sampled data
        # in position replacement_index
        if replacement_index < required_terms:
            array[replacement_index] = term_ui

    if eligible_count < required_terms:
        raise ValueError(f"Only {eligible_count} eligible MeSH terms were found, but {required_terms} are required.")

    return set(array), eligible_count   

def write_selected_records(input_path, output_path, selected_term_ids):
    """
    Writes complete descriptor context but retains only sampled Term elements.
    """

    # starts parsing and yields element from start to end events, therefore not loading the whole file at once
    context = ET.iterparse(input_path, events=("start", "end"))
    # takes the first event from parser, in order to get original document root
    _, root = next(context)
    # creates new empty root with same tag name and attributes like the original XML
    # where terms can be sampled
    sample_root = ET.Element(root.tag, root.attrib)
    selected_count = 0
    descriptor_count = 0
    written_term_ids = set()

    for event, record in context:
        if event != "end" or record.tag != "DescriptorRecord":
            continue

        selected_in_record = 0

        # extracts concept information for terms
        for concept in record.findall("ConceptList/Concept"):
            term_list = concept.find("TermList")
            if term_list is None:
                continue

            for term in list(term_list.findall("Term")):
                term_ui = term.findtext("TermUI")

                # TermUI can occur in more than one Record in original XML.
                # Sample only contains unique TermUI values, meaning only the first occurrence is retained in sample XML
                if (term_ui not in selected_term_ids or term_ui in written_term_ids):
                    term_list.remove(term)
                else:
                    written_term_ids.add(term_ui)
                    selected_in_record += 1

        # if there are still DescriptorRecords that still contain any sampled term elements after filtering
        if selected_in_record:
            # deep copy of whole record is stored
            sample_root.append(copy.deepcopy(record))
            # selected counter increases for every sampled items that were written
            selected_count += selected_in_record
            # counts the amount of descriptors in the sampled data
            descriptor_count += 1

        record.clear()

    # calculates whether there are any missing term ids from the selected term ids
    missing_term_ids = selected_term_ids - written_term_ids
    if selected_count != len(selected_term_ids) or missing_term_ids:
        raise ValueError(f"Selected {len(selected_term_ids)} term IDs, but {selected_count} were written as unique Term elements. "
            f"({len(missing_term_ids)} selected IDs are still missing)."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Writes xml file
    ET.ElementTree(sample_root).write(output_path, encoding="utf-8", xml_declaration=True)
    return descriptor_count, selected_count


def create_sample_xml(input_path, output_path, required_terms, seed=MESH_SAMPLE_SEED):
    """
    Creates a reduced MeSH XML file containing at most SOURCE_ROWS elements from the original dataset.
    This approach samples across the complete source instead of only taking the first DescriptorRecords.
    """

    # samples the selected_term_ids and counts eligible item amount
    selected_term_ids, eligible_count = select_term_ids(input_path, required_terms, seed)

    # writes the record context for selected terms
    descriptor_count, selected_count = write_selected_records(input_path, output_path, selected_term_ids)

    print(f"Eligible terms in original source: {eligible_count}")
    print(f"Sampled terms: {selected_count}")
    print(f"DescriptorRecords retained: {descriptor_count}")
    print(f"Sample seed: {seed}")
    print(f"Saved to: {output_path}")    


if __name__ == "__main__":
    create_sample_xml(INPUT_PATH, OUTPUT_PATH, SOURCE_ROWS)
