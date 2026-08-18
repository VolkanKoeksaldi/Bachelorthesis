from pathlib import Path
import copy
import random
import xml.etree.ElementTree as ET
from experiment_config import MESH_SAMPLE_SEED, SOURCE_ROWS, experiment_path

source_dir = Path(__file__).resolve().parent
prototype_dir = source_dir.parent

INPUT_PATH = prototype_dir / "data" / "raw" / "mesh" / "desc2026.xml"
OUTPUT_PATH = experiment_path("prepared/mesh/desc2026_sample.xml")


def iter_eligible_terms(input_path):
    """
    Iterates through XML file and searches for unique eligible TermUI values.
    By using yield each matching term is returned immediately while parsing the file.
    The caller can process items one by one without materializing the
    full result set in the memory.

    Parameters:
        input_path: MeSH XML file source path
    """

    seen_term_ids = set()

    # Processes only completed DescriptorRecords.
    for _, descriptor_record in ET.iterparse(input_path, events=("end",)):
        if descriptor_record.tag != "DescriptorRecord":
            continue

        # Checks whether the descriptor_record contains any TreeNumber.
        tree_number = any(element.text and element.text.strip() 
                          for element in descriptor_record.findall("TreeNumberList/TreeNumber"))

        # Extracts information from descriptor records with tree number.
        if tree_number:
            for term in descriptor_record.findall("ConceptList/Concept/TermList/Term"):
                term_ui = term.findtext("TermUI")
                term_text = term.findtext("String")

                # Skips incomplete terms and duplicate TermUI values.
                if not term_ui or not term_text or term_ui in seen_term_ids:
                    continue

                seen_term_ids.add(term_ui)

                # Yields one value at a time and pauses until the next value is requested.
                yield term_ui

        # Clears the processed record in order to save memory usage. 
        descriptor_record.clear()

def select_term_ids(input_path, required_terms, seed):
    """
    Selects a deterministic uniform sample of eligible TermUI values.

    Parameters:
        input_path: MeSH XML file source path
        required_terms: Number of unique terms that need to be selected
        seed: Randomizer seed for the selection of sample data
    
    Returns:
        set: a unique set of every selected TermUI
        count: number of eligible MeSH terms encountered
    """

    rng = random.Random(seed)
    array = []
    # Counts eligible items.
    count = 0

    # Iterates through every term_ui from the eligible terms.
    for term_ui in iter_eligible_terms(input_path):
        count += 1

        if len(array) < required_terms:
            array.append(term_ui)
            continue

        # Chooses a random replacement index from 0 to count - 1.
        replacement_index = rng.randrange(count)

        # Checks whether the index is lower than the number of required terms.
        # If yes: term_ui can be placed in the array of sampled data.
        if replacement_index < required_terms:
            array[replacement_index] = term_ui

    if count < required_terms:
        raise ValueError(f"Only {count} eligible MeSH terms were found, but "
                         f"{required_terms} are required.")

    return set(array), count   

def write_selected_records(input_path, output_path, selected_term_ids):
    """
    Writes a reduced MeSH XML file that contains only the selected terms.
    The DescriptorRecord elements and their metadata are preserved, but only
    sampled Term elements are retained. If a TermUI occurs multiple times, only its first
    occurrence is stored.

    input:
        input_path: MeSH XML file source path
        output_path: Path to the reduced output XML file
        selected_term_ids: Set of unique sampled TermUI values
    
    return:
        descriptor_count: Number of written DescriptorRecords
        selected_count: Number of written unique items
    """

    # Starts parsing and yields element from start to end.
    context = ET.iterparse(input_path, events=("start", "end"))

    # Gets the original document root to preserve tag and attributes.
    _, root = next(context)

    # Creates new empty root with same tag name and attributes
    sample_root = ET.Element(root.tag, root.attrib)
    selected_count = 0
    descriptor_count = 0
    written_term_ids = set()

    for event, descriptor_record in context:
        if event != "end" or descriptor_record.tag != "DescriptorRecord":
            continue

        selected_in_record = 0

        for concept in descriptor_record.findall("ConceptList/Concept"):
            term_list = concept.find("TermList")
            if term_list is None:
                continue

            # Creates a stable copy with list() because elements are removed from TermList.
            for term in list(term_list.findall("Term")):
                term_ui = term.findtext("TermUI")

                # A TermUI can occur in multiple DescriptorRecords in the original XML.
                # Takes only its first selected occurrence.
                if term_ui not in selected_term_ids or term_ui in written_term_ids:
                    term_list.remove(term)
                else:
                    written_term_ids.add(term_ui)
                    selected_in_record += 1

        # DescriptorRecords without any selected terms are ignored.
        if selected_in_record:
            # Creates a deep copy because otherwise the parsed element is cleared.
            sample_root.append(copy.deepcopy(descriptor_record))
            selected_count += selected_in_record
            descriptor_count += 1

        descriptor_record.clear()

    # Verifies that every selected Term ID was written exactly once
    missing_term_ids = selected_term_ids - written_term_ids
    if selected_count != len(selected_term_ids) or missing_term_ids:
        raise ValueError(f"Selected {len(selected_term_ids)} term IDs, "
                         f"but {selected_count} were written as elements. "
                         f"({len(missing_term_ids)} selected IDs are still missing).")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(sample_root).write(output_path, encoding="utf-8", xml_declaration=True)

    return descriptor_count, selected_count


def create_sample_xml(input_path, output_path, required_terms, seed=MESH_SAMPLE_SEED):
    """
    Creates the reduced MeSH XML file that contains the required number of unique sampled terms.

    Parameters:
        input_path: MeSH XML file source path
        output_path: Path to the reduced output XML file
        required_terms: Number of unique terms that need to be included
        seed: Seed is used for reproducible sampling
    """

    selected_term_ids, eligible_count = select_term_ids(input_path, required_terms, seed)

    # Writes the descriptor record context for selected terms.
    descriptor_count, selected_count = write_selected_records(input_path, output_path,
                                                              selected_term_ids)

    print(f"Eligible terms in original source: {eligible_count}")
    print(f"Sampled terms: {selected_count}")
    print(f"DescriptorRecords: {descriptor_count}")
    print(f"Sample seed: {seed}")
    print(f"Saved to: {output_path}")    


if __name__ == "__main__":
    create_sample_xml(INPUT_PATH, OUTPUT_PATH, SOURCE_ROWS)
