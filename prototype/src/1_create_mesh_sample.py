from pathlib import Path
import copy
import xml.etree.ElementTree as ET
from experiment_config import MESH_MAX_RECORDS, experiment_path


# Input and output paths for original and reduced MeSH datasets
INPUT_PATH = Path("prototype/data/raw/mesh/desc2026.xml")
OUTPUT_PATH = experiment_path("prepared/mesh/desc2026_sample.xml")

# Sets the maximum number of elements included in the sample
MAX_RECORDS = MESH_MAX_RECORDS

def create_sample_xml(input_path: Path, output_path: Path, max_records: int):
    """
    Creates a reduced MeSH XML file containing at most max_records descriptor elements from the original dataset.
    """

    # Parses the XML file incrementally to avoid loading it entirely into memory
    context = ET.iterparse(input_path, events=("start", "end"))

    # Retrieves the root element from the XML document
    _, root = next(context)

    # Creates a new root element with the same information
    sample_root = ET.Element(root.tag, root.attrib)

    count = 0

    for event, elem in context:
        if event == "end" and elem.tag == "DescriptorRecord":
            # Copies the record, because the original element is cleared in order to release memory
            sample_root.append(copy.deepcopy(elem))
            count += 1

            elem.clear()

            if count >= max_records:
                break

    output_path.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.ElementTree(sample_root)
    tree.write(output_path, encoding="utf-8", xml_declaration=True)

    print(f"Created sample XML with {count} DescriptorRecords:")
    print(output_path)


if __name__ == "__main__":
    create_sample_xml(INPUT_PATH, OUTPUT_PATH, MAX_RECORDS)
