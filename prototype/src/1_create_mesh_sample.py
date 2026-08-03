from pathlib import Path
import copy
import xml.etree.ElementTree as ET

INPUT_PATH = Path("prototype/data/raw/mesh/desc2026.xml")
OUTPUT_PATH = Path("prototype/data/raw/mesh/desc2026_sample.xml")

MAX_RECORDS = 1000

def create_sample_xml(input_path: Path, output_path: Path, max_records: int):
    context = ET.iterparse(input_path, events=("start", "end"))

    _, root = next(context)

    sample_root = ET.Element(root.tag, root.attrib)

    count = 0

    for event, elem in context:
        if event == "end" and elem.tag == "DescriptorRecord":
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
