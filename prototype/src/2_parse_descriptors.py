from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
import json

# Pfad zur kleinen MeSH-Sample Datei.
# Wichtig: Wir arbeiten erstmal hiermit, damit das Parsen schneller geht.
INPUT_PATH = Path("prototype/data/raw/mesh/desc2026_sample.xml")

# Pfad für die verarbeitete Ausgabe.
# Aus der XML wird eine einfache CSV Tabelle erzeugt
# mit der wir später leichter Fragmente und Overlaps berechnen können
OUTPUT_PATH = Path("prototype/output/processed/mesh_descriptors_sample.csv")

def extract_tree_levels(tree_number: str):
    """
    Zerlegt eine MeSH Tree Number in mehrere Hierarchie-Ebenen.
    
    Beispiel:
    Tree Number: C14.280.647
    
    Daraus wird:
    top_category = C
    branch_code = C14
    subbranch_code = C14.280
    
    Diese Ebenen können später als verschiedene Fragmentierungsschemata genutzt werden.
    Zum Beispiel:
    - Fragmentierung nach top_category
    - Fragmentierung nach branch_code
    - Fragmentierung nach subbranch_code
    """

    # Eine Tree Number wie C14.280.647 wird an den Punkten jeweils getrennt, damit wir daraus ein Array bauen:
    # Ergebnis: ["C14", "280", "647"]
    parts = tree_number.split(".")

    # Die erste Stelle der Tree Number beschreibt die Hauptkategorie zb. C
    top_category = tree_number[0]

    # Der erste Block beschreibt den Branch, also zb C14
    branch_code = parts[0]

    # Die ersten zwei Blocke gemeinsam beschreiben die Unterkategorie, also zb C14.280
    # Falls es keine zweite Ebene gibt, dann bleibt nur der branch_code
    if len(parts) >= 2:
        subbranch_code = ".".join(parts[:2])
    else:
        subbranch_code = parts[0]

    return top_category, branch_code, subbranch_code

def extract_metadata(descriptor_record):
    """
    Um weitere Informationen eines Descriptors zu extrahieren.
    -> Scope Notes
    -> Entry Terms und Synonyme
    -> Zulässige Qualifier
    -> Annotation
    """

    scope_notes = []

    for scope_note in descriptor_record.findall("ConceptList/Concept/ScopeNote"):
        if scope_note.text:
            scope_notes.append(scope_note.text)

    entry_terms = []

    for term in descriptor_record.findall("ConceptList/Concept/TermList/Term/String"):
        if term.text:
            entry_terms.append(term.text)

    allowable_qualifiers = []

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
    Liest MeSH Descriptor XML-Datei ein und extrahiert die wichtigsten Informationen
    
    Für jeden Descriptor werden asugelesen:
    - DescriptorUI: eindeutige ID des MeSH Begriffs
    - DescriptorName: Name des MeSH-Begriffs
    - TreeNumber: Position des Begriffs in der MeSH-Hierarchie
    
    Wichtig:
    Ein Descriptor kann mehrere Tree Numbers bestizen.
    Deshalb kann derselbe Descriptor mehrfach in der CSV-Ausgabe vorkommen.
    Das ist kein Fehler, sondern für unsere Overlap-Analyse später sogar wichtig.
    """

    # Sammeln von allen extrahierten Zeilen
    rows = []

    # iterparse liest XML-Datei schrittweise
    for event, elem in ET.iterparse(xml_path, events=("end",)):

        # wir interessieren uns nur für vollständige DescriptorRecord-Elemente.
        # Jeder DescriptorRecord beschreibt einen MeSH Descriptor.
        if elem.tag == "DescriptorRecord":

            # Eindeutige ID des Descriptors, z.B. D006331
            descriptor_ui = elem.findtext("DescriptorUI")

            # Name des Descriptors, z.B. Heart Diseases
            descriptor_name = elem.findtext("DescriptorName/String")

            metadata = extract_metadata(elem)
            metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))

            item_size_bytes = len(((descriptor_ui or "") + (descriptor_name or "") + metadata_json).encode("utf-8"))

            # Alle gültigen Tree Numbers dieses Descriptors sammeln.
            tree_numbers = sorted({
                tree_elem.text.strip()
                for tree_elem in elem.findall("TreeNumberList/TreeNumber")
                if tree_elem.text and tree_elem.text.strip()
            })

            # Descriptors ohne Tree Number gehören nicht zu den verwendeten
            # hierarchischen Fragmentierungen.
            if not tree_numbers:
                elem.clear()
                continue

            # Deterministische Auswahl:
            # Die lexikografisch kleinste Tree Number ist die primäre Tree Number.
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
            
            # aktuelles XML-Element wird danach aus dem Speicher entfernt
            elem.clear()

    return pd.DataFrame(rows)

def process_mesh_descriptors(input_path: Path, output_path: Path):
    """
    Ausgabeordner erstellen, MeSH XML Dateien parsen, Statistik ausgeben und diese dann als CSV speichern.
    """
    # Erstellt Ausgabeordner, falls er noch nicht existiert
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # XML-Datei wird geparsed und in tabellarische Form gebracht
    df = parse_mesh_descriptors(input_path)
    if not df["descriptor_ui"].is_unique:
        duplicate_ids = (
            df.loc[df["descriptor_ui"].duplicated(keep=False), "descriptor_ui"]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Die deterministische Tree-Number-Auswahl hat weiterhin "
            f"doppelte Descriptoren erzeugt: {duplicate_ids[:10]}"
        )

    # Kontrollausgaben:
    print("Rows:", len(df))
    print("Unique descriptors:", df["descriptor_ui"].nunique())
    print("Unique tree numbers:", df["tree_number"].nunique())
    print()
    # Ergebnis als CSV speichern für spätere Fragmentierung und Overlap Detection
    df.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")
    return df

def main():
    process_mesh_descriptors(INPUT_PATH, OUTPUT_PATH)


if __name__ == "__main__":
    main()