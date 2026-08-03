# Generates imdb_titles.csv

from pathlib import Path
import json
import pandas as pd

basics_path = Path("prototype/data/raw/imdb/title.basics.tsv")
ratings_path = Path("prototype/data/raw/imdb/title.ratings.tsv")
output_path = Path("prototype/output/processed/imdb_titles.csv")

basics_column = [
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "endYear",
    "runtimeMinutes",
    "genres"
]

ratings_column = [
    "tconst",
    "averageRating",
    "numVotes"
]

MAX_TITLES = 56341
RANDOM_SEED = 42

TITLE_TYPES = None # bei None wird das nicht gefiltert nach Title Typen oder zb ["movie", "tvSeries", "tvMovie"]

def read_tsv(path: Path, use_columns=None):
    """
    Liest eine TSV Datei.
    """

    return pd.read_csv(path, sep="\t", usecols=use_columns, na_values="\\N", compression="infer", low_memory=False)

def clean_values(value):
    """
    Um fehlende Pandas Werte in None umzuwandeln.
    """

    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()
    
    return value

def build_metadata(row):
    """
    Baut ein JSON Objekt aus den benötigten Feldern
    """

    return {
        "original_title": clean_values(row.get("originalTitle")),
        "title_type": clean_values(row.get("titleType")),
        "is_adult": clean_values(row.get("isAdult")),
        "start_year": clean_values(row.get("startYear")),
        "end_year": clean_values(row.get("endYear")),
        "runtime_minutes": clean_values(row.get("runtimeMinutes")),
        "genres": parse_genres(row.get("genres")),
        "average_rating": clean_values(row.get("averageRating")),
        "num_votes": clean_values(row.get("numVotes"))
    }

def calculate_item_size(row):
    item_data = (str(row["tconst"])
                 + str(row["primaryTitle"])
                 + row["metadata_json"])

    return len(item_data.encode("utf-8"))

def parse_genres(value):
    """
    Wandelt den Genre String in eine Liste um.
    """

    if pd.isna(value):
        return []

    return [genre.strip() for genre in str(value).split(",") if genre.strip()]

def get_primary_genre(value):
    genres = parse_genres(value)

    if not genres:
        return "UNKNOWN"

    return genres[0]

def prepare_titles():
    """
    Lädt die title.basics Datei und Ratings und verbindet diese als CSV-Datei
    """

    basics = read_tsv(basics_path, use_columns=basics_column)
    ratings = read_tsv(ratings_path, use_columns=ratings_column)

    # hier werden die Dateien über tconst gejoined:
    data = basics.merge(ratings, on="tconst", how="left")

    # Beispiel-Filter mit definierten Bedingungen:
    data["titleType"] = data["titleType"].fillna("UNKNOWN")

    if TITLE_TYPES is not None:
        data = data[data["titleType"].isin(TITLE_TYPES)].copy()

    data = data[data["primaryTitle"].notna()].copy()

    if MAX_TITLES is not None:
        data = data.sample(n=min(MAX_TITLES, len(data)), random_state=RANDOM_SEED).reset_index(drop=True)

    integer_columns = [
        "isAdult",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "numVotes"
    ]

    for column in integer_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")    

    def compute_decade(year):
        if pd.isna(year):
            return "UNKNOWN"
        return f"{(int(year) // 10) * 10}s"

    data["decade"] = data["startYear"].apply(compute_decade)

    data["averageRating"] = pd.to_numeric(data["averageRating"], errors="coerce")

    data["primary_genre"] = data["genres"].apply(get_primary_genre)

    # Hier wird die Metadata nun als JSON gespeichert:
    data["metadata_json"] = data.apply(build_metadata, axis=1).apply(json.dumps)

    data["genres"] = data["genres"].fillna("UNKNOWN")

    data["item_size_bytes"] = data.apply(calculate_item_size, axis=1)

    # Für weitere Verarbeitungen wird jetzt result definiert, womit nur relevante Spalten entnommen werden:
    result = data[["tconst", "primaryTitle", "titleType", "decade", "primary_genre", "genres", "metadata_json", "item_size_bytes"]].copy()

    result = result.rename(columns={"tconst": "title_id", "primaryTitle": "primary_title", "titleType": "title_type"})

    # Ausgabe
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Die Datei wurde gespeichert unter {output_path}")

if __name__ == "__main__":
    prepare_titles()