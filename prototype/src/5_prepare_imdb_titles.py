from pathlib import Path
import json
import pandas as pd

basics_path = Path("prototype/data/raw/imdb/title.basics.tsv")
ratings_path = Path("prototype/data/raw/imdb/title.ratings.tsv")
output_path = Path("prototype/output/processed/imdb_titles.csv")

# Columns required from title.basics.tsv
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

# Columns required from title.ratings.tsv
ratings_column = [
    "tconst",
    "averageRating",
    "numVotes"
]

MAX_TITLES = 1000

# Ensures that same sample is selected in repeated executions
RANDOM_SEED = 42

# Optional restriction to selected title types.
# If it is set to None, titles are not filtered.
# Example: ["movie", "tvSeries", "tvMovie"]
TITLE_TYPES = None

def read_tsv(path: Path, use_columns=None):
    """
    Reads selected columns from IMDb dataset.
    IMDb uses the string "\\N" to represent missing values.
    """

    # Loads selected tsv columns and converts IMDb \N values into missing values
    # sep="\t" separates columns with a tabulator
    # only using columns defined in use_columns
    # uses \N for missing values
    # compression is automatically seen. If data is compressed then pandas decompresses it.
    # low_memory=False means that pandas analyses the data, with which data types are seen better
    return pd.read_csv(path, sep="\t", usecols=use_columns, na_values="\\N", compression="infer", low_memory=False)

def clean_values(value):
    """
    Converts missing values to None and NumPy scalar values to corresponding Python values.
    """

    if pd.isna(value):
        return None

    # Converts values into Python values
    if hasattr(value, "item"):
        return value.item()
    
    return value

def parse_genres(value):
    """
    Converts a genre string separated by commas into a Python list.
    """

    if pd.isna(value):
        return []

    return [genre.strip() for genre in str(value).split(",") if genre.strip()]

def get_primary_genre(value):
    """
    Returns the first genre of a title.
    Titles without genre information are assigned "UNKNOWN" as primary genre.
    """

    genres = parse_genres(value)

    if not genres:
        return "UNKNOWN"

    return genres[0]

def build_metadata(row):
    """
    Builds the metadata object for an IMDb title.
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
    """
    approximates the item size as the utf-8 size of title-id, primary title, and metadata
    """
    item_data = (str(row["tconst"])
                 + str(row["primaryTitle"])
                 + row["metadata_json"])

    return len(item_data.encode("utf-8"))

def compute_decade(year):
    """
    Converts year into a decade label.
    """
    if pd.isna(year):
        return "UNKNOWN"
    
    return f"{(int(year) // 10) * 10}s"

def prepare_titles():
    """
    Loads the title and rating datasets.
    Afterwards these are merged on the title id.
    Then selects a reproducible sample, derives the fragmentation attributes and
    stores the processed titles as a CSV file.
    """

    # Loads required columns
    basics = read_tsv(basics_path, use_columns=basics_column)
    ratings = read_tsv(ratings_path, use_columns=ratings_column)

    # Joins title information and ratings on unique IMDb title ID
    data = basics.merge(ratings, on="tconst", how="left")

    # Replaces missing title types with "UNKNOWN"
    data["titleType"] = data["titleType"].fillna("UNKNOWN")

    # applies optional title type filters
    if TITLE_TYPES is not None:
        data = data[data["titleType"].isin(TITLE_TYPES)].copy()

    # titles without primary title are not used as a processed item
    data = data[data["primaryTitle"].notna()].copy()

    # draws a random sample of configured MAX_TITLES size
    if MAX_TITLES is not None:

        # samples the data using a random state. Afterwards resets the index of the data.
        data = data.sample(n=min(MAX_TITLES, len(data)), random_state=RANDOM_SEED).reset_index(drop=True)

    # converts integer columns
    integer_columns = [
        "isAdult",
        "startYear",
        "endYear",
        "runtimeMinutes",
        "numVotes"
    ]

    for column in integer_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("Int64")    

    # converts ratings to numeric value
    data["averageRating"] = pd.to_numeric(data["averageRating"], errors="coerce")

    # calculated decade
    data["decade"] = data["startYear"].apply(compute_decade)

    # calculates primary_genre
    data["primary_genre"] = data["genres"].apply(get_primary_genre)

    # Adds the metadata json as information
    data["metadata_json"] = data.apply(build_metadata, axis=1).apply(json.dumps)

    # Uses UNKNOWN for missing genre strings in CSV
    data["genres"] = data["genres"].fillna("UNKNOWN")

    # Calculates the item size for each processed title
    data["item_size_bytes"] = data.apply(calculate_item_size, axis=1)

    # Used to retain only required columns for the result
    result = data[["tconst", "primaryTitle", "titleType", "decade", "primary_genre", "genres", "metadata_json", "item_size_bytes"]].copy()

    # Uses consistent column names in output. renames tconst into title_id
    result = result.rename(columns={"tconst": "title_id", "primaryTitle": "primary_title", "titleType": "title_type"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    prepare_titles()