from pathlib import Path
import json
import pandas as pd
from experiment_config import copy_factor, IMDB_RANDOM_SEED, SOURCE_ROWS, target_rows, experiment_path
from clustering_utils import expand_base_table

source_dir = Path(__file__).resolve().parent
prototype_dir = source_dir.parent

basics_path = prototype_dir / "data" / "raw" / "imdb" / "title.basics.tsv"
ratings_path = prototype_dir / "data" / "raw" / "imdb" / "title.ratings.tsv"
output_path = experiment_path("processed/imdb_titles.csv")

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

# Optional in case movie title is supposed to be filtered by title types.
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
    # \N for missing values
    # data compressions are automatically seen. If data is compressed then pandas decompresses it.
    # low_memory=False means that pandas analyses the file in chunks, without mixed type inference
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

def compute_decade(year):
    """
    Converts year into a decade label.
    """
    if pd.isna(year):
        return "UNKNOWN"
    
    return f"{(int(year) // 10) * 10}s"

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
    calculates the item size as the utf-8 size of title-id, primary title, and metadata
    """
    item_data = (row["title_id"]
                 + row["source_title_id"]
                 + row["primary_title"]
                 + row["metadata_json"])

    return len("".join(str(value or "") for value in item_data).encode("utf-8"))



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
    data = basics.merge(ratings, on="tconst", how="left", validate="one_to_one")

    # Replaces missing title types with "UNKNOWN"
    data["titleType"] = data["titleType"].fillna("UNKNOWN")

    # applies the optional title type filters
    if TITLE_TYPES is not None:
        data = data[data["titleType"].isin(TITLE_TYPES)].copy()

    # titles without primary title are not used as a processed item
    data = data[data["primaryTitle"].notna()].copy()

    # draws a random sample of configured SOURCE_ROWS size
    if len(data) < SOURCE_ROWS:
        raise ValueError(f"IMDb contains {len(data)} titles, but {SOURCE_ROWS} are expected.")
    base_df = data.sample(n=SOURCE_ROWS, random_state=IMDB_RANDOM_SEED).reset_index(drop = True)

    # converts isAdult to integer value
    base_df["isAdult"] = pd.to_numeric(base_df["isAdult"], errors="coerce").astype("Int64")

    # converts startYear to integer value
    base_df["startYear"] = pd.to_numeric(base_df["startYear"], errors="coerce").astype("Int64")

    # converts endYear to integer value
    base_df["endYear"] = pd.to_numeric(base_df["endYear"], errors="coerce").astype("Int64")

    # converts runtimeMinutes to integer value
    base_df["runtimeMinutes"] = pd.to_numeric(base_df["runtimeMinutes"], errors="coerce").astype("Int64")

    # converts numVotes to integer value
    base_df["numVotes"] = pd.to_numeric(base_df["numVotes"], errors="coerce").astype("Int64")  

    # converts ratings to numeric value
    base_df["averageRating"] = pd.to_numeric(base_df["averageRating"], errors="coerce")

    # calculated decade
    base_df["decade"] = base_df["startYear"].apply(compute_decade)

    # calculates primary_genre
    base_df["primary_genre"] = base_df["genres"].apply(get_primary_genre)

    # Adds the metadata json as information
    base_df["metadata_json"] = base_df.apply(build_metadata, axis=1).apply(lambda value: json.dumps(value,
                                                                                                    ensure_ascii=False,
                                                                                                    separators=(",", ":")))

    # Uses UNKNOWN for missing genre strings in CSV
    base_df["genres"] = base_df["genres"].fillna("UNKNOWN")

    # renames columns to snake case names
    base_df = base_df.rename(columns={
        "tconst": "source_title_id",
        "primaryTitle": "primary_title",
        "titleType": "title_type",
    })

    # sets DataFrame as the most important information for Fragments
    base_df = base_df[[
        "source_title_id",
        "primary_title",
        "title_type",
        "decade",
        "primary_genre",
        "genres",
        "metadata_json",
    ]]

    if not base_df["source_title_id"].is_unique:
        raise ValueError("Selected IMDb source_title_id values are not unique.")

    # Copies base dataframe copy factor times
    result = expand_base_table(
        base_df,
        copy_factor,
        id_column="title_id",
        id_prefix="IT",
    )

    # calculates the item size in bytes
    result["item_size_bytes"] = result.apply(calculate_item_size, axis=1)

    
    result = result[[
        "title_id",
        "source_title_id",
        "primary_title",
        "title_type",
        "decade",
        "primary_genre",
        "genres",
        "metadata_json",
        "item_size_bytes",
        "copy_number",
    ]]

    if len(result) != target_rows:
        raise ValueError(f"Expected {target_rows} rows, but created {len(result)}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"Unique source titles: {len(base_df)}")
    print(f"Copy factor: {copy_factor}")
    print(f"IMDb rows: {len(result)}")
    print(f"Unique title IDs: {result['title_id'].nunique()}")
    print(f"Saved to: {output_path}")

    return result



if __name__ == "__main__":
    prepare_titles()