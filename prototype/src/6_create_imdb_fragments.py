# Generates imdb_fragments.csv
# Generates imdb_fragment_memberships.csv

from pathlib import Path
import pandas as pd

input_path = Path("prototype/output/processed/imdb_titles.csv")
membership_output_path = Path("prototype/output/processed/imdb_fragment_memberships.csv")

fragment_output_path = Path("prototype/output/processed/imdb_fragments.csv")

def create_memberships(titles_df):
    """
    Erstellt die Memberships für jeden Titel mit "title_type, decade, genre" als Spalten.
    Ein Titel mit mehreren Genres erhält hier dann mehrere Genre-Memberships
    """

    membership_rows = []

    for row in titles_df.itertuples(index=False):
        # Membership für Titeltyp
        membership_rows.append({
             "fragment_id": f"title_type_{row.title_type}",
             "scheme": "title_type",
             "value": row.title_type,
             "title_id": row.title_id,
             "primary_title": row.primary_title
        })

        # Membership für Decade
        membership_rows.append({
             "fragment_id": f"decade_{row.decade}",
             "scheme": "decade",
             "value": row.decade,
             "title_id": row.title_id,
             "primary_title": row.primary_title
        })

        membership_rows.append({
            "fragment_id": f"primary_genre_{row.primary_genre}",
            "scheme": "primary_genre",
            "value": row.primary_genre,
            "title_id": row.title_id,
            "primary_title": row.primary_title
        })

    return pd.DataFrame(membership_rows)

def create_fragments(memberships):
    """
    Fasst die memberships zu Fragmenten zusammen.
    """

    fragment_rows = []

    for (fragment_id, scheme, value), group in memberships.groupby(["fragment_id", "scheme", "value"]):

        unique_titles = (group
                         .drop_duplicates(subset=["title_id"])
                         .sort_values("title_id"))

        title_ids = unique_titles["title_id"].astype(str).tolist()

        title_names = unique_titles["primary_title"].astype(str).tolist()

        fragment_rows.append({
               "fragment_id": fragment_id,
               "scheme": scheme,
               "value": value,
               "fragment_size": len(title_ids),
               "title_ids": ",".join(title_ids),
               "title_names": "|".join(title_names)
        })

    return pd.DataFrame(fragment_rows)
          

def process_imdb_fragments(input_path, membership_output_path, fragment_output_path):
    titles_df = pd.read_csv(input_path)

    memberships_df = create_memberships(titles_df)

    fragments_df = create_fragments(memberships_df)

    membership_output_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_output_path.parent.mkdir(parents=True, exist_ok=True)
    memberships_df.to_csv(membership_output_path, index=False)
    fragments_df.to_csv(fragment_output_path, index=False)

    print(f"Titles: {titles_df['title_id'].nunique()}")
    print(f"Memberships Anzahl: {len(memberships_df)}")
    print(f"Fragments Anzahl: {len(fragments_df)}")

def main():
    process_imdb_fragments(input_path, membership_output_path, fragment_output_path)


if __name__ == "__main__":
    main()
    