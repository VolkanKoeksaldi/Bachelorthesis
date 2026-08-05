from pathlib import Path
import pandas as pd

input_path = Path("prototype/output/processed/imdb_titles.csv")
membership_output_path = Path("prototype/output/processed/imdb_fragment_memberships.csv")

fragment_output_path = Path("prototype/output/processed/imdb_fragments.csv")

def create_memberships(titles_df):
    """
    Creates fragment memberships for each title based on the title type, decade, and primary genre.
    Each title then belongs to exactly one fragment in each scheme.
    """

    membership_rows = []

    # iterates over all titles without including the index from the DataFrame
    for row in titles_df.itertuples(index=False):
        # membership based on title type
        membership_rows.append({
             "fragment_id": f"title_type_{row.title_type}",
             "scheme": "title_type",
             "value": row.title_type,
             "title_id": row.title_id,
             "primary_title": row.primary_title
        })

        # membership based on decade
        membership_rows.append({
             "fragment_id": f"decade_{row.decade}",
             "scheme": "decade",
             "value": row.decade,
             "title_id": row.title_id,
             "primary_title": row.primary_title
        })

        # membership based on primary genre
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
    Title memberships are grouped into fragments.
    """

    fragment_rows = []

    # Each group represents one fragment
    for (fragment_id, scheme, value), group in memberships.groupby(["fragment_id", "scheme", "value"]):

        # Ensures that each title occurs only once within a fragment and orders the titles by their ids.
        unique_titles = (group
                         .drop_duplicates(subset=["title_id"])
                         .sort_values("title_id"))

        # tolist() transforms the pandas series of the string with title_id column into a python list.
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
    """
    Loads the prepared titles and creates their fragment memberships.
    Afterwards the memberships are grouped into fragments and both result tables are stored.
    """

    titles_df = pd.read_csv(input_path)

    memberships_df = create_memberships(titles_df)

    fragments_df = create_fragments(memberships_df)

    membership_output_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_output_path.parent.mkdir(parents=True, exist_ok=True)
    memberships_df.to_csv(membership_output_path, index=False)
    fragments_df.to_csv(fragment_output_path, index=False)

    print(f"Unique titles: {titles_df['title_id'].nunique()}")
    print(f"Number of Memberships: {len(memberships_df)}")
    print(f"Number of Fragments: {len(fragments_df)}")

def main():
    process_imdb_fragments(input_path, membership_output_path, fragment_output_path)


if __name__ == "__main__":
    main()