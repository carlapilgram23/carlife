from pathlib import Path
import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

FILES = {
    "CDMX.csv": "CDMX",
    "London.csv": "London",
    "Wanna go.csv": "Wanna go",
}


def load_places():
    frames = []

    for filename, list_name in FILES.items():
        path = RAW_DIR / filename
        df = pd.read_csv(path)

        # Standardize column names
        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        # Remember which Google Maps list it came from
        df["source_list"] = list_name

        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def clean_places(df):
    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove rows without a place title
    df = df.dropna(subset=["title"])

    # Clean whitespace from place names
    df["title"] = df["title"].str.strip()

    # Remove exact duplicates
    df = df.drop_duplicates()

    # Classify places as visited or wanna go
    df["status"] = df["source_list"].map({
        "CDMX": "visited",
        "London": "visited",
        "Wanna go": "wanna_go"
    })

    return df.reset_index(drop=True)


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    places = load_places()

    print(f"Raw rows: {len(places)}")

    places = clean_places(places)

    print(f"Clean places: {len(places)}")

    print("\nPlaces by list:")
    print(places["source_list"].value_counts())

    print("\nPlaces by status:")
    print(places["status"].value_counts())

    print(f"\nTotal places: {len(places)}")

    output = PROCESSED_DIR / "master_places.csv"
    places.to_csv(output, index=False)

    print(f"\nSaved → {output}")


if __name__ == "__main__":
    main()