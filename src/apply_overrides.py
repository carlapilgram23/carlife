from pathlib import Path

import pandas as pd


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path(
    "data/processed/places_with_food_classification.csv"
)

OVERRIDES_FILE = Path(
    "data/manual/food_classification_overrides.csv"
)

OUTPUT_FILE = Path(
    "data/processed/places_with_food_classification_final.csv"
)


# =========================================================
# LOAD DATA
# =========================================================

places = pd.read_csv(
    INPUT_FILE
)

overrides = pd.read_csv(
    OVERRIDES_FILE
)


# =========================================================
# VALIDATION
# =========================================================

required_columns = [
    "google_place_id",
    "cuisine",
    "cuisine_detail",
    "specialty",
    "specialty_detail",
    "manual_status",
]

missing_columns = [
    column
    for column in required_columns
    if column not in overrides.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in overrides file: {missing_columns}"
    )


# =========================================================
# APPLY OVERRIDES
# =========================================================

override_columns = [
    "cuisine",
    "cuisine_detail",
    "specialty",
    "specialty_detail",
]

override_count = 0


for _, override in overrides.iterrows():

    place_id = override[
        "google_place_id"
    ]

    mask = (
        places["google_place_id"]
        == place_id
    )

    matches = mask.sum()

    if matches == 0:

        print(
            f"WARNING: Place ID not found: "
            f"{place_id}"
        )

        continue

    if matches > 1:

        print(
            f"WARNING: Duplicate Place ID: "
            f"{place_id}"
        )

    for column in override_columns:

        value = override[
            column
        ]

        if pd.notna(value):

            places.loc[
                mask,
                column
            ] = value

    places.loc[
        mask,
        "food_classification_source"
    ] = "MANUAL_OVERRIDE"

    places.loc[
        mask,
        "food_classification_confidence"
    ] = 1.0

    places.loc[
        mask,
        "food_classification_evidence"
    ] = (
        "Manually reviewed and approved."
    )

    override_count += matches


# =========================================================
# SAVE FINAL DATASET
# =========================================================

places.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print(
    f"Overrides applied: "
    f"{override_count}"
)

print(
    f"Saved → {OUTPUT_FILE}"
)

print(
    "\nClassification source:"
)

print(
    places[
        "food_classification_source"
    ].value_counts(
        dropna=False
    )
)