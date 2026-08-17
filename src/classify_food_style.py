from pathlib import Path
import json
import os
import time

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path("data/processed/places_final.csv")
OUTPUT_FILE = Path("data/processed/places_with_food_style.csv")
REVIEW_FILE = Path("data/processed/food_style_to_review.csv")

# Start small while testing
TEST_LIMIT = 10

# Only accept classifications above this threshold automatically
AUTO_ACCEPT_THRESHOLD = 0.85


# =========================================================
# API
# =========================================================

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Add it to your .env file."
    )

client = OpenAI(
    api_key=API_KEY
)


# =========================================================
# EXISTING GOOGLE-BASED FOOD STYLE RULES
# =========================================================

FOOD_STYLE_MAP = {
    # Cuisines
    "italian_restaurant": "Italian",
    "japanese_restaurant": "Japanese",
    "mexican_restaurant": "Mexican",
    "mediterranean_restaurant": "Mediterranean",
    "british_restaurant": "British",
    "french_restaurant": "French",
    "spanish_restaurant": "Spanish",
    "american_restaurant": "American",
    "indian_restaurant": "Indian",
    "chinese_restaurant": "Chinese",
    "asian_restaurant": "Asian",
    "thai_restaurant": "Thai",
    "greek_restaurant": "Greek",
    "lebanese_restaurant": "Lebanese",
    "australian_restaurant": "Australian",
    "middle_eastern_restaurant": "Middle Eastern",
    "basque_restaurant": "Basque",
    "latin_american_restaurant": "Latin American",
    "turkish_restaurant": "Turkish",
    "korean_restaurant": "Korean",
    "cantonese_restaurant": "Cantonese",
    "austrian_restaurant": "Austrian",
    "filipino_restaurant": "Filipino",
    "eastern_european_restaurant": "Eastern European",
    "persian_restaurant": "Persian",
    "peruvian_restaurant": "Peruvian",
    "polish_restaurant": "Polish",
    "german_restaurant": "German",
    "vietnamese_restaurant": "Vietnamese",
    "brazilian_restaurant": "Brazilian",
    "cajun_restaurant": "Cajun",
    "south_indian_restaurant": "South Indian",
    "caribbean_restaurant": "Caribbean",
    "south_american_restaurant": "South American",
    "taiwanese_restaurant": "Taiwanese",

    # Food styles
    "seafood_restaurant": "Seafood",
    "oyster_bar_restaurant": "Seafood",
    "steak_house": "Steakhouse",
    "vegetarian_restaurant": "Vegetarian",
    "vegan_restaurant": "Vegan",
    "halal_restaurant": "Halal",
    "barbecue_restaurant": "BBQ",
    "chicken_restaurant": "Chicken",

    # Strong format → cuisine/style signals
    "taco_restaurant": "Mexican",
    "sushi_restaurant": "Japanese",
    "ramen_restaurant": "Japanese",
    "japanese_curry_restaurant": "Japanese",
    "pizza_restaurant": "Italian",
    "hamburger_restaurant": "American",
    "falafel_restaurant": "Middle Eastern",
    "fish_and_chips_restaurant": "British",
    "chinese_noodle_restaurant": "Chinese",
}


def get_food_styles(types):

    if pd.isna(types):
        return None

    place_types = types.split(", ")

    styles = [
        FOOD_STYLE_MAP[t]
        for t in place_types
        if t in FOOD_STYLE_MAP
    ]

    if not styles:
        return None

    return ", ".join(
        dict.fromkeys(styles)
    )


# =========================================================
# PROMPT
# =========================================================

def build_prompt(row):

    return f"""
Classify the likely food style of this restaurant.

Use ONLY the evidence provided below.

Restaurant:
Name: {row.get("title")}
City: {row.get("city")}
Neighborhood: {row.get("neighborhood")}
Primary Google type: {row.get("primary_type")}
Google types: {row.get("google_types")}
Price level: {row.get("price_level")}
Google rating: {row.get("google_rating")}
Number of reviews: {row.get("review_count")}

The food_style should describe the restaurant's cuisine or dominant food concept.

Good examples:
Italian
Mexican
Japanese
Seafood
Steakhouse
Middle Eastern
Contemporary Mexican
Modern European
British
Mediterranean
Tacos
Burgers
Sandwiches
Fusion

Rules:

1. Do not invent a cuisine when there is not enough evidence.
2. Names can be evidence when they strongly indicate a cuisine or concept.
3. City and neighborhood are context, not proof of cuisine.
4. If uncertain, use "Unknown".
5. Confidence must be between 0 and 1.
6. Evidence should be brief and explain exactly why the classification was made.
"""


# =========================================================
# AI CLASSIFICATION
# =========================================================

def classify_with_ai(row):

    response = client.responses.create(
        model="gpt-5-mini",

        input=[
            {
                "role": "system",
                "content": (
                    "You are classifying restaurant metadata "
                    "for a data science project. "
                    "Be conservative. Never guess without evidence."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(row),
            },
        ],

        text={
            "format": {
                "type": "json_schema",
                "name": "food_style_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "food_style": {
                            "type": "string"
                        },
                        "confidence": {
                            "type": "number"
                        },
                        "evidence": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "food_style",
                        "confidence",
                        "evidence"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    result = json.loads(
        response.output_text
    )

    return result


# =========================================================
# LOAD DATA
# =========================================================

def load_places():

    places = pd.read_csv(
        INPUT_FILE
    )

    # Create Google-derived food style first
    places["food_style"] = places[
        "google_types"
    ].apply(get_food_styles)

    places["food_style_source"] = None
    places["food_style_confidence"] = None
    places["food_style_evidence"] = None

    google_mask = places[
        "food_style"
    ].notna()

    places.loc[
        google_mask,
        "food_style_source"
    ] = "GOOGLE_TYPES"

    places.loc[
        google_mask,
        "food_style_confidence"
    ] = 1.0

    places.loc[
        google_mask,
        "food_style_evidence"
    ] = "Derived from Google Places types."

    return places


# =========================================================
# CLASSIFY MISSING RESTAURANTS
# =========================================================

def classify_missing(places):

    mask = (
        (places["category"] == "Restaurant")
        & places["food_style"].isna()
    )

    missing = places[
        mask
    ].copy()

    print(
        f"Restaurants requiring AI classification: "
        f"{len(missing)}"
    )

    # Test on only 10 first
    if TEST_LIMIT:
        missing = missing.head(
            TEST_LIMIT
        )

    print(
        f"Processing now: {len(missing)}\n"
    )

    for counter, (index, row) in enumerate(
        missing.iterrows(),
        start=1
    ):

        print(
            f"[{counter}/{len(missing)}] "
            f"{row['title']}"
        )

        try:

            result = classify_with_ai(
                row
            )

            food_style = result[
                "food_style"
            ]

            confidence = float(
                result["confidence"]
            )

            evidence = result[
                "evidence"
            ]

            # Unknown is never automatically accepted
            if (
                food_style.lower() == "unknown"
                or confidence < AUTO_ACCEPT_THRESHOLD
            ):

                source = "AI_REVIEW"

            else:

                source = "AI_METADATA"

            places.at[
                index,
                "food_style"
            ] = food_style

            places.at[
                index,
                "food_style_confidence"
            ] = confidence

            places.at[
                index,
                "food_style_evidence"
            ] = evidence

            places.at[
                index,
                "food_style_source"
            ] = source

            print(
                f"  → {food_style} "
                f"({confidence:.2f})"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            places.at[
                index,
                "food_style_source"
            ] = "AI_ERROR"

        time.sleep(
            0.2
        )

    return places


# =========================================================
# SAVE
# =========================================================

def save_results(places):

    places.to_csv(
        OUTPUT_FILE,
        index=False
    )

    review = places[
        places[
            "food_style_source"
        ].isin([
            "AI_REVIEW",
            "AI_ERROR"
        ])
    ].copy()

    review.to_csv(
        REVIEW_FILE,
        index=False
    )

    print(
        f"\nSaved → {OUTPUT_FILE}"
    )

    print(
        f"Review cases → {REVIEW_FILE}"
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(places):

    restaurants = places[
        places["category"]
        == "Restaurant"
    ]

    print(
        "\n=============================="
    )

    print(
        "FOOD STYLE SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        f"Restaurants: "
        f"{len(restaurants)}"
    )

    print(
        "\nClassification source:"
    )

    print(
        restaurants[
            "food_style_source"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nCoverage:"
    )

    print(
        f"{restaurants['food_style'].notna().mean():.1%}"
    )

    print(
        "\nTop food styles:"
    )

    print(
        restaurants[
            "food_style"
        ].value_counts()
        .head(25)
    )


# =========================================================
# MAIN
# =========================================================

def main():

    places = load_places()

    places = classify_missing(
        places
    )

    save_results(
        places
    )

    print_summary(
        places
    )


if __name__ == "__main__":
    main()