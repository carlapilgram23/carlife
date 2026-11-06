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

OUTPUT_FILE = Path(
    "data/processed/places_with_food_classification.csv"
)

REVIEW_FILE = Path(
    "data/processed/food_classification_to_review.csv"
)

# Keep at 10 while testing.
# Change to None when ready for the full run.
TEST_LIMIT = None

AUTO_ACCEPT_THRESHOLD = 0.85
WEB_SEARCH_THRESHOLD = 0.85


# =========================================================
# API
# =========================================================

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY was not found. "
        "Check your .env file."
    )

client = OpenAI(
    api_key=API_KEY
)


# =========================================================
# CONTROLLED CUISINE TAXONOMY
# =========================================================

ALLOWED_CUISINES = [
    "Mexican",
    "Italian",
    "Japanese",
    "French",
    "British",
    "Spanish",
    "Mediterranean",
    "Chinese",
    "Korean",
    "Indian",
    "Thai",
    "Greek",
    "Lebanese",
    "Middle Eastern",
    "American",
    "Colombian",
    "Peruvian",
    "Brazilian",
    "Caribbean",
    "Taiwanese",
    "Vietnamese",
    "Turkish",
    "Polish",
    "German",
    "Persian",
    "Filipino",
    "Australian",
    "Austrian",
    "Eastern European",
    "Latin American",
    "South American",
    "Asian",
    "European",
    "Fusion",
    "Other",
    "Unknown",
]


# =========================================================
# CONTROLLED SPECIALTY TAXONOMY
# =========================================================

ALLOWED_SPECIALTIES = [
    "Pizza",
    "Tacos",
    "Sushi",
    "Bagels",
    "Seafood",
    "Oysters",
    "Burgers",
    "Steak",
    "Sandwiches",
    "Ramen",
    "Dumplings",
    "Noodles",
    "BBQ",
    "Chicken",
    "Falafel",
    "Fish & Chips",
    "Deli",
    "Breakfast",
    "Brunch",
    "Fine Dining",
    "Bistro",
    "Vegetarian",
    "Vegan",
    "Fast Food",
    "Takeaway",
    "Pasta",
    "Ceviche",
    "Pastries",
    "Unknown",
]


# =========================================================
# GOOGLE TYPES → CUISINE
# =========================================================

CUISINE_MAP = {
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
    "latin_american_restaurant": "Latin American",
    "turkish_restaurant": "Turkish",
    "korean_restaurant": "Korean",
    "austrian_restaurant": "Austrian",
    "filipino_restaurant": "Filipino",
    "eastern_european_restaurant": "Eastern European",
    "persian_restaurant": "Persian",
    "peruvian_restaurant": "Peruvian",
    "polish_restaurant": "Polish",
    "german_restaurant": "German",
    "vietnamese_restaurant": "Vietnamese",
    "brazilian_restaurant": "Brazilian",
    "taiwanese_restaurant": "Taiwanese",

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


# =========================================================
# GOOGLE TYPES → SPECIALTY
# =========================================================

SPECIALTY_MAP = {
    "seafood_restaurant": "Seafood",
    "oyster_bar_restaurant": "Oysters",
    "steak_house": "Steak",
    "barbecue_restaurant": "BBQ",
    "chicken_restaurant": "Chicken",

    "taco_restaurant": "Tacos",
    "sushi_restaurant": "Sushi",
    "ramen_restaurant": "Ramen",
    "pizza_restaurant": "Pizza",
    "hamburger_restaurant": "Burgers",
    "falafel_restaurant": "Falafel",
    "fish_and_chips_restaurant": "Fish & Chips",

    "bagel_shop": "Bagels",
    "sandwich_shop": "Sandwiches",
    "dumpling_restaurant": "Dumplings",
    "noodle_shop": "Noodles",
    "deli": "Deli",
    "bistro": "Bistro",

    "vegetarian_restaurant": "Vegetarian",
    "vegan_restaurant": "Vegan",

    "breakfast_restaurant": "Breakfast",
    "brunch_restaurant": "Brunch",
    "fine_dining_restaurant": "Fine Dining",
    "fast_food_restaurant": "Fast Food",
    "meal_takeaway": "Takeaway",
}


# =========================================================
# HELPERS
# =========================================================

def split_types(types):
    if pd.isna(types):
        return []

    return [
        item.strip()
        for item in types.split(",")
    ]


def get_google_matches(types, mapping):
    place_types = split_types(types)

    values = [
        mapping[t]
        for t in place_types
        if t in mapping
    ]

    return list(
        dict.fromkeys(values)
    )


def get_primary_google_value(types, mapping):
    values = get_google_matches(
        types,
        mapping
    )

    if not values:
        return None

    return values[0]


def get_google_detail(types, mapping):
    values = get_google_matches(
        types,
        mapping
    )

    if not values:
        return None

    return ", ".join(values)


def get_google_cuisine(types):
    return get_primary_google_value(
        types,
        CUISINE_MAP
    )


def get_google_cuisine_detail(types):
    return get_google_detail(
        types,
        CUISINE_MAP
    )


def get_google_specialty(types):
    return get_primary_google_value(
        types,
        SPECIALTY_MAP
    )


def get_google_specialty_detail(types):
    return get_google_detail(
        types,
        SPECIALTY_MAP
    )


# =========================================================
# STRUCTURED OUTPUT SCHEMA
# =========================================================

FOOD_SCHEMA = {
    "type": "json_schema",
    "name": "restaurant_food_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {

            "cuisine": {
                "type": "string",
                "enum": ALLOWED_CUISINES,
            },

            "cuisine_detail": {
                "type": "string"
            },

            "specialty": {
                "type": "string",
                "enum": ALLOWED_SPECIALTIES,
            },

            "specialty_detail": {
                "type": "string"
            },

            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1
            },

            "evidence": {
                "type": "string"
            }
        },

        "required": [
            "cuisine",
            "cuisine_detail",
            "specialty",
            "specialty_detail",
            "confidence",
            "evidence"
        ],

        "additionalProperties": False
    }
}


# =========================================================
# METADATA PROMPT
# =========================================================

def build_metadata_prompt(row):

    cuisines = ", ".join(
        ALLOWED_CUISINES
    )

    specialties = ", ".join(
        ALLOWED_SPECIALTIES
    )

    return f"""
You are classifying food information for a restaurant dataset.

Every record has ALREADY been classified as a Restaurant.

Your task is to identify:

1. cuisine
2. cuisine_detail
3. specialty
4. specialty_detail

Restaurant metadata:

Name:
{row.get("title")}

Google name:
{row.get("google_name")}

City:
{row.get("city")}

Neighborhood:
{row.get("neighborhood")}

Full address:
{row.get("formatted_address")}

Primary Google type:
{row.get("primary_type")}

Google types:
{row.get("google_types")}

Price level:
{row.get("price_level")}

Google rating:
{row.get("google_rating")}

Review count:
{row.get("review_count")}


CUISINE

Cuisine MUST be exactly one of:

{cuisines}

Cuisine should be one broad analytical category.

Examples:

Italian-American pizza
→ cuisine = Italian
→ cuisine_detail = Italian-American / New York-style

Contemporary Mexican
→ cuisine = Mexican
→ cuisine_detail = Contemporary Mexican

Sinaloan food with Asian influences
→ cuisine = Fusion
→ cuisine_detail = Sinaloan-Asian seafood fusion

New York Jewish deli
→ cuisine = American
→ cuisine_detail = New York-style Jewish deli


SPECIALTY

Specialty MUST be exactly one of:

{specialties}

Specialty should be one clean analytical category.

Examples:

New York-style pizza by the slice
→ specialty = Pizza
→ specialty_detail = New York-style thin-crust pizza by the slice

Wood-fired Neapolitan pizza
→ specialty = Pizza
→ specialty_detail = Neapolitan wood-fired pizza

Seafood restaurant known for ceviches
→ specialty = Seafood
→ specialty_detail = Ceviches, oysters and grilled seafood

Bagel shop with lox and pastrami
→ specialty = Bagels
→ specialty_detail = Bagels, lox and pastrami sandwiches


RULES

1. These places are already restaurants.
2. Do not classify establishment type.
3. cuisine must be exactly one controlled category.
4. specialty must be exactly one controlled category.
5. cuisine_detail may be descriptive.
6. specialty_detail may be descriptive.
7. Do not infer cuisine from city or neighborhood alone.
8. Google types are stronger evidence than restaurant names.
9. If cuisine is unsupported, return Unknown.
10. If specialty is unsupported, return Unknown.
11. Confidence must reflect actual evidence.
12. It is better to return Unknown than invent information.
"""


# =========================================================
# METADATA CLASSIFICATION
# =========================================================

def classify_with_metadata(row):

    response = client.responses.create(
        model="gpt-5-mini",

        input=[
            {
                "role": "system",
                "content": (
                    "Classify restaurant food metadata conservatively. "
                    "Never invent unsupported information."
                ),
            },
            {
                "role": "user",
                "content": build_metadata_prompt(row),
            },
        ],

        text={
            "format": FOOD_SCHEMA
        }
    )

    return json.loads(
        response.output_text
    )


# =========================================================
# WEB SEARCH PROMPT
# =========================================================

def build_web_prompt(row):

    cuisines = ", ".join(
        ALLOWED_CUISINES
    )

    specialties = ", ".join(
        ALLOWED_SPECIALTIES
    )

    return f"""
Research the EXACT restaurant below using web search.

The Google identity and location information are provided
to disambiguate the establishment.

Restaurant:

Original title:
{row.get("title")}

Google name:
{row.get("google_name")}

City:
{row.get("city")}

Neighborhood:
{row.get("neighborhood")}

Full address:
{row.get("formatted_address")}

Google Maps URL:
{row.get("google_maps_uri")}

Original saved Google Maps URL:
{row.get("url")}

Google Place ID:
{row.get("google_place_id")}

Google types:
{row.get("google_types")}


FIRST IDENTIFY THE EXACT ESTABLISHMENT.

Prefer:

1. Official restaurant website
2. Official menu
3. Official social pages
4. Reputable restaurant guides
5. Reputable editorial coverage


CUISINE

Cuisine MUST be exactly one of:

{cuisines}


SPECIALTY

Specialty MUST be exactly one of:

{specialties}


Use cuisine_detail and specialty_detail
to preserve richer information.

Examples:

cuisine = Colombian
cuisine_detail = Contemporary coastal Colombian / Caribbean
specialty = Seafood
specialty_detail = Ceviches, grilled fish and Caribbean seafood

cuisine = Italian
cuisine_detail = Italian-American / New York-style
specialty = Pizza
specialty_detail = Thin, foldable New York-style pizza by the slice

cuisine = Mexican
cuisine_detail = Contemporary Mexican, farm-to-table
specialty = Fine Dining
specialty_detail = Seasonal interpretations of regional Mexican dishes

cuisine = Fusion
cuisine_detail = Sinaloan-Asian seafood fusion
specialty = Seafood
specialty_detail = Aguachiles, ceviches and Asian-inspired seafood dishes


RULES

1. Identify the exact establishment first.
2. Do not infer cuisine from location alone.
3. Prefer first-party information.
4. cuisine must be exactly one controlled category.
5. specialty must be exactly one controlled category.
6. cuisine_detail may be descriptive.
7. specialty_detail may be descriptive.
8. If exact identity cannot be confirmed, return Unknown.
9. Confidence reflects both identity and classification confidence.
10. Evidence should briefly explain the public information used.
"""


# =========================================================
# WEB CLASSIFICATION
# =========================================================

def classify_with_web(row):

    response = client.responses.create(
        model="gpt-5-mini",

        tools=[
            {
                "type": "web_search",
                "search_context_size": "low",
            }
        ],

        tool_choice="auto",

        input=[
            {
                "role": "system",
                "content": (
                    "Research the exact restaurant before classifying it. "
                    "Prefer official websites and menus. "
                    "Be conservative about entity matching."
                ),
            },
            {
                "role": "user",
                "content": build_web_prompt(row),
            },
        ],

        text={
            "format": FOOD_SCHEMA
        }
    )

    return json.loads(
        response.output_text
    )


# =========================================================
# SHOULD WEB SEARCH RUN?
# =========================================================

def needs_web_search(result):

    cuisine = (
        result.get("cuisine")
        or ""
    ).strip()

    specialty = (
        result.get("specialty")
        or ""
    ).strip()

    confidence = float(
        result.get(
            "confidence",
            0
        )
    )

    return (
        confidence < WEB_SEARCH_THRESHOLD
        or cuisine == "Unknown"
        or specialty == "Unknown"
    )


# =========================================================
# LOAD DATA
# =========================================================

def load_places():

    places = pd.read_csv(
        INPUT_FILE
    )

    # -----------------------------------------------------
    # GOOGLE CUISINE
    # -----------------------------------------------------

    places["cuisine"] = places[
        "google_types"
    ].apply(
        get_google_cuisine
    )

    places["cuisine_detail"] = places[
        "google_types"
    ].apply(
        get_google_cuisine_detail
    )

    # -----------------------------------------------------
    # GOOGLE SPECIALTY
    # -----------------------------------------------------

    places["specialty"] = places[
        "google_types"
    ].apply(
        get_google_specialty
    )

    places["specialty_detail"] = places[
        "google_types"
    ].apply(
        get_google_specialty_detail
    )

    # -----------------------------------------------------
    # PROVENANCE
    # -----------------------------------------------------

    places[
        "food_classification_source"
    ] = None

    places[
        "food_classification_confidence"
    ] = None

    places[
        "food_classification_evidence"
    ] = None

    google_mask = (
        places["cuisine"].notna()
        | places["specialty"].notna()
    )

    places.loc[
        google_mask,
        "food_classification_source"
    ] = "GOOGLE_TYPES"

    places.loc[
        google_mask,
        "food_classification_confidence"
    ] = 1.0

    places.loc[
        google_mask,
        "food_classification_evidence"
    ] = (
        "Derived directly from Google Places types."
    )

    return places


# =========================================================
# RESTAURANTS THAT NEED AI
# =========================================================

def get_missing_restaurants(places):

    restaurant_mask = (
        places["category"]
        == "Restaurant"
    )

    cuisine_missing = (
        places["cuisine"].isna()
    )

    return places[
        restaurant_mask
        & cuisine_missing
    ].copy()


# =========================================================
# CLASSIFY MISSING RESTAURANTS
# =========================================================

def classify_missing(places):

    missing = get_missing_restaurants(
        places
    )

    print(
        f"Restaurants requiring AI classification: "
        f"{len(missing)}"
    )

    if TEST_LIMIT is not None:
        missing = missing.head(
            TEST_LIMIT
        )

    print(
        f"Processing now: "
        f"{len(missing)}\n"
    )

    total = len(missing)

    for counter, (index, row) in enumerate(
        missing.iterrows(),
        start=1
    ):

        print(
            f"[{counter}/{total}] "
            f"{row['title']}"
        )

        try:

            # =============================================
            # STAGE 1 — METADATA
            # =============================================

            metadata_result = classify_with_metadata(
                row
            )

            final_result = metadata_result
            source = "AI_METADATA"

            print(
                "  Metadata → "
                f"{metadata_result['cuisine']} | "
                f"{metadata_result['cuisine_detail']} | "
                f"{metadata_result['specialty']} | "
                f"{metadata_result['specialty_detail']} | "
                f"{metadata_result['confidence']:.2f}"
            )

            # =============================================
            # STAGE 2 — WEB
            # =============================================

            if needs_web_search(
                metadata_result
            ):

                print(
                    "  Searching web..."
                )

                web_result = classify_with_web(
                    row
                )

                final_result = web_result
                source = "AI_WEB"

                print(
                    "  Web → "
                    f"{web_result['cuisine']} | "
                    f"{web_result['cuisine_detail']} | "
                    f"{web_result['specialty']} | "
                    f"{web_result['specialty_detail']} | "
                    f"{web_result['confidence']:.2f}"
                )

            # =============================================
            # FINAL RESULT
            # =============================================

            cuisine = final_result[
                "cuisine"
            ].strip()

            cuisine_detail = final_result[
                "cuisine_detail"
            ].strip()

            specialty = final_result[
                "specialty"
            ].strip()

            specialty_detail = final_result[
                "specialty_detail"
            ].strip()

            confidence = float(
                final_result[
                    "confidence"
                ]
            )

            evidence = final_result[
                "evidence"
            ].strip()

            # -------------------------------------------------
            # CUISINE
            # -------------------------------------------------

            if cuisine != "Unknown":

                places.at[
                    index,
                    "cuisine"
                ] = cuisine

            # -------------------------------------------------
            # CUISINE DETAIL
            # -------------------------------------------------

            if (
                cuisine_detail
                and cuisine_detail.lower()
                != "unknown"
            ):

                places.at[
                    index,
                    "cuisine_detail"
                ] = cuisine_detail

            # -------------------------------------------------
            # SPECIALTY
            # -------------------------------------------------

            if specialty != "Unknown":

                places.at[
                    index,
                    "specialty"
                ] = specialty

            # -------------------------------------------------
            # SPECIALTY DETAIL
            # -------------------------------------------------

            if (
                specialty_detail
                and specialty_detail.lower()
                != "unknown"
            ):

                places.at[
                    index,
                    "specialty_detail"
                ] = specialty_detail

            # -------------------------------------------------
            # QA
            # -------------------------------------------------

            if (
                confidence
                < AUTO_ACCEPT_THRESHOLD
                or cuisine == "Unknown"
            ):

                final_source = (
                    f"{source}_REVIEW"
                )

            else:

                final_source = source

            places.at[
                index,
                "food_classification_source"
            ] = final_source

            places.at[
                index,
                "food_classification_confidence"
            ] = confidence

            places.at[
                index,
                "food_classification_evidence"
            ] = evidence

            # Save progress after every classification
            places.to_csv(
                OUTPUT_FILE,
                index=False
            )

            print(
                f"  FINAL → "
                f"Cuisine: {cuisine} | "
                f"Detail: {cuisine_detail} | "
                f"Specialty: {specialty} | "
                f"Specialty detail: {specialty_detail} | "
                f"{confidence:.2f} | "
                f"{final_source}"
            )

        except Exception as error:

            print(
                f"  ERROR: {error}"
            )

            places.at[
                index,
                "food_classification_source"
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

    review_mask = (
        places[
            "food_classification_source"
        ]
        .fillna("")
        .str.contains(
            "REVIEW|ERROR",
            regex=True
        )
    )

    review = places[
        review_mask
    ].copy()

    review.to_csv(
        REVIEW_FILE,
        index=False
    )

    print(
        f"\nSaved → {OUTPUT_FILE}"
    )

    print(
        f"Review cases → "
        f"{REVIEW_FILE}"
    )


# =========================================================
# SUMMARY
# =========================================================

def print_summary(places):

    restaurants = places[
        places["category"]
        == "Restaurant"
    ].copy()

    print(
        "\n=============================="
    )

    print(
        "FOOD CLASSIFICATION SUMMARY"
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
            "food_classification_source"
        ].value_counts(
            dropna=False
        )
    )

    print(
        "\nCuisine coverage:"
    )

    print(
        f"{restaurants['cuisine'].notna().mean():.1%}"
    )

    print(
        "\nCuisine detail coverage:"
    )

    print(
        f"{restaurants['cuisine_detail'].notna().mean():.1%}"
    )

    print(
        "\nSpecialty coverage:"
    )

    print(
        f"{restaurants['specialty'].notna().mean():.1%}"
    )

    print(
        "\nSpecialty detail coverage:"
    )

    print(
        f"{restaurants['specialty_detail'].notna().mean():.1%}"
    )

    print(
        "\nTop cuisines:"
    )

    print(
        restaurants[
            "cuisine"
        ].value_counts()
        .head(25)
    )

    print(
        "\nTop specialties:"
    )

    print(
        restaurants[
            "specialty"
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