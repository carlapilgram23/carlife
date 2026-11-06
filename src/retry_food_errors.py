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

INPUT_FILE = Path(
    "data/processed/places_with_food_classification_final.csv"
)

OUTPUT_FILE = INPUT_FILE


# =========================================================
# OPENAI
# =========================================================

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found.")

client = OpenAI(api_key=API_KEY)


# =========================================================
# CONTROLLED TAXONOMIES
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
# STRUCTURED OUTPUT
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
                "type": "string",
            },
            "specialty": {
                "type": "string",
                "enum": ALLOWED_SPECIALTIES,
            },
            "specialty_detail": {
                "type": "string",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "evidence": {
                "type": "string",
            },
        },
        "required": [
            "cuisine",
            "cuisine_detail",
            "specialty",
            "specialty_detail",
            "confidence",
            "evidence",
        ],
        "additionalProperties": False,
    },
}


# =========================================================
# WEB CLASSIFICATION
# =========================================================

def classify_with_web(row):

    cuisines = ", ".join(ALLOWED_CUISINES)
    specialties = ", ".join(ALLOWED_SPECIALTIES)

    prompt = f"""
Research the EXACT restaurant below using web search.

Restaurant name:
{row.get("title")}

Google name:
{row.get("google_name")}

City:
{row.get("city")}

Neighborhood:
{row.get("neighborhood")}

Address:
{row.get("formatted_address")}

Google Maps URL:
{row.get("google_maps_uri")}

Original Google Maps URL:
{row.get("url")}

Google Place ID:
{row.get("google_place_id")}

Google types:
{row.get("google_types")}


First identify the exact establishment.

Prefer evidence from:
1. Official restaurant website
2. Official menu
3. Official social media
4. Reputable restaurant guides
5. Reputable editorial coverage


CUISINE

Cuisine MUST be exactly one of:

{cuisines}

Choose ONE broad analytical cuisine category.

Use cuisine_detail for nuance.


SPECIALTY

Specialty MUST be exactly one of:

{specialties}

Choose ONE specialty if the restaurant clearly has one.

Use specialty_detail for nuance.


RULES

1. Identify the exact establishment before classifying.
2. Do not infer cuisine from location alone.
3. Do not invent information.
4. Prefer first-party evidence.
5. cuisine must be one controlled category.
6. specialty must be one controlled category.
7. If cuisine cannot be established, use Unknown.
8. If specialty cannot be established, use Unknown.
9. Confidence should reflect the strength of the evidence.
10. Evidence should briefly explain what information supports the classification.
"""

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
                "content": prompt,
            },
        ],

        text={
            "format": FOOD_SCHEMA
        },
    )

    return json.loads(response.output_text)


# =========================================================
# LOAD
# =========================================================

places = pd.read_csv(INPUT_FILE)

errors = places[
    places["food_classification_source"] == "AI_ERROR"
].copy()

print(f"AI errors found: {len(errors)}\n")


# =========================================================
# RETRY
# =========================================================

for counter, (index, row) in enumerate(
    errors.iterrows(),
    start=1,
):

    print(
        f"[{counter}/{len(errors)}] "
        f"{row['title']}"
    )

    try:

        result = classify_with_web(row)

        cuisine = result["cuisine"].strip()
        cuisine_detail = result["cuisine_detail"].strip()

        specialty = result["specialty"].strip()
        specialty_detail = result["specialty_detail"].strip()

        confidence = float(result["confidence"])
        evidence = result["evidence"].strip()

        # ---------------------------------------------
        # SAVE CLASSIFICATION
        # ---------------------------------------------

        if cuisine != "Unknown":
            places.at[index, "cuisine"] = cuisine

        if (
            cuisine_detail
            and cuisine_detail.lower() != "unknown"
        ):
            places.at[
                index,
                "cuisine_detail"
            ] = cuisine_detail

        if specialty != "Unknown":
            places.at[
                index,
                "specialty"
            ] = specialty

        if (
            specialty_detail
            and specialty_detail.lower() != "unknown"
        ):
            places.at[
                index,
                "specialty_detail"
            ] = specialty_detail

        # ---------------------------------------------
        # SOURCE
        # ---------------------------------------------

        if confidence >= 0.85 and cuisine != "Unknown":
            source = "AI_WEB_RETRY"
        else:
            source = "AI_WEB_RETRY_REVIEW"

        places.at[
            index,
            "food_classification_source"
        ] = source

        places.at[
            index,
            "food_classification_confidence"
        ] = confidence

        places.at[
            index,
            "food_classification_evidence"
        ] = evidence

        # Save immediately
        places.to_csv(
            OUTPUT_FILE,
            index=False
        )

        print(
            f"  → Cuisine: {cuisine}"
        )

        print(
            f"  → Detail: {cuisine_detail}"
        )

        print(
            f"  → Specialty: {specialty}"
        )

        print(
            f"  → Specialty detail: "
            f"{specialty_detail}"
        )

        print(
            f"  → Confidence: "
            f"{confidence:.2f}"
        )

        print(
            f"  → Source: {source}\n"
        )

    except Exception as error:

        print(
            f"  ERROR AGAIN: {error}\n"
        )

    time.sleep(0.2)


# =========================================================
# FINAL SAVE + SUMMARY
# =========================================================

places.to_csv(
    OUTPUT_FILE,
    index=False
)

restaurants = places[
    places["category"] == "Restaurant"
].copy()

print("==============================")
print("FINAL FOOD SUMMARY")
print("==============================")

print(
    f"Restaurants: {len(restaurants)}"
)

print("\nClassification source:")

print(
    restaurants[
        "food_classification_source"
    ].value_counts(
        dropna=False
    )
)

print(
    "\nCuisine coverage:",
    f"{restaurants['cuisine'].notna().mean():.1%}"
)

print(
    "Cuisine detail coverage:",
    f"{restaurants['cuisine_detail'].notna().mean():.1%}"
)

print(
    "Specialty coverage:",
    f"{restaurants['specialty'].notna().mean():.1%}"
)

print(
    "Specialty detail coverage:",
    f"{restaurants['specialty_detail'].notna().mean():.1%}"
)

print(
    "\nRemaining errors:",
    (
        restaurants[
            "food_classification_source"
        ] == "AI_ERROR"
    ).sum()
)