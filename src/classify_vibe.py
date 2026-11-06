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

VIBE_FILE = Path("data/processed/places_with_vibe.csv")
FOOD_FILE = Path("data/processed/places_with_food_classification_final.csv")

# Resume if a previous run exists; otherwise start from the food classification
INPUT_FILE = VIBE_FILE if VIBE_FILE.exists() else FOOD_FILE

OUTPUT_FILE = VIBE_FILE

REVIEW_FILE = Path(
    "data/processed/vibe_to_review.csv"
)


# =========================================================
# SETTINGS
# =========================================================

# IMPORTANT:
# Start with only 10 restaurants.
# Once we review the results, we can increase/remove this.
TEST_LIMIT = 1000

MODEL = "gpt-5-mini"

REVIEW_THRESHOLD = 0.80


# =========================================================
# OPENAI
# =========================================================

load_dotenv(override=True)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found."
    )

client = OpenAI(
    api_key=API_KEY
)


# =========================================================
# CONTROLLED TAXONOMIES
# =========================================================

ALLOWED_VIBES = [
    "Intimate",
    "Cozy",
    "Lively",
    "Trendy",
    "Design-forward",
    "Elegant",
    "Casual",
    "Neighborhood",
    "Traditional",
    "Scenic",
    "Outdoor",
]


ALLOWED_OCCASIONS = [
    "Date Night",
    "Friends",
    "Group Dinner",
    "Business",
    "Family",
    "Special Occasion",
    "Brunch",
    "Quick Bite",
    "Drinks",
    "Solo",
]


ALLOWED_ENERGY = [
    "Quiet",
    "Relaxed",
    "Buzzing",
    "High-energy",
]


# =========================================================
# STRUCTURED OUTPUT
# =========================================================

VIBE_SCHEMA = {
    "type": "json_schema",
    "name": "restaurant_vibe_classification",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {

            "vibes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ALLOWED_VIBES,
                },
                "minItems": 1,
                "maxItems": 4,
            },

            "vibe_detail": {
                "type": "string",
            },

            "occasions": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ALLOWED_OCCASIONS,
                },
                "minItems": 1,
                "maxItems": 5,
            },

            "energy": {
                "type": "string",
                "enum": ALLOWED_ENERGY,
            },

            "romantic_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },

            "lively_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },

            "formality_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },

            "design_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },

            "group_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
            },

            "conversation_score": {
                "type": "integer",
                "minimum": 1,
                "maximum": 5,
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
            "vibes",
            "vibe_detail",
            "occasions",
            "energy",
            "romantic_score",
            "lively_score",
            "formality_score",
            "design_score",
            "group_score",
            "conversation_score",
            "confidence",
            "evidence",
        ],

        "additionalProperties": False,
    },
}


# =========================================================
# WEB CLASSIFICATION
# =========================================================

def classify_vibe_with_web(row):

    vibes = ", ".join(ALLOWED_VIBES)
    occasions = ", ".join(ALLOWED_OCCASIONS)
    energies = ", ".join(ALLOWED_ENERGY)

    prompt = f"""
Research the EXACT restaurant below using web search.

We are building Carlife, a restaurant recommendation engine.

The goal is NOT to classify the food.
The goal is to understand what it FEELS LIKE to visit this place
and what situations it is best suited for.


RESTAURANT

Name:
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

Cuisine:
{row.get("cuisine")}

Cuisine detail:
{row.get("cuisine_detail")}

Specialty:
{row.get("specialty")}

Price level:
{row.get("price_level")}

Google rating:
{row.get("google_rating")}

Review count:
{row.get("review_count")}


=========================================================
RESEARCH
=========================================================

First identify the exact establishment.

Research its:

- atmosphere
- interior design
- lighting
- seating
- noise / energy
- formality
- typical crowd
- whether people go for dates
- whether it works for groups
- whether conversation is easy
- terrace / garden / outdoor seating
- views if relevant
- whether it is considered trendy or buzzy
- whether it feels local / neighborhood-oriented
- whether it is traditional / old-school
- occasions for which reputable sources recommend it

Prefer evidence in this order:

1. Official restaurant website
2. Official restaurant social media
3. Reputable restaurant guides
4. Reputable editorial coverage
5. Multiple consistent review descriptions

Do NOT rely on one random review.


=========================================================
VIBES
=========================================================

Choose between 1 and 4 vibes.

Allowed values:

{vibes}


Definitions:

Intimate
= small-scale, close, low-key, good for personal conversation.

Cozy
= warm, comfortable, welcoming atmosphere.

Lively
= noticeable social energy, movement, noise or busy atmosphere.

Trendy
= fashionable, buzzy, currently popular or scene-driven.

Design-forward
= interior design or architecture is a distinctive and
meaningful part of the restaurant's identity or experience.

Do NOT select Design-forward merely because a place is
attractive, stylish, photogenic or Instagrammable.

A restaurant can have design_score = 3 without receiving
the Design-forward vibe.

Elegant
= polished, sophisticated or refined.

Casual
= informal, relaxed, low-formality.

Neighborhood
= feels like a local neighborhood spot or regular haunt.

Traditional
= classic, historic, old-school or rooted in tradition.

Scenic
= notable views or surroundings are an important part
of the experience.

Outdoor
= terrace, patio, garden or outdoor seating is a meaningful
part of the experience and a reason someone might choose
the venue.

Do NOT select Outdoor merely because a restaurant has
a few pavement or street-side tables.


IMPORTANT:

Do NOT select a vibe merely because it is plausible.

Select only vibes supported by evidence.

Maximum 4.


=========================================================
OCCASIONS
=========================================================

Choose between 1 and 5.

Allowed values:

{occasions}


Definitions:

Date Night
= a place you would actively recommend for a date because
the atmosphere meaningfully supports it.

Do NOT select Date Night merely because two people could
comfortably eat or drink there.

As a general consistency rule:
if romantic_score < 3, do not select Date Night.

Friends
= particularly suitable for a meal or outing with friends.

Group Dinner
= works well for larger groups.

Business
= appropriate for business lunch/dinner or professional meeting.

Family
= particularly suitable for families.

Special Occasion
= suitable for birthdays, anniversaries or celebrations.

Brunch
= meaningful brunch/breakfast destination.

Quick Bite
= suitable for a relatively quick/informal meal.

Drinks
= worthwhile destination for drinks as part of the experience.

Solo
= comfortable/suitable for eating alone.


Do not select every technically possible occasion.

Select occasions for which the restaurant is
particularly well suited.


=========================================================
ENERGY
=========================================================

Choose exactly one:

{energies}


Quiet
= subdued and notably calm.

Relaxed
= moderate/low energy but not silent.

Buzzing
= lively social atmosphere.

High-energy
= loud, crowded, party-like or highly energetic.


=========================================================
SCORES
=========================================================

All scores MUST be integers from 1 to 5.


ROMANTIC SCORE

1 = not romantic
2 = could work but not especially suited
3 = good casual date
4 = clearly good for date night
5 = exceptionally romantic


LIVELY SCORE

1 = very quiet
2 = calm
3 = moderate atmosphere
4 = lively
5 = very energetic / loud / party-like


FORMALITY SCORE

1 = extremely casual
2 = casual
3 = smart casual
4 = upscale
5 = formal / fine-dining level


DESIGN SCORE

1 = basic / design not relevant
2 = pleasant but ordinary
3 = attractive
4 = notable design
5 = exceptional architecture/interior design


GROUP SCORE

1 = poor for groups
2 = limited
3 = workable
4 = good for groups
5 = particularly strong for groups


CONVERSATION SCORE

Score based on how easy it is to sustain a conversation,
not on whether the venue feels intimate.

1 = very difficult to talk
2 = noisy / conversation somewhat difficult
3 = normal
4 = good for conversation
5 = exceptionally calm and easy for conversation

Do NOT assume intimate = quiet.

Music, tightly packed tables, crowd noise and high-energy
service should reduce this score.

A venue described as loud should normally score 1-2.

A venue described as buzzy should normally score no higher
than 3 unless there is strong evidence that conversation
remains easy.


=========================================================
VIBE DETAIL
=========================================================

Write a concise description of the actual atmosphere.

Good example:

"Compact candlelit wine bar with counter seating,
low lighting and a buzzy neighborhood atmosphere."

Bad example:

"Nice trendy restaurant with good vibes."

Be concrete.


=========================================================
CONFIDENCE
=========================================================

Confidence measures how strongly the available evidence
supports the classification.

0.90-1.00
= strong direct evidence from official/editorial sources.

0.80-0.89
= good consistent evidence.

0.60-0.79
= reasonable but incomplete evidence.

Below 0.60
= weak or ambiguous evidence.


=========================================================
EVIDENCE
=========================================================

Briefly explain what evidence drove the classification.

Mention concrete signals such as:

- candlelit dining room
- communal tables
- counter seating
- terrace
- DJ
- natural wine bar
- tasting-menu setting
- historic dining room
- reviewers consistently describe it as loud
- official site markets it for celebrations

Do NOT invent information.
"""

    response = client.responses.create(
        model=MODEL,

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
                    "You are classifying restaurant atmosphere "
                    "for a recommendation engine. "
                    "Research the exact establishment before "
                    "classifying it. "
                    "Use evidence rather than stereotypes based "
                    "on cuisine, neighborhood or price. "
                    "Be conservative."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],

        text={
            "format": VIBE_SCHEMA
        },
    )

    return json.loads(
        response.output_text
    )


# =========================================================
# LOAD DATA
# =========================================================

places = pd.read_csv(
    INPUT_FILE
)


# =========================================================
# CREATE COLUMNS
# =========================================================

new_columns = [
    "vibes",
    "vibe_detail",
    "occasions",
    "energy",
    "romantic_score",
    "lively_score",
    "formality_score",
    "design_score",
    "group_score",
    "conversation_score",
    "vibe_source",
    "vibe_confidence",
    "vibe_evidence",
]


for column in new_columns:

    if column not in places.columns:
        places[column] = None


# =========================================================
# SELECT RESTAURANTS
# =========================================================

restaurants = places[
    places["category"] == "Restaurant"
].copy()


# Only places that have not already been classified
to_classify = restaurants[
    restaurants["vibe_source"].isna()
].copy()


# =========================================================
# TEST SAMPLE
# =========================================================

# We deliberately want a diverse sample rather than simply
# taking the first 10 rows.

preferred_test_places = [
    "La Bonvi - Virreyes",
    "The Rose & Crown",
    "WatchHouse Somerset House",
    "The Ship",
    "Eno",
    "The Hunter's Moon",
    "Panadería Rosetta",
    "Helados Meñique",
    "Caravel",
    "Whiski Rooms",
]


available_test = to_classify[
    to_classify["title"].isin(
        preferred_test_places
    )
].copy()


# If some names are not present, fill the sample randomly.
if len(available_test) < TEST_LIMIT:

    already_selected = set(
        available_test.index
    )

    remaining = to_classify[
        ~to_classify.index.isin(
            already_selected
        )
    ]

    needed = (
        TEST_LIMIT
        - len(available_test)
    )

    if len(remaining) > 0:

        extra = remaining.sample(
            n=min(
                needed,
                len(remaining)
            ),
            random_state=42,
        )

        available_test = pd.concat(
            [
                available_test,
                extra,
            ]
        )


if TEST_LIMIT is None:
    to_process = to_classify
else:
    to_process = available_test.head(
        TEST_LIMIT
    )


print(
    f"Restaurants available: "
    f"{len(restaurants)}"
)

print(
    f"Restaurants requiring vibe classification: "
    f"{len(to_classify)}"
)

print(
    f"Processing now: "
    f"{len(to_process)}\n"
)


# =========================================================
# CLASSIFY
# =========================================================

for counter, (index, row) in enumerate(
    to_process.iterrows(),
    start=1,
):

    print(
        f"[{counter}/{len(to_process)}] "
        f"{row['title']}"
    )

    try:

        result = classify_vibe_with_web(
            row
        )

        vibes = result[
            "vibes"
        ]

        occasions = result[
            "occasions"
        ]

        vibe_detail = result[
            "vibe_detail"
        ].strip()

        energy = result[
            "energy"
        ]

        confidence = float(
            result["confidence"]
        )

        evidence = result[
            "evidence"
        ].strip()


        # =================================================
        # SAVE VALUES
        # =================================================

        places.at[
            index,
            "vibes"
        ] = " | ".join(vibes)

        places.at[
            index,
            "vibe_detail"
        ] = vibe_detail

        places.at[
            index,
            "occasions"
        ] = " | ".join(occasions)

        places.at[
            index,
            "energy"
        ] = energy


        score_columns = [
            "romantic_score",
            "lively_score",
            "formality_score",
            "design_score",
            "group_score",
            "conversation_score",
        ]

        for column in score_columns:

            places.at[
                index,
                column
            ] = int(
                result[column]
            )


        # =================================================
        # SOURCE
        # =================================================

        if confidence >= REVIEW_THRESHOLD:

            source = "AI_WEB"

        else:

            source = "AI_WEB_REVIEW"


        places.at[
            index,
            "vibe_source"
        ] = source

        places.at[
            index,
            "vibe_confidence"
        ] = confidence

        places.at[
            index,
            "vibe_evidence"
        ] = evidence


        # Save after every restaurant
        places.to_csv(
            OUTPUT_FILE,
            index=False
        )


        # =================================================
        # PRINT RESULT
        # =================================================

        print(
            "  Vibes:",
            " | ".join(vibes)
        )

        print(
            "  Occasions:",
            " | ".join(occasions)
        )

        print(
            f"  Energy: {energy}"
        )

        print(
            "  Scores:",
            f"romantic={result['romantic_score']},",
            f"lively={result['lively_score']},",
            f"formal={result['formality_score']},",
            f"design={result['design_score']},",
            f"group={result['group_score']},",
            f"conversation={result['conversation_score']}",
        )

        print(
            f"  Detail: {vibe_detail}"
        )

        print(
            f"  Confidence: "
            f"{confidence:.2f}"
        )

        print(
            f"  Source: {source}\n"
        )


    except Exception as error:

        print(
            f"  ERROR: {error}\n"
        )

        places.at[
            index,
            "vibe_source"
        ] = "AI_ERROR"

        places.to_csv(
            OUTPUT_FILE,
            index=False
        )


    time.sleep(0.2)


# =========================================================
# SAVE
# =========================================================

places.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# REVIEW FILE
# =========================================================

review = places[
    places["vibe_source"].isin(
        [
            "AI_WEB_REVIEW",
            "AI_ERROR",
        ]
    )
].copy()


review.to_csv(
    REVIEW_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

restaurants_final = places[
    places["category"] == "Restaurant"
].copy()


classified = restaurants_final[
    restaurants_final["vibe_source"].notna()
]


print(
    "\n=============================="
)

print(
    "VIBE CLASSIFICATION SUMMARY"
)

print(
    "=============================="
)

print(
    f"Restaurants: "
    f"{len(restaurants_final)}"
)

print(
    f"Vibe classified: "
    f"{len(classified)}"
)

print(
    "\nClassification source:"
)

print(
    restaurants_final[
        "vibe_source"
    ].value_counts(
        dropna=False
    )
)


if len(classified) > 0:

    print(
        "\nAverage scores:"
    )

    for column in [
        "romantic_score",
        "lively_score",
        "formality_score",
        "design_score",
        "group_score",
        "conversation_score",
    ]:

        values = pd.to_numeric(
            classified[column],
            errors="coerce",
        )

        print(
            f"{column}: "
            f"{values.mean():.2f}"
        )


print(
    f"\nSaved → {OUTPUT_FILE}"
)

print(
    f"Review cases → {REVIEW_FILE}"
)