from pathlib import Path
import pandas as pd


# =========================================================
# FILE
# =========================================================

INPUT_FILE = Path("data/processed/places_with_vibe.csv")


# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv(INPUT_FILE)

restaurants = df[
    df["category"] == "Restaurant"
].copy()

print(f"Restaurants loaded: {len(restaurants)}")


# =========================================================
# HELPERS
# =========================================================

def contains_any(value, options):
    """
    Returns True if the value contains any of the options.
    """

    if pd.isna(value):
        return False

    value = str(value).lower()

    return any(
        option.lower() in value
        for option in options
    )


def score_distance(value, target):
    """
    Gives a higher score when a restaurant's 1-5 score
    is close to what the user wants.

    Exact match = 5 points
    1 point away = 4 points
    2 points away = 3 points
    etc.

    Missing vibe information = 0 points.
    """

    if pd.isna(value):
        return 0

    distance = abs(float(value) - target)

    return max(0, 5 - distance)


# =========================================================
# RECOMMENDER
# =========================================================

def recommend(
    locations=None,
    cuisine=None,
    occasion=None,
    romantic=None,
    lively=None,
    formality=None,
    conversation=None,
    max_results=5,
):

    data = restaurants.copy()


    # =====================================================
    # HARD FILTERS
    # =====================================================

    # LOCATION

    if locations:

        data = data[
            data["neighborhood"].apply(
                lambda x: contains_any(x, locations)
            )
        ]


    # CUISINE

    if cuisine:

        data = data[
            data["cuisine"]
            .fillna("")
            .str.lower()
            == cuisine.lower()
        ]


    # If filters leave us with nothing

    if len(data) == 0:

        print("\nNo restaurants found with those filters.")

        return data


    # =====================================================
    # START SCORE
    # =====================================================

    data["recommendation_score"] = 0.0


    # =====================================================
    # OCCASION
    # =====================================================

    if occasion:

        occasion_match = (
            data["occasions"]
            .fillna("")
            .apply(
                lambda x:
                occasion.lower() in str(x).lower()
            )
        )

        data.loc[
            occasion_match,
            "recommendation_score"
        ] += 5


    # =====================================================
    # ROMANTIC SCORE
    # =====================================================

    if romantic is not None:

        data["recommendation_score"] += (
            data["romantic_score"]
            .apply(
                lambda x:
                score_distance(x, romantic)
            )
        )


    # =====================================================
    # LIVELY SCORE
    # =====================================================

    if lively is not None:

        data["recommendation_score"] += (
            data["lively_score"]
            .apply(
                lambda x:
                score_distance(x, lively)
            )
        )


    # =====================================================
    # FORMALITY SCORE
    # =====================================================

    if formality is not None:

        data["recommendation_score"] += (
            data["formality_score"]
            .apply(
                lambda x:
                score_distance(x, formality)
            )
        )


    # =====================================================
    # CONVERSATION SCORE
    # =====================================================

    if conversation is not None:

        data["recommendation_score"] += (
            data["conversation_score"]
            .apply(
                lambda x:
                score_distance(x, conversation)
            )
        )


    # =====================================================
    # GOOGLE RATING BONUS
    # =====================================================

    if "google_rating" in data.columns:

        data["recommendation_score"] += (
            pd.to_numeric(
                data["google_rating"],
                errors="coerce"
            )
            .fillna(0)
            * 0.5
        )


    # =====================================================
    # SORT RESULTS
    # =====================================================

    data = data.sort_values(
        "recommendation_score",
        ascending=False
    )


    # =====================================================
    # COLUMNS TO DISPLAY
    # =====================================================

    columns_to_show = [
        "google_name",
        "neighborhood",
        "cuisine",
        "google_rating",
        "price_level",
        "vibes",
        "occasions",
        "energy",
        "romantic_score",
        "lively_score",
        "formality_score",
        "conversation_score",
        "recommendation_score",
    ]


    # Only keep columns that actually exist

    columns_to_show = [
        column
        for column in columns_to_show
        if column in data.columns
    ]


    # =====================================================
    # RETURN TOP RESULTS
    # =====================================================

    return data[
        columns_to_show
    ].head(max_results)


# =========================================================
# TEST CARLIFE
# =========================================================

if __name__ == "__main__":

    results = recommend(

        # Roma + Condesa area
        locations=[
            "Roma Norte",
            "Roma Sur",
            "Colonia Condesa",
            "Hipódromo",
        ],

        # No cuisine filter for now.
        # We want Carlife to recommend based on the PLAN.
        cuisine=None,

        # Occasion
        occasion="Date Night",

        # Desired vibe
        romantic=4,
        lively=4,
        formality=3,
        conversation=4,

        # Number of recommendations
        max_results=5,
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print("\n")

    print("=" * 90)
    print("CARLIFE RECOMMENDATIONS")
    print("=" * 90)

    if results.empty:

        print("No recommendations found.")

    else:

        print(
            results.to_string(
                index=False
            )
        )