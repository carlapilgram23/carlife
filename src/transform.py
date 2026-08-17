import pandas as pd

places = pd.read_csv("data/processed/master_places_enriched.csv")

def categorize_place(place_type):

    if pd.isna(place_type):
        return "Other"

    # Restaurants & food
    if (
        "restaurant" in place_type
        or place_type in [
            "pizza_restaurant",
            "taco_restaurant",
            "sushi_restaurant",
            "sandwich_shop",
            "hamburger_restaurant",
            "bagel_shop",
            "meal_takeaway",
            "bar_and_grill",
            "deli",
            "bistro",
            "steak_house",
            "food_court",
        ]
    ):
        return "Restaurant"

    # Coffee, bakeries & tea
    if place_type in [
        "coffee_shop",
        "cafe",
        "cafeteria",
        "bakery",
        "coffee_roastery",
        "coffee_stand",
        "tea_house",
    ]:
        return "Coffee & Bakery"

    # Drinks & nightlife
    if place_type in [
        "pub",
        "irish_pub",
        "bar",
        "gastropub",
        "wine_bar",
        "cocktail_bar",
        "night_club",
    ]:
        return "Drinks & Nightlife"

    # Dessert
    if place_type in [
        "ice_cream_shop",
        "dessert_shop",
        "dessert_restaurant",
    ]:
        return "Dessert"

    # Food shopping
    if place_type in [
        "food_store",
        "grocery_store",
        "supermarket",
        "market",
    ]:
        return "Food Shopping"

    # Culture & things to do
    if place_type in [
        "museum",
        "art_museum",
        "art_gallery",
        "performing_arts_theater",
        "movie_theater",
        "tourist_attraction",
        "castle",
        "historical_landmark",
        "library",
        "live_music_venue",
        "hindu_temple",
    ]:
        return "Culture & Things to Do"

    # Outdoors
    if place_type in [
        "park",
        "national_park",
        "garden",
        "mountain_peak",
        "scenic_spot",
    ]:
        return "Outdoors"

    # Wellness & sports
    if place_type in [
        "gym",
        "spa",
        "yoga_studio",
        "fitness_center",
        "sports_complex",
        "sports_school",
        "nail_salon",
    ]:
        return "Wellness"

    # Hotels
    if place_type in [
        "hotel",
        "lodging",
    ]:
        return "Hotel"

    # Shopping
    if (
        "store" in place_type
        or "shop" in place_type
        or place_type == "garden_center"
    ):
        return "Shopping"

    return "Other"


places["category"] = places[
    "primary_type"
].apply(categorize_place)

print(
    places["category"].value_counts()
)

# Save final dataset
places.to_csv(
    "data/processed/places_final.csv",
    index=False
)

print(
    "\nSaved → data/processed/places_final.csv"
)