from pathlib import Path
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv


# =========================================================
# FILES
# =========================================================

INPUT_FILE = Path("data/processed/master_places.csv")
OUTPUT_FILE = Path("data/processed/master_places_enriched.csv")
REVIEW_FILE = Path("data/processed/places_to_review.csv")

RESOLVE_URL = "https://mapstools.googleapis.com/v1alpha:resolveMapsUrls"
PLACE_DETAILS_URL = "https://places.googleapis.com/v1/places"

BATCH_SIZE = 20


# =========================================================
# API KEY
# =========================================================

load_dotenv()

API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

if not API_KEY:
    raise ValueError(
        "GOOGLE_MAPS_API_KEY was not found. "
        "Check your .env file."
    )


# =========================================================
# PLACE DETAILS FIELD MASK
# =========================================================

DETAILS_FIELD_MASK = ",".join([
    "id",
    "displayName",
    "formattedAddress",
    "addressComponents",
    "location",
    "primaryType",
    "types",
    "businessStatus",
    "rating",
    "userRatingCount",
    "priceLevel",
    "googleMapsUri",
])


# =========================================================
# RESOLVE GOOGLE MAPS URL
# =========================================================

def resolve_maps_urls(urls):
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
    }

    body = {
        "urls": urls
    }

    response = requests.post(
        RESOLVE_URL,
        headers=headers,
        json=body,
        timeout=60,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# EXTRACT PLACE ID
# =========================================================

def extract_place_id(entity):
    resource = entity.get("place")

    if not resource:
        return None

    return resource.replace(
        "places/",
        ""
    )


# =========================================================
# PLACE DETAILS
# =========================================================

def get_place_details(place_id):
    url = f"{PLACE_DETAILS_URL}/{place_id}"

    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": API_KEY,
        "X-Goog-FieldMask": DETAILS_FIELD_MASK,
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()

    return response.json()


# =========================================================
# ADDRESS COMPONENTS
# =========================================================

def extract_address_components(address_components):
    result = {
        "neighborhood": None,
        "city": None,
        "state": None,
        "country": None,
        "country_code": None,
        "postal_code": None,
    }

    if not address_components:
        return result

    for component in address_components:
        types = component.get("types", [])
        long_text = component.get("longText")
        short_text = component.get("shortText")

        if (
            "neighborhood" in types
            and result["neighborhood"] is None
        ):
            result["neighborhood"] = long_text

        elif (
            "sublocality_level_1" in types
            and result["neighborhood"] is None
        ):
            result["neighborhood"] = long_text

        if "locality" in types:
            result["city"] = long_text

        elif (
            "administrative_area_level_2" in types
            and result["city"] is None
        ):
            result["city"] = long_text

        if "administrative_area_level_1" in types:
            result["state"] = long_text

        if "country" in types:
            result["country"] = long_text
            result["country_code"] = short_text

        if "postal_code" in types:
            result["postal_code"] = long_text

    return result


# =========================================================
# FLATTEN PLACE DETAILS
# =========================================================

def flatten_place_details(place_id, details):
    display_name = details.get(
        "displayName",
        {}
    )

    location = details.get(
        "location",
        {}
    )

    address = extract_address_components(
        details.get(
            "addressComponents",
            []
        )
    )

    return {
        "google_place_id": place_id,

        "google_name": display_name.get(
            "text"
        ),

        "google_maps_uri": details.get(
            "googleMapsUri"
        ),

        "formatted_address": details.get(
            "formattedAddress"
        ),

        "neighborhood": address[
            "neighborhood"
        ],

        "city": address[
            "city"
        ],

        "state": address[
            "state"
        ],

        "country": address[
            "country"
        ],

        "country_code": address[
            "country_code"
        ],

        "postal_code": address[
            "postal_code"
        ],

        "latitude": location.get(
            "latitude"
        ),

        "longitude": location.get(
            "longitude"
        ),

        "primary_type": details.get(
            "primaryType"
        ),

        "google_types": ", ".join(
            details.get(
                "types",
                []
            )
        ),

        "business_status": details.get(
            "businessStatus"
        ),

        "google_rating": details.get(
            "rating"
        ),

        "review_count": details.get(
            "userRatingCount"
        ),

        "price_level": details.get(
            "priceLevel"
        ),

        "match_status": "RESOLVED_FROM_URL",
    }


# =========================================================
# LOAD EXISTING PROGRESS
# =========================================================

def load_existing_progress():
    if not OUTPUT_FILE.exists():
        return pd.DataFrame()

    existing = pd.read_csv(
        OUTPUT_FILE
    )

    return existing


# =========================================================
# SAVE CHECKPOINT
# =========================================================

def save_checkpoint(df):
    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"Checkpoint saved → {OUTPUT_FILE}"
    )


# =========================================================
# PROCESS ONE BATCH
# =========================================================

def enrich_batch(batch):
    urls = batch["url"].tolist()

    resolution = resolve_maps_urls(
        urls
    )

    entities = resolution.get(
        "entities",
        []
    )

    enriched_rows = []

    for index, (_, row) in enumerate(
        batch.iterrows()
    ):
        print(
            f"  {row['title']}"
        )

        if index >= len(entities):
            enriched_rows.append({
                **row.to_dict(),
                "google_place_id": None,
                "match_status": "URL_RESOLUTION_FAILED",
            })

            continue

        entity = entities[index]

        place_id = extract_place_id(
            entity
        )

        if not place_id:
            enriched_rows.append({
                **row.to_dict(),
                "google_place_id": None,
                "match_status": "URL_RESOLUTION_FAILED",
            })

            continue

        try:
            details = get_place_details(
                place_id
            )

            enrichment = flatten_place_details(
                place_id,
                details
            )

        except requests.RequestException as error:
            print(
                f"    Details error: {error}"
            )

            enrichment = {
                "google_place_id": place_id,
                "match_status": "DETAILS_ERROR",
            }

        enriched_rows.append({
            **row.to_dict(),
            **enrichment,
        })

        time.sleep(0.1)

    return pd.DataFrame(
        enriched_rows
    )


# =========================================================
# ENRICH ALL PLACES
# =========================================================

def enrich_all_places(places):
    existing = load_existing_progress()

    if not existing.empty:
        processed_urls = set(
            existing["url"]
            .dropna()
            .astype(str)
        )

        remaining = places[
            ~places["url"]
            .astype(str)
            .isin(processed_urls)
        ].copy()

        print(
            f"Existing enriched places: "
            f"{len(existing)}"
        )

    else:
        remaining = places.copy()

    print(
        f"Remaining places: "
        f"{len(remaining)}"
    )

    if remaining.empty:
        print(
            "Nothing left to enrich."
        )

        return existing

    all_results = []

    if not existing.empty:
        all_results.append(
            existing
        )

    total_remaining = len(
        remaining
    )

    for start in range(
        0,
        total_remaining,
        BATCH_SIZE
    ):
        end = min(
            start + BATCH_SIZE,
            total_remaining
        )

        batch = remaining.iloc[
            start:end
        ].copy()

        print(
            f"\nResolving batch "
            f"{start + 1}-{end} "
            f"of {total_remaining}"
        )

        try:
            enriched_batch = enrich_batch(
                batch
            )

        except requests.RequestException as error:
            print(
                f"Batch error: {error}"
            )

            print(
                "Stopping safely. "
                "Progress already saved."
            )

            break

        all_results.append(
            enriched_batch
        )

        current = pd.concat(
            all_results,
            ignore_index=True
        )

        current = current.drop_duplicates(
            subset=["url"],
            keep="last"
        )

        save_checkpoint(
            current
        )

        time.sleep(0.5)

    final = pd.concat(
        all_results,
        ignore_index=True
    )

    final = final.drop_duplicates(
        subset=["url"],
        keep="last"
    )

    return final


# =========================================================
# SUMMARY
# =========================================================

def print_summary(enriched):
    print(
        "\n=============================="
    )

    print(
        "ENRICHMENT SUMMARY"
    )

    print(
        "=============================="
    )

    print(
        f"Total places: "
        f"{len(enriched)}"
    )

    print(
        "\nStatus:"
    )

    print(
        enriched[
            "match_status"
        ].value_counts(
            dropna=False
        )
    )

    resolved = (
        enriched[
            "match_status"
        ]
        .eq(
            "RESOLVED_FROM_URL"
        )
        .sum()
    )

    resolution_rate = (
        resolved
        / len(enriched)
    )

    print(
        f"\nResolution rate: "
        f"{resolution_rate:.1%}"
    )

    print(
        "\nBusiness status:"
    )

    if "business_status" in enriched.columns:
        print(
            enriched[
                "business_status"
            ].value_counts(
                dropna=False
            )
        )


# =========================================================
# SAVE REVIEW CASES
# =========================================================

def save_review_cases(enriched):
    review_cases = enriched[
        enriched["match_status"]
        != "RESOLVED_FROM_URL"
    ].copy()

    review_cases.to_csv(
        REVIEW_FILE,
        index=False,
    )

    print(
        f"\nPlaces requiring review: "
        f"{len(review_cases)}"
    )

    print(
        f"Saved → {REVIEW_FILE}"
    )


# =========================================================
# MAIN
# =========================================================

def main():
    places = pd.read_csv(
        INPUT_FILE
    )

    print(
        f"Loaded {len(places)} places"
    )

    enriched = enrich_all_places(
        places
    )

    save_checkpoint(
        enriched
    )

    save_review_cases(
        enriched
    )

    print_summary(
        enriched
    )

    print(
        "\nDone."
    )


if __name__ == "__main__":
    main()