# carlife
**places for every plan**

I have over 1,000 places saved on Google Maps: restaurants, cafés, bars, museums, little shops. And I never remember where I wanted to go.

carlife turns those saved places into recommendations. Instead of googling "restaurants in la Roma", I give it a plan (*la Roma, with friends, something casual*) and it ranks my own saved places.

## Status

| Step | What it does | Status |
|---|---|---|
| Clean | Combines my Google Maps Takeout lists into one dataset (956 places in this version) | ✅ Done |
| Enrich | Adds location, neighborhood, type, price and rating with the Google Places API | ✅ Done |
| Categorize | Groups Google place types into simple categories (restaurant, café, bar…) | ✅ Done |
| Food style | Rules from Google types first; AI with web search for the rest; manual overrides on top | ✅ Done |
| Vibe | AI classifies each restaurant's vibe (cozy, lively, casual, elegant…) | 🟡 In progress |
| Recommender | Filters by neighborhood and cuisine, then ranks by occasion (friends, date night…) and how casual, lively or formal I want it | 🟡 First version |
| What AI can guess about me | Give AI only my saved places and compare its guesses with reality | ⏳ Next |

## How it works

```text
Google Maps saves
      ↓
Clean + enrich (Google Places API)
      ↓
Classify food style and vibe (rules → AI → my review)
      ↓
Recommender
```

**One rule from the start:** AI only decides on its own when it's confident. Everything else goes to a review file, and I correct it by hand (`data/manual/food_classification_overrides.csv`).

## Run it

```bash
python src/clean.py
python src/enrich.py
python src/transform.py
python src/classify_food_style.py
python src/apply_overrides.py
python src/classify_vibe.py
python src/recommender.py
```

API keys go in a `.env` file (not included):

```text
GOOGLE_MAPS_API_KEY=...
OPENAI_API_KEY=...
```

## Tech
`Python` · `pandas` · `Google Places API` · `OpenAI API (web search, structured outputs)`

## The bigger question
A single saved place means almost nothing. Hundreds of them probably say more about us than we think. That's the next part of this project.

## The experiment (next)

Once the dataset is complete, I'll give AI **only my saved places**. No CV, no bio, no social media.

Can it guess where I've lived, what food I like, which neighborhoods I gravitate toward, or what kind of traveler I am?

Then I'll compare its guesses with reality: what it gets right, what it gets wrong, and why.