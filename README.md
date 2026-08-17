# carlife
Exploring what AI can infer about me from 956 Google Maps saves and turning those insights into a personalized recommendation engine.

# 956 Places 🗺️

## What can AI learn about me from the places I save?

Over the years, I have saved **956 places on Google Maps**: restaurants, coffee shops, bars, stores, museums, and places across cities I've lived in, visited, or want to explore.

Individually, each save says very little.

Together, they create a **digital footprint**.

This project asks:

> **How much can AI infer about me from my Google Maps saves alone?**

## 📍 The Data

The project starts with my real Google Maps Takeout data:

| List      |  Places |
| :-------- | ------: |
| CDMX      |      85 |
| London    |     218 |
| Wanna go  |     653 |
| **Total** | **956** |

The raw data is included in this repository.

I enrich each place with information such as:

* Location and neighborhood
* Category and cuisine
* Price level and rating
* Reviews and recurring themes
* Atmosphere and occasion
* Semantic and behavioral tags

## 🏗️ The Pipeline

```text
956 Google Maps Saves
          ↓
     Clean + Enrich
          ↓
   Review Analysis
          ↓
  Feature Engineering
          ↓
 Master Places Dataset
          ↓
 ┌────────┼─────────┐
 ↓        ↓         ↓
Maps   Clusters    AI
 ↓        ↓      Inference
 └────────┼─────────┘
          ↓
     Taste Profile
          ↓
 Recommendation Engine
```

## 🧠 The Experiment

Once the dataset is built, I give AI **only the patterns contained in my saved places**.

No CV. No biography. No social media.

Can it infer:

* Where I've lived and traveled?
* What food I like?
* Which neighborhoods I gravitate toward?
* What price points and experiences I prefer?
* What my lifestyle might look like?
* What kind of traveler I am?

I then compare those inferences with reality to measure **what AI gets right, what it gets wrong, and why**.

## 🔎 Discovering My Taste

Using clustering, embeddings, NLP, and geospatial analysis, I explore whether algorithms can discover patterns in my taste **without me explicitly defining them**.

Do my saved places naturally form groups around certain cuisines, neighborhoods, price points, occasions, or experiences?

And how geographically narrow is **my version of a city**?

## 🎯 The 956 Place Problem

There is also a practical problem:

> **I have almost 1,000 saved places and can never remember where I wanted to go.**

So I use the same data to build a personalized recommendation engine.

Instead of:

> *What's a good restaurant in Mexico City?*

I want to ask:

> *Dinner in Roma or Juárez, casual but nice, good wine, not too expensive, and somewhere I've saved but haven't tried.*

The system uses my own taste patterns to rank the places that make the most sense.

## 🚀 From Data to Carlife

The same places intelligence can help power **Carlife**, making it easier to turn hundreds of saved locations into curated city guides.

```
Google Maps → Places Intelligence → Recommendations → Carlife Guides
```

AI helps me organize and discover.

**I decide what is worth recommending.**

## 🛠️ Tech

`Python` · `pandas` · `scikit-learn` · `NLP` · `Embeddings` · `Clustering` · `GeoPandas` · `LLMs` · `Recommendation Systems`

## 💭 The Bigger Question

A single Google Maps save means almost nothing.

Hundreds of them might reveal much more.

> **How much do our seemingly insignificant digital traces reveal about us?**

