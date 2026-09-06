"""
taxonomy.py
-----------
Controlled topic keyword vocabulary and the tag-based classifier,
`classify_tags()`.

Methodology (read before touching the keyword dictionaries):

1. Built by inspecting case-insensitive tag frequency counts on the
   cleaned dataset and, for every tag that could plausibly belong to
   more than one bucket, reading the actual articles carrying it.
   Categories were NOT decided in advance.

2. `classify_tags()` is MULTI-LABEL: an article is assigned to EVERY
   topic that has at least one matching tag, not just one "best" topic.
   An article tagged both "Sudan" and "Racism", for example, genuinely
   belongs to both "Middle East & Global Conflict" and "Society &
   Identity" -- picking only one would throw away real information
   about what that article covers.

3. Articles with no tags, or tags that don't match any topic's keyword
   set, are left UNCLASSIFIED (empty list). There is no section-based
   fallback and no forced default. An article that genuinely can't be
   placed is better excluded from the topic universe than silently
   forced somewhere, which would manufacture editorial signal that
   isn't really there.

4. Bare, highly generic tokens ("us", "uk", "news", "politics" alone)
   are EXCLUDED from every keyword set: inspecting their contexts
   showed they attach to wildly different topics (a "US" tag on a
   China trade-war piece, a film review, AND an Iran-conflict story).

5. TOPIC UNIVERSE REFINEMENTS (history, in the order they happened):
   - "Student Life & Wellbeing" and "Literature & Books" were dropped
     early on -- 2 articles each across the whole ~10-year archive is
     too sparse to ever produce a meaningful weekly baseline.
   - "Careers & Employability" is dropped for the same reason (too few
     articles to be useful in the quantitative topic universe).
   - "Editorial & Commentary" (the old section-based override for the
     recurring "Pi Perspective" round-up column) no longer exists as a
     topic at all, since this module is purely tag-driven with no
     section-level special-casing. Those round-up articles typically
     carry only generic tags (e.g. "UK News", "London News") that don't
     match any topic's keyword set, so they end up correctly
     unclassified rather than forced into a fake catch-all category.
   - "Culture, Society & Identity" (formerly a single merged topic, 67
     articles) was SPLIT into "Art & Culture" and "Society & Identity".
     The split was done by classifying each of the merged topic's 55
     keywords by MEANING ONLY (never by looking at which article the
     keyword was attached to): does the keyword describe a produced
     cultural artifact/event (-> Art & Culture) or social behavior,
     identity, or civic life (-> Society & Identity)? The result is
     lopsided by construction -- only 4 keywords ("culture",
     "nostalgia", "penny dreadfuls", "true crime") landed in Art &
     Culture, because the original merged bucket was overwhelmingly
     social/identity-themed to begin with; this is an honest
     consequence of the keyword vocabulary, not a design flaw. A few
     placements were genuine judgment calls rather than clear-cut:
     "journalism"/"media"/"news and media" went to Society & Identity
     (treated as a social institution, not creative cultural output),
     and "instagram"/"tiktok"/"internet"/"social media"/"memes" went to
     Society & Identity (treated as social platforms/behavior, not
     cultural artifacts). Both are defensible either way and are
     flagged here so they can be revisited.
   - TAXONOMY FROZEN AT 16 TOPICS: "Music" (5 articles) was dropped
     after a temporal-density check showed it was essentially dormant
     for 170 of 195 weeks in the modelling period, with all its
     activity clustered in a 3-month burst -- there is no meaningful
     week-to-week signal to rank. Music's 5 articles are NOT
     reassigned to any other topic -- they simply fall out of the
     topic universe, the same as any other unclassified article.
     "Art & Culture" (9 articles) and "Sport & Varsity" (24 articles)
     were kept despite also having long zero-streaks, because their
     activity is spread across the modelling period rather than
     concentrated in one short burst.
     This is the FINAL topic universe: no further additions, removals,
     or keyword changes are to be made based on how the signal or
     backtest later performs. Changing the topic universe to chase
     better backtest numbers would be a form of look-ahead bias on the
     research design itself, not just the data.
"""

from __future__ import annotations

TOPIC_KEYWORDS: dict[str, set[str]] = {

    "UK Politics": {
        "uk politics", "conservative party", "labour party", "labour",
        "tory party", "tory society", "tory", "conservatives",
        "conservativism", "reform uk", "keir starmer", "starmer",
        "uk government", "brexit", "prime minister", "parliament",
        "house of lords", "rule of law", "constitution", "supreme court",
        "andy burnham", "green party", "british politics",
        "united kingdom", "government",
    },

    "US & International Politics": {
        "us politics", "usa", "trump", "donald trump", "elections",
        "election", "2024 us presidential elections", "democracy",
        "ice", "us army", "mamdani", "epstein", "andrew mountbatten-windsor",
        "monarchy", "queen elizabeth ii", "modi", "india", "mexico",
        "north america", "united states",
    },

    "Middle East & Global Conflict": {
        "iran", "israel", "israel-gaza", "gaza", "gaza war", "syria",
        "syrian civil war", "yemen", "sudan", "lebanon", "war", "conflict",
        "middle east", "human rights", "humanitarian aid",
        "humanitarian crisis", "rwanda", "democratic republic of congo",
        "genocide", "darfur", "civil war", "airstrikes", "civil rights",
        "civil rights movement", "refugee", "slavery", "united nations",
        "aid",
    },

    "Global Affairs & Geopolitics": {
        "china", "russia", "france", "macron", "emmanuel macron",
        "europe", "european union", "eurozone", "hong kong", "tariffs",
        "trade war", "diplomacy", "international relations",
        "barnier", "michel barnier", "geopolitics", "kazakhstan",
        "bangladesh", "asia", "ireland", "world news",
    },

    "UCL & Campus": {
        "ucl", "ucl news", "ucl's student union", "student union",
        "students union", "ucl society", "campus", "ucl careers",
        "ucl culture", "ucl event", "university college london",
        "ucl student", "su", "su electon", "leadership race",
        "student union president", "ucl 200", "marxist society",
        "cultural societies", "leaders conference",
    },

    "Health & Medicine": {
        "health", "medicine", "mental health", "healthcare", "health care",
        "nhs", "cancer", "diabetes", "addiction", "medical negligence",
        "assisted dying", "insurance", "american healthcare",
        "united healthcare", "smoking", "smoke-free ucl", "medication",
    },

    "Education": {
        "education", "higher education", "decolonising the curriculum",
        "university", "academia", "international students", "grammar",
        "language",
    },

    "Economics & Finance": {
        "economics", "finance", "markets", "bank of england", "bonds",
        "budget", "cost of living", "world economy", "oil and gas",
    },

    "Climate & Environment": {
        "environment", "climate crisis", "climate change", "climate",
        "net zero", "sustainability", "la wildfires", "just stop oil",
        "renewable energy", "nuclear energy", "eco-friendly",
        "greenwashing", "consumerism", "invasive species", "squirrels",
        "wildlife", "endangered species", "animal trade", "greenland",
    },

    "Science & Technology": {
        "technology", "science", "ai", "artificial intelligence", "tech",
        "neuroscience", "3d printing", "robotics", "nanoparticles",
        "big tech", "laser", "eye", "lhara", "science of life",
        "royal society",
    },

    "Sport & Varsity": {
        "sport", "sports", "varsity", "varsity series", "varsity 2025",
        "ucl varsity", "ucl sports", "football", "women's football",
        "women's sports", "men's sports", "hockey", "rugby", "basketball",
        "tennis", "premier league", "champions league", "world cup",
        "cricket", "futsal", "karting", "ultimate frisbee",
        "american sports", "parasports", "paralympics", "promotion",
        "championship", "england football", "manchester united",
        "chelsea", "england",
    },

    "Film & Television": {
        "tv and film", "film", "movies", "movie", "netflix", "cinema",
        "movie review", "film review", "oscars", "awards", "awards season",
        "documentary", "bridgerton", "game of thrones", "streaming services",
        "hollywood", "bollywood", "actors", "paramount", "warner bros",
        "the academy", "polanski", "tv", "timothee chalamet",
    },

    "Theatre & Visual Arts": {
        "theatre", "london theatre", "exhibitions", "art", "museums and art",
        "ucl shows", "ucl dramasoc", "drama society", "tate britain",
        "shakespeare", "banksy", "play", "shows", "ucl arts",
        "international body of art", "graffiti", "public art",
        "museums", "exhibition", "bloomsbury theatre",
    },

    "Art & Culture": {
        "culture", "nostalgia", "penny dreadfuls", "true crime",
    },

    "Society & Identity": {
        "social media", "tiktok", "instagram", "gen z", "gen-z",
        "internet culture", "queer identity", "queer", "feminism",
        "misogyny", "representation", "dating apps", "dating",
        "desensitisation", "conspiracy", "political apathy",
        "islamophobia", "racism", "colonialism", "postcolonialism",
        "inequality", "human trafficking", "crime", "diversity", "lgbt",
        "lgbt history month", "identity", "internet", "generation",
        "millenials", "self-expression", "trans-rights", "privilege",
        "wealthy elite", "wealth disparity", "elitism", "academia",
        "memes", "censorship", "protests", "protest", "demonstration",
        "vigil", "voting", "vote", "debate", "journalism", "media",
        "news and media", "psychology", "black lives matter",
    },

    "Lifestyle, Food & Travel": {
        "food", "matcha", "ramen", "udon", "dessert", "cake", "acai",
        "ice cream", "night life", "london life", "fashion", "tattoos",
        "yoga", "wellness", "romance", "relationships", "friendship",
        "runway", "shoes", "positivity", "spring", "parks", "outdoor",
        "sunshine", "gossip", "confessions", "aesthetics",
        "restaurant review",
    },
}

# The final, frozen list of every topic in the quantitative universe --
# exactly 16 topics. "Music" is deliberately absent (see refinement
# note above); its articles are not reassigned anywhere. Used by
# features.py to build the complete weekly panel, including topics
# that end up with zero classified articles in a given week.
ALL_TOPICS = sorted(TOPIC_KEYWORDS.keys())


def classify_tags(tags_list: list[str]) -> list[str]:
    """Classify one article's tags against the controlled taxonomy.

    Multi-label: returns EVERY topic that has at least one matching tag
    (lowercased, whitespace-stripped comparison), not just a single
    "best" topic. An article with no tags, or whose tags don't match
    any topic's keyword set, returns an empty list -- it is genuinely
    unclassified, not forced into a default category.

    Example:
        classify_tags(["Sudan", "Racism"])
        -> ["Middle East & Global Conflict", "Society & Identity"]
    """
    lowered_tags = {tag.lower().strip() for tag in tags_list}

    matched_topics = []
    for topic, keywords in TOPIC_KEYWORDS.items():
        if lowered_tags & keywords:
            matched_topics.append(topic)

    return matched_topics