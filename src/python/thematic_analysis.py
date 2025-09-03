## MODULES

import pandas as pd
from src.python.preanalysis import generate
from src.python.parse import serialize


## FUNCTION

# Reanalyze the themes and keywords
def reanalyze(daily_theme, **kwargs):
    prompt = """
    You are an expert in political science thematic analysis. Your task is to reduce and generalize keywords while ensuring consistent themes.
    
    # Procedure
    
    1. You will receive an array of entries containing:
       - `rownum`: row number of the original dataset
       - `kw`: keyword
       - `thm`: theme
       - `rx_kw`: rationale for the keyword
       - `rx_thm`: rationale for the theme
    
    2. Merge similar or overlapping keywords into a generalized keyword (max 3 words) based on their theme and rationales.  
       - One theme may contain multiple keywords.  
       - One keyword must only belong to one theme.  
    
    3. Continue merging until each theme contains about 3–5 generalized keywords. If needed, revise the theme name so it accurately represents the merged keywords.  
    
    4. For each entry, return:  
       - The original `rownum`  
       - The new generalized `kw` in ALL CAPS  
       - The revised or confirmed `thm` in ALL CAPS  
       - `rx_kw`: concise rationale (max 200 characters) stating which keywords were merged  
       - `rx_thm`: concise rationale (max 200 characters) explaining the generalized theme  
    
    # Rules
    
    1. Always check that a proposed keyword is relevant to its theme and rationales.  
    2. Keywords must be a maximum of 3 words.  
    3. Keywords and themes must be written in CAPITALS.  
    4. Every entry in the output must retain its `rownum`.  
    5. Each theme must contain AT LEAST 2 keywords.
    
    # Example
    
    Input:
    
    ```
    [
       {
         "rownum": 0,
         "kw": "PRABOWO ANTI-CORRUPTION",
         "thm": "GOVERNMENT INTEGRITY",
         "rx_kw": "The summary highlights President Prabowo's firm stance against corruption and his rejection of an amnesty request.",
         "rx_thm": "The entry focuses on the government's commitment to upholding integrity and fighting corruption at the highest level."
       },
       {
         "rownum": 1,
         "kw": "KEMNAKER EXTORTION CASE",
         "thm": "CORRUPTION INVESTIGATION",
         "rx_kw": "The summary details a specific extortion case involving a key suspect in the Ministry of Manpower.",
         "rx_thm": "The entry describes an ongoing investigation into a significant corruption case within a government ministry."
       }
    ]
    ```
    
    Output:
    
    ```
    [
       {
         "rownum": 0,
         "kw": "COMBATING CORRUPTION",
         "thm": "ANTI-CORRUPTION GOVERNANCE",
         "rx_kw": "PRABOWO ANTI-CORRUPTION and KEMNAKER EXTORTION CASE were merged",
         "rx_thm": "Theme revised to reflect generalized anti-corruption governance"
       },
       {
         "rownum": 1,
         "kw": "COMBATING CORRUPTION",
         "thm": "ANTI-CORRUPTION GOVERNANCE",
         "rx_kw": "PRABOWO ANTI-CORRUPTION and KEMNAKER EXTORTION CASE were merged",
         "rx_thm": "Theme revised to reflect generalized anti-corruption governance"
       }
    ]
    ```
    """
    prompt_analyze = [
        prompt, f"Theme Array:\n{daily_theme}"
    ]
    response = generate(content = prompt_analyze, **kwargs)
    
    return response

# Refine the themes and keywords
def refine_theme(daily_theme, schema, n = 3, model="gemini-2.5-flash"):
    """
    Recursively refine daily_theme values using refine_theme.

    Parameters
    ----------
    daily_theme : dict
        Dictionary with dates (str) as keys and values to refine.
    schema : type
        Expected schema for refine_theme (e.g., list[Theme]).
    n : int
        Number of recursive refinement iterations.
    model : str, optional
        Model name for refine_theme.

    Returns
    -------
    dict
        Refined daily_theme after n iterations.
    """
    if n <= 0:
        return daily_theme
    
    print(f"Iteration left: {n - 1}")

    refined_theme = {}
    for k, v in daily_theme.items():
        result = None
        while result is None:
            print(f"Refining theme for {k}...")
            result = reanalyze(v, model = model, schema = schema)
        refined_theme[k] = result

    return refine_theme(refined_theme, schema, n - 1, model = model)

# Transform the theme table into dictionary
def transform_theme(tbl):
    themes = [
        {date: thms}
        for date, thms in (
            tbl[["date", "thm"]]
            .drop_duplicates()
            .groupby("date")["thm"]
            .apply(list)
            .items()
        )
    ]
    return themes

# Assign topic to theme
def assign_topic(tbl, **kwargs):
    prompt = """
    You are an expert in qualitative analysis for political research. You will process an array of daily theme dictionaries.
    
    ## Input format
    
    Each entry is a dictionary where the key is a date (`YYYY-MM-DD`) and the value is an array of themes:
    
    ```
    [
      {"2025-08-24": ["ANTI-CORRUPTION GOVERNANCE", "LAW ENFORCEMENT ACTIONS", ...]},
      {"2025-08-25": ["PUBLIC HEALTH & WELLBEING", "NATIONAL UNITY & LEADERSHIP", ...]},
      ...
    ]
    ```
    
    ## Procedure
    
    1. For each date, review the `thm` array.  
    2. Merge themes that have overlapping or closely related meaning into a broader/generalized theme (max 3-5 words).  
    3. Group two or more linked themes into a single `topic`.  
       - A topic preferably includes at least two themes.  
       - One date may produce multiple topics if themes are not related.  
       - One date must have at least one topic.
    4. For each topic, provide a brief rationale (up to 200 characters, 1 sentence) explaining the link.  
    5. For each topic, provide a brief interpretation of what the topic and linked themes implies.
    
    ## Output format
    
    Return an array of dictionaries, one per date, with this structure:
    
    ```
    [
      {
        "2025-08-24": {
          "thm": ["ANTI-CORRUPTION GOVERNANCE", "LAW ENFORCEMENT ACTIONS", "NATIONAL UNITY & LEADERSHIP", ...],
          "topics": [
            {
              "topic": "GOVERNMENT ACCOUNTABILITY & LEGITIMACY",
              "linked_themes": ["ANTI-CORRUPTION GOVERNANCE", "LAW ENFORCEMENT ACTIONS"],
              "rationale": "Both themes emphasize state measures to strengthen trust through law enforcement.",
              "interpret": "Anti-corruption rules and fair enforcement are central to sustain government legitimacy."
            },
            {
              "topic": "LEADERSHIP",
              "linked_themes": ["NATIONAL UNITY & LEADERSHIP", ...],
              "rationale": "This theme stands alone, focusing on leadership symbolism for unity.",
              "interpret": "Leadership unites people, fostering national cohesion and guiding collective purpose."
            }
          ]
        }
      }
    ]
    ```
    
    ## Rules
    - Themes and topics must be written in ALL CAPS (max 3–5 words).  
    - Each topic must include a clear rationale.  
    - Process each date independently.  
    """
    themes = transform_theme(tbl)
    prompt_analyze = [
        prompt, f"Theme Array:\n{themes}"
    ]
    response = generate(content = prompt_analyze, **kwargs)
    return serialize(response)

# Tabulate the daily topic
def tabulate_topic(data):
    """
    Convert a nested JSON structure into a DataFrame with columns:
    - date
    - topic
    - theme (concatenated linked themes)
    
    Parameters
    ----------
    data : list
        List of dictionaries with date, thm, and topics.

    Returns
    -------
    pd.DataFrame
    """
    records = []

    for day in data:
        date = day["date"]
        for topic_entry in day["topics"]:
            topic = topic_entry["topic"]
            themes = "; ".join(topic_entry["linked_themes"])
            records.append({"date": date, "topic": topic, "theme": themes})

    df = pd.DataFrame(records)
    return df
