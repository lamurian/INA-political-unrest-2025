## MODULES

import pandas as pd
from src.python.preanalysis import generate


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
    refined_theme = {
        k: reanalyze(v, model = model, schema = schema)
        for k, v in daily_theme.items()
    }

    return refine_theme(refined_theme, schema, n - 1, model = model)
