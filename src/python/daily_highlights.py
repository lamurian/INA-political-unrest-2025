## MODULES

import pandas as pd
from src.python.preanalysis import generate
from src.python.parse import (load_or_create, write_data, serialize)


## FUNCTION

# Run function per day group
def iter_by_day(tbl, FUN, **kwargs):
    """
    Iterate over each date in tbl['pubDateTime'], transform entries to a list of dicts,
    and call `FUN` for each day's data.

    Args:
        tbl: pandas DataFrame with at least 'pubDateTime' and 'summary' columns
        FUN: a function that takes a list of dicts [{rownum, summary}, ...]
                       and returns processed output
    Returns:
        dict: keyed by date, containing the output of FUN for that day
    """
    results_by_date = {}
    
    # Ensure pubDateTime is datetime
    tbl["pubDateTime"] = pd.to_datetime(tbl["pubDateTime"], errors="coerce")
    
    # Group by date
    for date, group in tbl.groupby(tbl["pubDateTime"].dt.date):
        date_str = date.strftime("%Y-%m-%d")
        print(f"Processing news from {date}")
        news = [
            {
                "rownum": idx,
                "summary": row["summary"],
                "topic": row["topic"],
                "highlight": row["highlight"]
            }
            for idx, row in group.iterrows()
        ]
        processed = FUN(news, **kwargs)
        results_by_date[date_str] = processed
        print(f"Finished!")
    
    return results_by_date

# Highlight an array of news
def highlight_news(news, **kwargs):
    """
    Highlight a batch of news articles to extract the main highlight.

    This function sends a structured prompt to a language model (via the `generate` function)
    asking it to analyze a list of news items and determine the key highlight or central theme
    that is most frequently reported in the array.

    Args:
        news (list[str] or list[dict]): 
            An array of news entries. Each entry is a string representing the news content.
            Example: 
            [
                "The government increased rice prices.",
                "Farmers are concerned about market stability."
            ]
        **kwargs: 
            Additional keyword arguments that are passed directly to the 
            `generate` function, such as:
                - `model`: The name of the AI model to use (e.g., "gemini-2.5-flash").
                - `schema`: The expected response schema (e.g., `str` or `list[str]`).

    Returns:
        The response from the `generate` function, which should contain the 
        main highlight extracted from the news array. The type of this response
        depends on the schema provided via `kwargs`.

    Example:
        ```python
        news_array = [
            {"rownum": 1, "summary": "Unrest in Jakarta continues."},
            {"rownum": 2, "summary": "Several protests occurred in other regions."}
        ]

        highlight = highlight_news(news_array, model="gemini-2.5-flash", schema=str)
        print(highlight)  # e.g., "Protests and unrest across regions"
        ```

    Notes:
        - The function constructs a prompt that instructs the AI to extract the main idea.
        - `generate` handles the API call, retries, and rate-limiting.
        - The output will typically be a concise string summarizing the key theme 
          of the input news array.
    """
    prompt = """
    You are a news analysis assistant. You will receive an array of news summaries. Your task is to identify and report the main highlight of the current news, focusing on the most frequently reported topic or theme. 

    The report must:
    - Be approximately 100 words.  
    - Directly describe the main highlight without preamble or introduction. Begin immediately with the report.  
    - Emphasize any instances of political unrest, protests, or reported violence.  
    - Explain why these events are occurring, including underlying causes or tensions indicated in the news.  
    - Capture the key trends, patterns, and significance of the news items collectively.  
    - Be factual and concise while providing sufficient context for understanding the events.
    """
    prompt_highlight = [
        prompt, f"News Array:\n{news}"
    ]
    response = generate(content = prompt_highlight, **kwargs)

    return response

# Thematic analysis on the news array
def analyze_news(news, **kwargs):
    prompt = """
    You are an expert political qualitative analyst. Your task is to conduct a fact-grounded thematic analysis. You will receive an array of entries, each containing a row number, news summary, topic, and highlight.  
    
    For each entry:
    
    1. Identify one keyword (up to 3 words) that best represents the entry. Write the keyword in ALL CAPS.  
    2. Assign a theme (can cover multiple keywords, but each keyword belongs to only one theme). Write the theme in ALL CAPS.  
    3. Provide a concise rationale for the keyword, max 200 characters, 1 sentence.  
    4. Provide a concise rationale for the theme, max 200 characters, 1 sentence.  
    
    Return results in the following JSON format:
    
    {
      "rownum": {row number},
      "kw": "{KEYWORD}",
      "thm": "{THEME}",
      "rx_kw": "{Concise rationale for keyword}",
      "rx_thm": "{Concise rationale for theme}"
    }
    
    Example input:
    
    ```
    {"rownum": 56, "summary": "Thousands, including ministers, participated in the free Merdeka Run 8.0K, part of Indonesia's 80th Independence Day celebrations. Starting at the Presidential Palace, the event featured two categories and an Air Force flypast, fostering national unity.", "topic": "Merdeka Run 8.0K Independence Day", "highlight": "National celebration and unity"}
    ```
    
    Example output:
    
    ```
    {"rownum": 56, "kw": "INDEPENDENCE DAY", "thm": "CELEBRATION", "rx_kw": "Recurring statements emphasizing the independence day", "rx_thm": "Summary has a celebratory remark towards the main keyword"}
    ```
    
    Process each entry independently but maintain a consistent style for all keywords and themes.
    """
    prompt_analyze = [
        prompt, f"News Array:\n{news}"
    ]
    response = generate(content = prompt_analyze, **kwargs)
    
    return response

# Provide a retry logic when generating content
def retry_by_day(FUN, tbl, path, model, schema):
    response = load_or_create(
        FUN = iter_by_day,
        path = path,
        params = {
            "iter_by_day": {
                "tbl": tbl,
                "FUN": FUN,
                "model": model,
                "schema": schema
            }
        }
    )

    while any(v is None for v in response.values()):
        print("Some values are missing...")

        for key, value in response.items():
            if value is None:
                print(f"Retrying {FUN.__name__} on {key}")
                response[key] = FUN(
                    tbl[tbl["pubDateTime"].dt.strftime("%Y-%m-%d") == key],
                    model = model,
                    schema = schema
                )

        print("Finished!")

    write_data(response, path)
    return response

# Assign highlight to the input news
def assign_highlight(tbl, path, schema):
    daily_highlight = retry_by_day(
        FUN = highlight_news,
        tbl = tbl,
        path = path,
        model = "gemini-2.0-flash",
        schema = schema
    )
    return serialize(daily_highlight)

# Assign the theme and persistently retry if not generated properly
def assign_theme(tbl, path, schema):
    daily_theme = retry_by_day(
        FUN = analyze_news,
        tbl = tbl,
        path = path,
        model = "gemini-2.5-flash",
        schema = schema
    )
    return serialize(daily_theme)
