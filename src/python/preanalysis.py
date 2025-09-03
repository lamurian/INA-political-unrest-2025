## MODULES

import os
import httpx
from dotenv import load_dotenv
from collections import Counter
from google import genai
from google.genai import types
from google.genai.errors import APIError
from ratelimit import limits, sleep_and_retry
from more_itertools import chunked
from sklearn.feature_extraction.text import TfidfVectorizer
from tenacity import (
    retry, wait_exponential, stop_never, retry_if_exception_type
)


# Call Google Gemini to generate content
@retry(
    retry = retry_if_exception_type(
        (httpx.ConnectError, httpx.ReadTimeout, OSError, APIError)
    ),
    wait = wait_exponential(multiplier = 1, min = 4, max = 60),
    stop = stop_never,
    reraise = True
)
@sleep_and_retry
@limits(calls = 150, period = 60)
def generate(model, content, schema):
    """
    Call Google Gemini AI to generate structured content with retries, rate-limiting, 
    and exponential backoff.

    This function wraps the Gemini API call with the following features:
        1. **Retry**: If network or API errors occur, it retries indefinitely with 
           exponential backoff (1s * 2^n, capped between 4s and 60s). 
           Uses `tenacity.retry` for robust retry handling.
        2. **Rate-limiting**: Ensures no more than 150 requests per 60 seconds 
           using `ratelimit`.
        3. **Sleep between calls**: Automatically waits between calls to respect API limits.

    Args:
        model (str): Name of the Gemini model to use (e.g., "gemini-2.5-flash").
        content (str or list): Prompt(s) or content to send to the model.
        schema (type): A Pydantic model class or a list type describing the expected 
                       response format. The response will be parsed into this schema.

    Returns:
        Parsed response from Gemini, conforming to the provided schema. 
        The returned object can be:
            - `response.parsed` (if schema is a Pydantic model or list of models)
            - Structured JSON (if a JSON schema is provided)

    Example:
        ```python
        class Recipe(BaseModel):
            recipe_name: str
            ingredients: list[str]

        recipes = generate(
            model="gemini-2.5-flash",
            content="List a few popular cookie recipes.",
            schema=list[Recipe]
        )
        ```

    Notes:
        - The function loads the API key from environment variables (`GENAI_KEY`) 
          using `dotenv`.
        - Retries and rate-limiting decorators ensure robust execution for 
          large batch requests.
        - `temperature` is fixed to 0 for deterministic outputs.
    """

    # Load env variables and activate Google Gemini clients
    load_dotenv()
    GENAI_KEY = os.getenv("GENAI_KEY")
    client = genai.Client(api_key = GENAI_KEY)

    # Call Gemini AI model
    response = client.models.generate_content(
        model = model,
        contents = content,
        config = types.GenerateContentConfig(
            responseMimeType = "application/json",
            responseSchema = schema,
            temperature = 0
        )
    )

    result = response.parsed
    
    return result

# Extract top keyword based on text frequency metrics
def extract_keywords(keyword_series, top_n = 100):
    # Preprocess keywords
    keyword_list = keyword_series.fillna("").apply(
        lambda x: [k.strip().upper().replace(" ", "_") for k in x.split("; ") if k.strip()]
    )
    
    # Flatten list to calculate TF
    all_keywords = [k for sublist in keyword_list for k in sublist]
    tf_counter = Counter(all_keywords)
    
    # Prepare documents for IDF
    docs = keyword_list.apply(lambda x: " ".join(x)).tolist()
    vectorizer = TfidfVectorizer(use_idf=True, norm=None, smooth_idf=True, lowercase=False)
    X = vectorizer.fit_transform(docs)
    idf_scores = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    
    # Calculate TF-IDF
    tf_idf_scores = {k: tf_counter[k] * idf_scores.get(k, 0) for k in tf_counter.keys()}
    
    # Extract top_n for each metric
    top_tf = set([k for k, _ in sorted(tf_counter.items(), key=lambda x: x[1], reverse=True)[:top_n]])
    top_tf_idf = set([k for k, _ in sorted(tf_idf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]])
    
    # Find overlap
    overlap_keywords = top_tf & top_tf_idf  # intersection

    # Build the output dictionary
    result = [
        {
            "keyword": kw,
            "tf": tf_counter[kw],
            "idf": idf_scores.get(kw, 0),
            "tf_idf": tf_idf_scores.get(kw, 0)
        }
        for kw in overlap_keywords
    ]
    
    return result

# Extract top keywords and normalize them
def normalize_keywords(tbl):
    normalize = """
    You are a keyword normalization assistant. Given the following list of keywords:
    1. find similar keywords and merge them, 
    2. translate all keywords into English, 
    3. standardize naming using uppercase letters and underscores, 
    4. create umbrella/superset keywords where appropriate. 
    Return the result as a list of unique keywords.

    Input Keywords:
    """

    keywords = extract_keywords(tbl.keyword, top_n = 100)
    prompt_normalize = f"{normalize}\n```\n{[x['keyword'] for x in keywords]}\n```"
    norm_keywords = generate(
        model = "gemini-2.0-flash",
        content = prompt_normalize,
        schema = list[str]
    )

    return norm_keywords

# Parse the news feed then assign appropriate keywords and topics
def clean_news(tbl, norm_keywords):
    news = [
        {
            "rownum": row[0],
            "content": row[1]["content"]
        }
        for row in tbl.iterrows()
    ]

    analyze = """
    You are an expert news analyzer. You will receive a batch of 45 news articles. For each article, perform the following:

    1. Analyze the article content carefully.  
    2. Extract up to 10 relevant keywords that align with the provided reference keyword.  
    3. Infer the main topic of the article (3–5 words, very specific).  
    4. Infer the highlighted theme of the article (3–5 words, broader/general category).  
    5. Provide a concise summary (approx. 300 characters) that reflects the keywords, topic, and theme.  
    6. Indicate the presence of political unrest: answer strictly with `True` or `False`. 
    7. Indicate the relevance to the national situation: answer strictly with `True` or `False`.
    7. Indicate the presence of violence: answer strictly with `True` or `False`.

    Process each article independently but maintain a consistent style across the batch.
    """

    results = []
    for i, chunk in enumerate(chunked(news, 45)):
        print(f"Processing chunk index {i}")

        prompt_analyze = [
            analyze, f"Reference keyword: {norm_keywords}", f"Article:\n{chunk}"
        ]

        response = generate(
            model = "gemini-2.5-flash",
            content = prompt_analyze,
            schema = list[Summary]
        )

        results = results + response
        print(f"Finished!")

    # Convert the results into a data frame then merge to the original data
    tbl_summary = pd.DataFrame([r.dict() for r in results]).set_index("rownum")

    tbl_clean = tbl.drop(columns=["title", "keyword", "match_pattern"]) \
                   .merge(
                        tbl_summary,
                        left_index = True,
                        right_index = True,
                        how = "inner"
                   )

    tbl_clean["pubDateTime"] = pd.to_datetime(tbl_clean["pubDateTime"], errors = "coerce")
    tbl_clean["keyword"] = tbl_clean["keyword"].apply(lambda x: "; ".join(x))
    tbl_clean = tbl_clean.sort_values(by = "pubDateTime", ascending = True)

    return tbl_clean

