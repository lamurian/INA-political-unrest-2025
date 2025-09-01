## MODULES

import os
import httpx
import pandas as pd
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import APIError
from ratelimit import limits, sleep_and_retry
from more_itertools import chunked
from tenacity import (
    retry, wait_exponential, stop_never, retry_if_exception_type
)


## RESPONSE MODEL

class Summary(BaseModel):
    rownum: int
    keyword: list[str]
    topic: str
    highlight: str
    summary: str
    is_unrest: bool
    is_ina: bool
    is_violent: bool


## FUNCTION

# Load dataset
def read_news(path):
    tbl = pd.read_csv(path)
    tbl_clean = tbl[~tbl["url"].isnull() | ~tbl["content"].isnull()]
    return tbl_clean

# Extract top keyword based on text frequency metrics
def extract_keywords(keyword_series, top_n = 1000):
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


## PROCEDURE

# Load env variables and activate Google Gemini clients
load_dotenv()
GENAI_KEY = os.getenv("GENAI_KEY")
client = genai.Client(api_key = GENAI_KEY)

# Read the dataset
tbl = read_news("data/raw/data.csv")

# Extract top keywords and normalize them
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

# Parse the news feed then assign appropriate keywords and topics
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

# Save the dataset into file
tbl_clean.to_csv("data/processed/preanalysis.csv", index = False)
