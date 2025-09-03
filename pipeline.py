## MODULES

import os
import json
import httpx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from scipy.interpolate import make_interp_spline
from matplotlib.dates import DateFormatter
from src.python.parse import read_data
from pydantic import (BaseModel, RootModel)
from typing import List
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
from src.python.parse import (
    read_data, write_data, load_or_create, serialize
)
from src.python.preanalysis import (
    generate, normalize_keywords, clean_news
)
from src.python.visualize import (
    viz_trend
)
from src.python.daily_highlights import (
    iter_by_day, assign_highlight, assign_theme
)
from src.python.thematic_analysis import (
    refine_theme, assign_topic, tabulate_topic
)


## RESPONSE MODEL

class Summary(BaseModel):
    rownum: int
    keyword: List[str]
    topic: str
    highlight: str
    summary: str
    is_unrest: bool
    is_ina: bool
    is_violent: bool

class Theme(BaseModel):
    rownum: int
    kw: str
    thm: str
    rx_kw: str
    rx_thm: str

class Topic(BaseModel):
    topic: str
    linked_themes: List[str]
    rationale: str
    interpret: str

class DayThemes(BaseModel):
    date: str
    thm: List[str]
    topics: List[Topic]

class TopicReport(RootModel):
    root: List[DayThemes]


## PROCEDURE

# Read the dataset
tbl = read_data("data/raw/data.csv")

# Extract top keywords and normalize them
norm_keywords = load_or_create(
    FUN = normalize_keywords,
    path = "data/processed/keywords.json",
    params = {
        "normalize_keywords": {"tbl": tbl}
    }
)

# Parse the news feed then assign appropriate keywords and topics
tbl_clean = load_or_create(
    FUN = clean_news,
    path = "data/processed/preanalysis.csv",
    params = {
        "read_data": {"parse_dates": ["pubDateTime"]},
        "clean_news": {"tbl": tbl, "norm_keywords": norm_keywords}
    }
)

# Subset the table
plt_tbl = tbl_clean.query("is_ina")
sub_tbl = tbl_clean.query("is_ina and is_unrest")


# Visualize the trend of unrest and violence reported in news
plt_trend = viz_trend(plt_tbl)
plt_trend.savefig("docs/fig/plt-trend-unrest.pdf", format="pdf", bbox_inches="tight")

# Generate daily news highlights
daily_highlight = assign_highlight(
    sub_tbl, "data/processed/daily_highlight.json", schema = str
)

# Perform a daily thematic analysis
daily_theme = assign_theme(
    sub_tbl, "data/processed/daily_theme.json", schema = list[Theme]
)

# Refine the keyword and theme
refined_theme = load_or_create(
    FUN = refine_theme,
    path = "data/processed/daily_theme_refined.json",
    params = {
        "refine_theme": {
            "daily_theme": daily_theme,
            "schema": list[Theme],
            "n": 3,
            "model": "gemini-2.5-flash"
        }
    }
)

# Flatten the theme as a data frame
tbl_theme = pd.DataFrame(
    [
        {**theme, "date": date}
        for date, themes in serialize(refined_theme).items()
        for theme in themes
    ]
)

tbl_theme = tbl_theme.set_index("rownum")

# Determine the daily topic based on given themes
daily_topic = load_or_create(
    FUN = assign_topic,
    path = "data/processed/daily_topic.json",
    params = {
        "assign_topic": {
            "tbl": tbl_theme,
            "model": "gemini-2.5-flash",
            "schema": TopicReport
        }
    }
)

# Save the topic as a dataframe
tbl_topic = tabulate_topic(daily_topic)
