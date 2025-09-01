## MODULES

import json
import pandas as pd
from src.python.parse import (
    read_news, load_or_create
)
from src.python.preanalysis import (
    generate, normalize_keywords, clean_news
)
from src.python.visualize import (
    viz_trend
)
from src.python.daily_highlights import (
    highlight_news, iter_by_day
)


## PROCEDURE

# Read the dataset
tbl = read_news("data/raw/data.csv")

# Extract top keywords and normalize them
norm_keywords = normalize_keywords(tbl)

# Parse the news feed then assign appropriate keywords and topics
tbl_clean = load_or_create(
    clean_news, tbl, "data/processed/preanalysis.csv", parse_dates = ["pubDateTime"]
)

# Subset the table
sub_tbl = tbl_clean.query("is_ina")

# Visualize the trend of unrest and violence reported in news
plt_trend = viz_trend(sub_tbl)
plt_trend.savefig("docs/fig/plt-trend-unrest.pdf", format="pdf", bbox_inches="tight")

# Generate daily news highlights
daily_highlight = iter_by_day(
    sub_tbl, highlight_news, model = "gemini-2.5-flash", schema = str
)

with open("data/processed/daily_highlight.json", "w", encoding="utf-8") as f:
    json.dump(daily_highlight, f, ensure_ascii = False, indent = 2)
