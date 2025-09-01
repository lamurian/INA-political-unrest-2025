SELECT
  pubDateTime,
  title,
  url,
  REGEXP_REPLACE(
    REGEXP_REPLACE( -- Remove unwanted lines
      ARRAY_TO_STRING(content, " "),
      r"(?mi)^(?:\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}|baca juga.*|share.*|tags.*|berita lainnya.*)\s*", 
      ""
    ),
    r"\r?\n", " " -- Replace newlines with space
  ) AS content,
  ARRAY_TO_STRING(keywords, "; ") AS keyword,
  REGEXP_CONTAINS(
    LOWER(ARRAY_TO_STRING(content, " ")),
    r"\b(demo(nstration|nstrations|nstrator(s)?)?|protest?(s|ers)?|anarki|anarch(y|ic)?|darurat|riot(s)?)\b"
  ) AS match_pattern
FROM `tonal-run-451811-t9.news_aggregator.curated_news`
WHERE pubDateTime >= "2025-08-24"
ORDER BY
  DATE(pubDateTime) ASC,
  match_pattern DESC
;
