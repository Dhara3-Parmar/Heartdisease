"""
HeartCare - usability feedback collection and analysis.

- add_feedback : thread-safe CSV logging (stores usability metrics without personal health data)
- summarize    : computes task success rate, mean ease score, device breakdown, and comment themes
- sus_score    : standard System Usability Scale (0-100) calculation
"""
import csv
import os
import threading
import time
from collections import Counter

import pandas as pd

FIELDS = ["time", "version", "device", "task_success", "ease_1to5", "comment"]
_LOCK = threading.Lock()

THEMES = {
    "labels/jargon":  ["confus", "unclear", "jargon", "term", "medical", "understand", "meaning", "label"],
    "length/scroll":  ["long", "scroll", "many", "too much"],
    "mobile/touch":   ["phone", "mobile", "small", "touch", "tap"],
    "speed":          ["slow", "lag", "wait", "loading"],
    "charts/results": ["chart", "graph", "bar", "shap", "result", "explanation"],
    "inputs":         ["slider", "type", "typing", "input", "field", "default"],
}


def _safe(text, limit=500):
    """Prevent CSV formula injection by prefixing =, +, -, @ with a single quote."""
    text = str(text).replace("\r", " ").replace("\n", " ").strip()[:limit]
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def add_feedback(path, device, task_success, ease, comment="", version="v2"):
    """Save a single feedback entry to CSV. Raises ValueError on invalid inputs."""
    if task_success not in ("Yes", "Partly", "No"):
        raise ValueError("task_success must be Yes / Partly / No")
    if not 1 <= int(ease) <= 5:
        raise ValueError("ease must be between 1 and 5")
    row = [time.strftime("%Y-%m-%d %H:%M:%S"), version, _safe(device, 30), task_success, int(ease), _safe(comment)]
    with _LOCK:
        new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(FIELDS)
            w.writerow(row)


def load_feedback(path):
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=FIELDS)


def summarize(df):
    """Aggregate usability feedback metrics and keyword themes."""
    if len(df) == 0:
        return {"n": 0}
    themes = Counter()
    for c in df["comment"].fillna("").str.lower():
        for name, words in THEMES.items():
            if any(w in c for w in words):
                themes[name] += 1
    return {"n": len(df),
            "success_rate_%": round((df["task_success"] == "Yes").mean() * 100, 1),
            "mean_ease": round(df["ease_1to5"].mean(), 2),
            "by_device": df.groupby("device")["ease_1to5"].agg(["count", "mean"]).round(2).to_dict("index"),
            "themes": dict(themes.most_common())}


def sus_score(answers):
    """Standard 10-item System Usability Scale (SUS) score in [0, 100]."""
    if len(answers) != 10 or any(not 1 <= a <= 5 for a in answers):
        raise ValueError("SUS requires 10 answers, each between 1 and 5")
    return sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers)) * 2.5
