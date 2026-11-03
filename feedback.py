"""
HeartCare - usability feedback ikatha karna aur analyze karna.

- add_feedback : app ke andar se feedback CSV me (thread-safe, health inputs store nahi hote)
- summarize    : success rate, ease rating, device-wise, aur comments ke themes (manual keyword coding)
- sus_score    : System Usability Scale (10 sawaal) ka standard scoring
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
    "labels/jargon":  ["confus", "unclear", "jargon", "term", "medical", "understand", "samajh", "meaning", "label"],
    "length/scroll":  ["long", "scroll", "many", "too much", "lamba", "lambi", "zyada"],
    "mobile/touch":   ["phone", "mobile", "small", "touch", "tap"],
    "speed":          ["slow", "lag", "wait", "loading", "dhima"],
    "charts/results": ["chart", "graph", "bar", "shap", "result", "explanation"],
    "inputs":         ["slider", "type", "typing", "input", "field", "default"],
}


def _safe(text, limit=500):
    """CSV injection se bachao: =,+,-,@ se shuru text ke aage ' laga do. Length limit."""
    text = str(text).replace("\r", " ").replace("\n", " ").strip()[:limit]
    return "'" + text if text[:1] in ("=", "+", "-", "@") else text


def add_feedback(path, device, task_success, ease, comment="", version="v2"):
    """Ek feedback row save karo. Galat values par ValueError."""
    if task_success not in ("Yes", "Partly", "No"):
        raise ValueError("task_success: Yes / Partly / No")
    if not 1 <= int(ease) <= 5:
        raise ValueError("ease 1 se 5 ke beech hona chahiye")
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
    """Feedback ka summary. Kam responses (n<5) par conclusions pakke nahi hote, isliye n bhi return hota hai."""
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
    """SUS: 10 jawab (1-5). Odd sawaal: jawab-1, even sawaal: 5-jawab, total x 2.5 -> 0-100."""
    if len(answers) != 10 or any(not 1 <= a <= 5 for a in answers):
        raise ValueError("SUS ke liye 10 jawab chahiye, har ek 1 se 5")
    return sum((a - 1) if i % 2 == 0 else (5 - a) for i, a in enumerate(answers)) * 2.5
