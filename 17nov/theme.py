"""
HeartCare - custom CSS styling, WCAG AA accessible color tokens, and smooth animations/transitions.

Accessibility standards enforced:
  * Text contrast >= 4.5:1 (WCAG AA), UI components >= 3.0:1 -> verified programmatically via audit()
  * Non-color indicators: risk badges pair directional icons with text labels
  * Keyboard focus rings (:focus-visible) and >= 44px touch targets
  * Respects prefers-reduced-motion
"""

TOKENS = {
    "bg": "#f3f6fb", "card": "#ffffff", "navy": "#142b53", "blue": "#244577", "ink": "#1e293b", "muted": "#475569",
    "line": "#64748b", "soft_line": "#e2e8f0", "focus": "#1d4ed8", "risk": "#b42323", "safe": "#177245",
    "warn_ink": "#6b3a05", "warn_bg": "#fdf0d9", "on_navy": "#ffffff", "sub_on_navy": "#dbe4f5",
}

PAIRS = [
    ("body text on card", "ink", "card", 4.5), ("body text on page", "ink", "bg", 4.5),
    ("muted text on card", "muted", "card", 4.5), ("muted text on page", "muted", "bg", 4.5),
    ("title on card", "navy", "card", 4.5), ("header title on navy", "on_navy", "navy", 4.5),
    ("header subtitle on navy", "sub_on_navy", "navy", 4.5), ("button text on blue", "on_navy", "blue", 4.5),
    ("Higher Risk text on card", "risk", "card", 4.5), ("Lower Risk text on card", "safe", "card", 4.5),
    ("warning text", "warn_ink", "warn_bg", 4.5),
    ("input border on card (UI)", "line", "card", 3.0), ("focus ring on card (UI)", "focus", "card", 3.0),
    ("focus ring on page (UI)", "focus", "bg", 3.0), ("risk bar on card (UI)", "risk", "card", 3.0), ("safe bar on card (UI)", "safe", "card", 3.0),
]


def _lum(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(fg, bg):
    """Compute WCAG contrast ratio between two hex colors (1 to 21)."""
    a, b = sorted([_lum(fg), _lum(bg)], reverse=True)
    return (a + 0.05) / (b + 0.05)


def audit(tokens=None):
    """Evaluate all foreground/background color pairs against WCAG thresholds."""
    t = tokens or TOKENS
    return [{"pair": n, "fg": t[f], "bg": t[b], "ratio": round(contrast(t[f], t[b]), 2), "needed": need,
             "pass": contrast(t[f], t[b]) >= need} for n, f, b, need in PAIRS]


def risk_badge(decision, names=("Lower Risk", "Higher Risk")):
    """Accessible badge combining directional icon + text label."""
    icon = "&#9650;" if decision == 1 else "&#9660;"
    cls = "badge-risk" if decision == 1 else "badge-safe"
    return f'<span class="badge {cls}">{icon} {names[decision]}</span>'


def root_vars():
    return ":root {" + "".join(f" --{k.replace('_', '-')}: {v};" for k, v in TOKENS.items()) + " }"


CSS = """
<style>
__ROOT__

.stApp { background: var(--bg); color: var(--ink); }
header, footer { visibility: hidden; }
.main .block-container { max-width: 850px; padding: 25px 20px; animation: page-in .35s ease both; }
h1, h2, h3 { color: var(--navy); }
p, label, .stMarkdown, .stCaption, [data-testid="stCaptionContainer"] { color: var(--ink); }
[data-testid="stCaptionContainer"] { color: var(--muted); font-size: 14px; }

.header { background-color: var(--navy); background-image: linear-gradient(135deg, var(--navy), var(--blue)); color: var(--on-navy); padding: 25px; border-radius: 14px;
  text-align: center; margin-bottom: 25px; animation: fade-up .5s ease both; }
.header h1 { color: var(--on-navy); margin: 0 0 4px; }
.header p { color: var(--sub-on-navy); margin: 0; }
.section { background: var(--card); padding: 20px; border-radius: 12px; margin-bottom: 18px; border: 1px solid var(--soft-line);
  animation: fade-up .45s ease both; transition: box-shadow .2s ease, transform .2s ease; }
.section:hover { box-shadow: 0 6px 18px rgba(20, 43, 83, .10); transform: translateY(-1px); }
.section h3 { color: var(--navy); margin-top: 0; }
.page-card { display: block; background: var(--card); border: 1px solid var(--soft-line); border-radius: 12px; padding: 16px 18px; margin: 10px 0; }

.stButton button, .stDownloadButton button { background: var(--blue); color: var(--on-navy); width: 100%; min-height: 44px; border-radius: 8px;
  border: 2px solid var(--blue); font-weight: 600;
  transition: background .15s ease, transform .1s ease, box-shadow .15s ease; }
.stButton button:hover, .stDownloadButton button:hover { background: var(--navy); border-color: var(--navy); color: var(--on-navy);
  box-shadow: 0 4px 12px rgba(20, 43, 83, .25); transform: translateY(-1px); }
.stButton button:active { transform: translateY(0) scale(.98); }
.stButton button p, .stDownloadButton button p { color: var(--on-navy); }

:focus-visible { outline: 3px solid var(--focus) !important; outline-offset: 2px !important; }
input:focus, textarea:focus, [data-baseweb="select"]:focus-within { box-shadow: 0 0 0 3px rgba(29, 78, 216, .35) !important; }

.badge { display: inline-block; font-weight: 700; padding: 3px 10px; border-radius: 999px; font-size: 14px; }
.badge-risk { background: #fbe9e9; color: var(--risk); } .badge-safe { background: #e4f4ec; color: var(--safe); }
.notice { background: var(--warn-bg); color: var(--warn-ink); border-radius: 10px; padding: 10px 14px; margin: 8px 0; line-height: 1.5; }

[data-testid="stSidebarNav"] a { border-radius: 8px; transition: background .15s ease, padding-left .15s ease; }
[data-testid="stSidebarNav"] a:hover { background: #e3eafa; padding-left: 14px; }

@keyframes fade-up { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
@keyframes page-in { from { opacity: 0; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}

@media (max-width: 600px) {
  .main .block-container { padding: 12px 10px; }
  .header { padding: 18px; } .header h1 { font-size: 24px; }
  .section { padding: 14px; }
}
</style>
""".replace("__ROOT__", root_vars())


def css(extra=""):
    return CSS + extra


STREAMLIT_CONFIG = """[theme]
base = "light"
primaryColor = "#244577"
backgroundColor = "#f3f6fb"
secondaryBackgroundColor = "#ffffff"
textColor = "#1e293b"
"""
