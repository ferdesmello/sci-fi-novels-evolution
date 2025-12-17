# paths.py
from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA = ROOT / "data"
ANSWERS = DATA / "answers"
BRUTE = DATA / "brute"
FILTERED = DATA / "filtered"
SAVED_PAGES = DATA / "saved_pages"
VARIABILITY_IN_ANSWERS = DATA / "variability_in_answers"

KEYS = ROOT / ".." / "KEYs"
SCRIPTS = ROOT / "scripts"

FIGURES = ROOT / "figures"