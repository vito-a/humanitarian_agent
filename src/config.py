from pathlib import Path
import os

# /home/jovyan/work
# Path.cwd()
ROOT = Path(os.getenv("AGENTS_DIR", "/home/jovyan/work"))
CURRENT_AGENT_DIR = ROOT / "humanitarian_agent"
DATA_DIR = CURRENT_AGENT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

DB_PATH = DATA_DIR / "rss.sqlite3"           # rename from ukraine.db
REPORTS_DIR = DATA_DIR / "reports"; REPORTS_DIR.mkdir(exist_ok=True, parents=True)
CACHE_DIR = DATA_DIR / "caches";  CACHE_DIR.mkdir(exist_ok=True, parents=True)

# Deterministic LLM defaults for MVP
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://host.docker.internal:1234/v1")
LMSTUDIO_MAIN_KEY = os.getenv("LMSTUDIO_MAIN_KEY", "lm-studio")
LMSTUDIO_LABELER_KEY = os.getenv("LMSTUDIO_LABELER_KEY", "lm-studio")
LMSTUDIO_SECTION_KEY = os.getenv("LMSTUDIO_SECTION_KEY", "lm-studio")

# Prefer smaller/faster model
# google/gemma-3-27b
# liquid/lfm2-1.2b
# google/gemma-3n-e4b
# liquid/lfm2-1.2b
# microsoft/phi-4-mini-reasoning
LMSTUDIO_MAIN_MODEL = os.getenv("LMSTUDIO_MAIN_MODEL", "liquid/lfm2-1.2b")
LMSTUDIO_LABELER_MODEL = os.getenv("LMSTUDIO_LABELER_MODEL", "liquid/lfm2-1.2b")
LMSTUDIO_SECTION_MODEL = os.getenv("LMSTUDIO_SECTION_MODEL", "google/gemma-3n-e4b")

# Enable/disable expensive LLM cleaning
CLEAN_WITH_LLM = True  # set True to use the llm_clean_text step

# LLM settings
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
TOP_P       = float(os.getenv("LLM_TOP_P", "0.95"))
MAX_TOKENS  = int(os.getenv("LLM_MAX_TOKENS", "2000"))
SEED        = int(os.getenv("LLM_SEED", "42"))  # LM Studio supports seed on newer builds

# LLM prompt settings
TITLE_PROMPT_MAX_LEN  = 300
DESC_PROMPT_MAX_LEN   = 1000
LEAD_PROMPT_MAX_LEN   = 1500
REPAIR_PROMPT_MAX_LEN = 3000

# LLM cleaning settings
LLM_CLEANED_TITLE_MAX_LEN = 300
LLM_CLEANED_DESC_MAX_LEN = 600
LLM_CLEANED_TEXT_MAX_LEN = 8000

# LLM sections generation settings
LLM_SECTION_TEXT_MAX_LEN = 8000
SECTIONS_MAX_CHARS = 1000

# Feed cache TTL (seconds) — default 1 day for MVP
HTTP_READ_TIMEOUT = 5
#86400
FEED_CACHE_TTL_SECONDS = int(os.getenv("FEED_CACHE_TTL_SECONDS", "172800"))

# RSS settings
MAX_ARTICLES = 15
MAX_KEY_POINTS = 2
PER_SECTION_CAP = 5
BODY_SUMMARY_MAX_LENGTH = 1000
DESCRIPTION_MAX_LENGTH = 8000
USE_RSS_DESCRIPTION = bool(int(os.getenv("USE_RSS_DESCRIPTION", "1")))
CATEGORY_BOOST_FACTOR = 2.0  # how much to multiply section weight for category matches
CORE_CATEGORY_BOOST   = 1.5  # if any 'core' keyword is found in categories, multiply total score by this

# Acceptance criteria knobs
MAX_DOC_AGE_DAYS = 180     # ≤ 6 months
ALLOW_DOMAINS = {
    # US and Western outlets (+ UN system). Keep tight for day-one.
    "news.google.com","reuters.com","apnews.com","npr.org","pbs.org","cnn.com",
    "abcnews.go","cbsnews.com","nbcnews.com","bloomberg.com","washingtonpost.com",
    "bbc.com","wsj.com","usatoday.com","latimes.com","politico.com","rferl.org",
    "un.org","reliefweb.int","unhcr.org","who.int","wfp.org","unmas.org",
    "savethechildren.org","foreignaffairs.com","nytimes.com","theguardian.com",
    "politico.eu","cbsnews.com","nbcnews.com","time.com","foxnews.com","latimes.com"
}
