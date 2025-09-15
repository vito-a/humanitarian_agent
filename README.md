# Humanitarian Agent

An offline, LLM-powered agent for generating structured humanitarian situation reports from trusted news and NGO feeds.
Designed for operations leads and analysts conducting due diligence on conflict-affected countries.

## Features

* **Data ingestion**: Scrapes and caches RSS feeds from Reuters, ReliefWeb/OCHA, UNHCR, WHO, and other vetted sources.
* **Processing pipeline**:

  * Pre-filters articles with keyword/domain checks.
  * Summarizes content using local LLMs for speed and reliability.
  * Extracts key humanitarian facts (casualties, infrastructure damage, healthcare strain, etc.).
* **Report generation**:

  * Outputs **DOCX** and **PDF** reports with narrative sections:
    * Summary
    * Ongoing conflict (by country & year)
    * Terror campaigns in most affected regions
    * Critical infrastructure damage
    * Supply interruptions (power, water, food)
    * Civilian casualties
    * Healthcare system strain
    * Why support is justified
    * Conclusion

  * Each section includes an **Evidence–Key Points** table:
    * **Evidence** = article title + clickable source
    * **Key Points** = concise facts and numbers

* **References**: Automatically generated with clickable links.
* **Deterministic outputs**: Uses fixed seeds/low temperature to minimize hallucinations.

## Guardrails

* Only uses credible Western sources (e.g., Reuters, Bloomberg, UN, US/UK gov).
* Every date/number is cited.
* No data older than 3 months.
* Inline citations (`[1]`) after each factual sentence.

## Packaging

* Runs fully offline.
* SQLite cache of feeds & summaries.
* Simple **Jupyter/Streamlit UI** for demos.
* Exportable DOCX and PDF outputs.

## Current Focus (MVP)

* Countries: **Ukraine**, **Syria**, **Yemen**
* Output: Briefing PDF with citations, generated live from cached docs.

---

## Quick Start

### 1. Prerequisites
- **Python** 3.10+  
- **pip** (or **uv**/**poetry**)  
- OS packages: `libxml2`, `libxslt` (for HTML parsing)  
- Disk space for cached feeds & models (a few GB)  
- (Optional) **Jupyter** and **Streamlit** for the demo UI

---

### 2. Clone & set up environment
```bash
git clone https://github.com/<your-org>/humanitarian-agent.git
cd humanitarian-agent

# Recommended: isolated env
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install Python deps
pip install -r requirements.txt
