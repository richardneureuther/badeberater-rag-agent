# Badeberater — RAG-Powered Swimming Spot Agent for Lake Constance

A bilingual (German/English) AI agent that recommends swimming spots around Lake Constance (Bodensee). Built with RAG retrieval over scraped real-world data, live weather, and live water temperature,  with the LLM autonomously deciding which tools to call based on the user's question.

Example:
> **"Wo kann ich heute in Konstanz mit meinen Kindern schwimmen gehen?"**

```
Badeberater: In Konstanz gibt es einige schöne Möglichkeiten zum Schwimmen!

  • Strandbad Horn (78464 Konstanz) — Das größte Strandbad am Bodensee
    mit Familienbereich, Kinderbecken und Spielplatz. Eintritt frei.
    → https://www.bodensee.de/erleben/baden-im-bodensee/strandbad-horn

  • Strandbad Wallhausen (78465 Konstanz) — Flaches Ufer für Kinder,
    Beachvolleyball und Mini-Golf.
    → https://www.bodensee.de/erleben/baden-im-bodensee/strandbad-wallhausen

  Wetter in Konstanz: 17.3°C, teilweise bewölkt.
  Wassertemperatur: 12°C — eher frisch, nur für Hartgesottene.
```


## Architecture

```
┌──────────────────────────────────┐
│  AGENT (Gemini 2.5 Flash)        │  LLM decides which tools to call
│  via LangChain + LangGraph       │  based on the user's question
├──────────────────────────────────┤
│  TOOLS                           │
│  ├─ search_bathing_spots(query,  │  RAG retrieval from ChromaDB
│  │     city?)                    │  with optional city post-filter
│  ├─ get_weather(location)        │  Open-Meteo API (free, no key)
│  └─ get_water_temperature()      │  Scraped from wassertemperatur.org
├──────────────────────────────────┤
│  RAG INDEX (ChromaDB)            │  38 spots embedded with
│  built from bodensee_swimming    │  multilingual sentence-transformers
│  .jsonl                          │  (paraphrase-multilingual-MiniLM)
├──────────────────────────────────┤
│  DATA PIPELINE                   │  Custom scraper for bodensee.de
│  Scrape_Bodensee_RAG.py          │  → JSONL → ChromaDB
└──────────────────────────────────┘
```


## Design Decisions

**Why RAG instead of letting the LLM answer from memory?**
Without RAG, the LLM hallucinates plausible but wrong details inventing for example entry fees, opening hours, and features. RAG grounds every recommendation in real scraped data. The system prompt enforces this: *"Never recommend a spot purely from your own knowledge."*

**Why multilingual embeddings?**
Source data is German, but users may query in English or German. I chose `paraphrase-multilingual-MiniLM-L12-v2` over the default English-only models to handle both languages in the same vector space. Embeddings run locally on CPU, free, no API calls.

**Why one chunk per spot?**
Each swimming spot description is short and self-contained (~500–1500 chars). Splitting would fragment the semantic signal, a query for "Rutsche und Massage" might match the slide-chunk but miss the massage-chunk of the same spot. One chunk per spot keeps retrieval coherent.

**Why is this an agent and not just a RAG pipeline?**
The LLM autonomously decides which tools to call. A question like *"Is today good for swimming?"* should trigger both the spot search and the weather tool. A question like *"Tell me about Strandbad Horn"* only triggers the search. The orchestration is not hardcoded, the LLM reasons about which tools are needed based on the user's intent.


## Setup

```bash
# Clone and enter
git clone https://github.com/richardneureuther/badeberater-rag-agent.git
cd badeberater-rag-agent

# Create virtual environment
python -m venv venv
# Windows:
.\venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# Install dependencies
python -m pip install -r requirements.txt

# Add your Gemini API key (free tier from aistudio.google.com)
# Create a file called .env with:
# GOOGLE_API_KEY=your-key-here

# Build the RAG index (run once, ~30-90 seconds)
python build_index.py

# Start the agent
python agent.py
```
> **Note:** First startup takes 30-90 seconds while Python loads the ML libraries. Subsequent tool calls within the same session are faster. The free tier of Gemini 2.5 Flash allows ~2-5 user questions per day (each question uses 2-4 API calls internally). If you see a 429 error, the daily quota has been reached.

## Project Structure

```
badeberater-rag-agent/
├── agent.py                  # Main agent with Gemini + LangGraph
├── tools.py                  # Three tools: RAG search, weather, water temp
├── build_index.py            # Embeds JSONL into ChromaDB (run once)
├── Scrape_Bodensee_RAG.py    # Data pipeline: bodensee.de → JSONL
├── bodensee_swimming.jsonl   # 38 scraped swimming spot records
├── test_retrieval.py         # Isolated RAG retrieval tests
├── test_tools.py             # Tool integration tests
├── requirements.txt          # Python dependencies
├── .env                      # API key (git-ignored)
├── .gitignore
└── chroma_db/                # Persisted vector store (git-ignored)
```

## Tech Stack

- **LLM**: Google Gemini 2.5 Flash (free tier, via `langchain-google-genai`)
- **Agent framework**: LangChain + LangGraph (`create_react_agent`)
- **Vector database**: ChromaDB (local, persisted to disk)
- **Embeddings**: `paraphrase-multilingual-MiniLM-L12-v2` (local, CPU, free)
- **Weather**: Open-Meteo API (free, no key required)
- **Water temperature**: Scraped from wassertemperatur.org/bodensee/
- **Data source**: Custom scraper for bodensee.de (38 swimming spots)

The LLM provider is swappable, switching from Google to Anthorpic or OpenAI requires changing two lines (import + model name). Everything else is provider-agnostic via LangChain's abstractions.
