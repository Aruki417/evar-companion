# E-VAR Companion
### AI-Powered VAR Explanation for Football Fans — IBM SkillsBuild AI Builders Challenge 2026

> *VAR made the call. Now understand why.*

[![IBM Granite](https://img.shields.io/badge/IBM%20Granite-watsonx.ai-052FAD?style=flat-square&logo=ibm)](https://www.ibm.com/watsonx)
[![StatsBomb](https://img.shields.io/badge/Data-StatsBomb%20Open%20Data-red?style=flat-square)](https://statsbomb.com/open-data/)
[![FIFA WC 2022](https://img.shields.io/badge/Tournament-FIFA%20World%20Cup%202022-gold?style=flat-square)]()
[![Languages](https://img.shields.io/badge/Languages-23-green?style=flat-square)]()

---

## The Problem

Every time VAR overturns a goal, the same thing happens: the referee walks to the monitor, 68 seconds pass in silence, and the decision is announced — with no explanation. For the billions of fans watching around the world, especially those new to football or watching in a second language, **VAR is a black box that erodes trust and alienates the audience football is trying to grow.**

No broadcaster explains the call in real time. No app breaks it down automatically. Fans are left to argue about something none of them fully understand.

---

## The Solution

**E-VAR Companion** is a zero-input, AI-powered VAR explainer. The moment VAR makes a call, the app:

1. **Auto-pauses** the match video at the exact VAR timestamp
2. **Draws** the offside line on the frozen frame
3. **Scores** the decision with a real ML danger probability (RandomForest, trained on 224 WC2022 offside events)
4. **Generates** a plain-language explanation using **IBM Granite on watsonx.ai** — grounded in real StatsBomb event data, not generic text
5. **Translates** everything automatically into **23 languages** — no user input required

Fan Mode keeps it simple. Pro Mode adds FIFA Law 11 citations and statistical depth. The same experience works for a casual viewer in Brazil and a football analyst in Japan.

---

## Live Demo

**Try it now (Streamlit — no setup needed):**
👉 **https://evar-companion.streamlit.app**

> ⚠️ **First load note:** The AI backend runs on Render's free tier and spins down after inactivity. The first IBM Granite explanation after a period of inactivity may take **up to 50 seconds** to respond — this is normal. Subsequent calls are instant. The full UI (all pages, charts, history, languages) loads immediately regardless.

**API backend (for judges who want to test directly):**
```
https://evar-companion.onrender.com/health
https://evar-companion.onrender.com/var-event/ARG_KSA_2022_E1?mode=fan&language=EN
https://evar-companion.onrender.com/var-event/ARG_KSA_2022_E1?mode=pro&language=FR
```

**Run locally:**
```bash
git clone https://github.com/Aruki417/evar-companion
cd evar-companion
pip install -r requirements.txt
export WATSONX_API_KEY="your-key"
export WATSONX_SPACE_ID="your-space-id"
export WATSONX_MODEL_ID="ibm/granite-4-h-small"
uvicorn evar_backend:app --port 8000 &
python3 -m http.server 3000
```
Open **http://localhost:3000**
```
Open **http://localhost:3000** (not file://).

---

## Features

| Feature | Detail |
|---|---|
| **Auto-pause VAR moments** | Video stops at 21s and 61s — the real ARG vs KSA VAR timestamps |
| **Offside line** | Drawn on the frozen frame at the moment of the pass |
| **ML danger score** | RandomForest trained on all 224 WC2022 StatsBomb offside events — 96% accuracy |
| **IBM Granite explanation** | Live text from `ibm/granite-3-3-8b-instruct` via watsonx.ai, grounded in real data |
| **Fan / Pro mode** | Plain language vs law-cited statistical breakdown |
| **23 languages** | Full UI + AI explanation translation — Arabic switches to RTL |
| **Fan Chat** | Ask any question about the VAR call — Granite answers in context |
| **History** | All 224 real offside events, named players, real minutes and distances |
| **Teams** | All 32 WC2022 nations, group stage, offside + card stats |
| **Laws DB** | All 17 FIFA laws with SVG illustrations + Granite analysis of Law 11 |

---

## Tech Stack

```
Frontend:     Single-file HTML/CSS/JS (index.html) — no framework, no build step
Backend:      FastAPI (Python) — evar_backend.py
AI model:     IBM Granite 3.3 8B Instruct — ibm/granite-3-3-8b-instruct via watsonx.ai
Fallback:     Ollama (granite3.2:2b) if watsonx is unreachable
ML model:     RandomForest (scikit-learn) — trained on StatsBomb WC2022 open data
ML explainer: SHAP (SHapley Additive exPlanations) — feature attribution per event
Data:         StatsBomb Open Data — FIFA World Cup 2022, match_id 3857300
Demo video:   Argentina vs Saudi Arabia, Group C, 22 Nov 2022
```

---

## Architecture

```
Browser (index.html @ :3000)
        │  GET /var-event/{id}?mode=fan|pro&language=EN
        │  GET /chat/{id}?q=...&mode=&language=
        ▼
FastAPI (evar_backend.py @ :8000)
        │
        ├─ Primary:  IBM watsonx.ai → granite-3-3-8b-instruct
        └─ Fallback: Ollama → granite3.2:2b (local)
```

The backend grounds every Granite prompt in real StatsBomb numbers — player name, offside margin in centimetres, distance from goal, ML danger probability, tournament percentile — so the AI explanation is always anchored to verified data, never hallucinated.

---

## ML Model Details

A **RandomForest classifier** (100 trees, max depth 4) was trained on **224 real offside events** from the FIFA World Cup 2022, sourced from the StatsBomb Open Data dataset.

**Features used:**
- Distance from goal (receiver location)
- Lateral distance from pitch centreline (centrality)
- Pass length
- Pass angle
- Match minute
- Period (first/second half)

**Results:**
- Accuracy: **96%**
- Top predictive factor: **distance from goal** (44.7% SHAP importance)
- Second: **central position** (33.2%)
- Third: **pass length** (15.1%)

SHAP values are computed per-event so each explanation reflects the specific factors that drove *that* prediction, not a generic average.

---

## The Two Demo VAR Events

| | Event 1 | Event 2 |
|---|---|---|
| **Player** | Lionel Messi | Lautaro Martínez |
| **Minute** | 22' | 28' |
| **Margin** | 3.0 cm (shoulder) | 3.2 cm (shoulder) |
| **Distance from goal** | 5.1 m | 20.1 m |
| **Zone** | Danger Zone | Shooting Range |
| **ML danger** | 58% | 77% |
| **Tournament percentile** | 94th | 66th |
| **Outcome** | Goal disallowed | Goal disallowed |

Both calls were among the most controversial of the tournament. ARG vs KSA finished with **11 offside events** — the most of any match at WC2022.

---

## Data & Credits

- **StatsBomb Open Data** — offside events, player locations, pass data
  Licensed under Creative Commons BY-SA 4.0. Used for educational/non-commercial purposes.
  https://github.com/statsbomb/open-data

- **FIFA World Cup Qatar 2022** — match data, competition structure
  competition_id: 43 · season_id: 106 · match_id: 3857300

- **IBM Granite** — foundation model for AI explanations
  Model: `ibm/granite-3-3-8b-instruct` via IBM watsonx.ai
  https://www.ibm.com/watsonx

- **Demo video** — Argentina vs Saudi Arabia, Group C, 22 Nov 2022
  Source: FIFA official match footage (YouTube: ZpvJ2OoCvlY)
  Used for educational demonstration purposes only.

---

## Running with Streamlit (for judges)

```bash
pip install streamlit
streamlit run streamlit_app.py
```

The Streamlit deployment wraps the full app with a public URL so judges can test without local setup.

---

## Repository Structure

```
var-companion/
├── index.html              # Full frontend — single file, all 7 pages
├── evar_backend.py         # FastAPI backend — watsonx.ai + Ollama
├── streamlit_app.py        # Streamlit wrapper for public deployment
├── var_data.py             # StatsBomb data pull + RandomForest training
├── requirements.txt        # Python dependencies
├── set_watsonx.sh          # Credentials template (never committed with real values)
├── .gitignore              # Excludes set_watsonx.sh and .venv
├── RUN_GRANITE.md          # Full setup guide
└── README.md               # This file
```

---

## Author

**Jessica** — MSc Business Analytics
IBM SkillsBuild AI Builders Challenge · June 2026

*Built with IBM Granite, StatsBomb Open Data, and a genuine frustration at watching VAR decisions with zero explanation.*
