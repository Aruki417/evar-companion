# E-VAR Companion 🟡⚽

**AI-powered VAR offside explainer for football fans**
IBM SkillsBuild AI Builders Challenge · June 2026

**Demo video:** [Watch Demo](https://drive.google.com/file/d/1_vV9ufHR4JpMKFfRmijwwhhY7jeZe0En/view?usp=sharing)
[![IBM Granite](https://img.shields.io/badge/AI-IBM%20Granite%20via%20watsonx.ai-blue)](https://www.ibm.com/watsonx)
[![Languages](https://img.shields.io/badge/Languages-23-green)]()

---

## The Problem We Are Solving

Every week, billions of football fans watch VAR decisions they do not understand.

A goal gets disallowed. The crowd erupts. The screen shows a freeze-frame. The commentator says *offside* — and the broadcast moves on. No explanation. No context. No answer to the question every fan is asking: **why?**

At the 2022 FIFA World Cup alone there were over **560 VAR decisions** — 224 offsides, 29 penalty reviews, 4 red cards, handballs, fouls. Each one triggered confusion in stadiums and living rooms across the world, in dozens of languages and cultures.

The data existed. The technology existed. But no one had built the bridge between the two.

VAR was introduced to make football fairer. The unintended consequence is that it made football *less understandable* for the fans who love it most. Transparency is not optional — it is owed to every supporter in every language who watches the game.

---

## Our AI Solution — E-VAR Companion

E-VAR Companion is a real-time AI assistant that automatically detects VAR moments during a match and instantly explains every decision — in plain language, to any fan, in any language.

The **E** in E-VAR stands for **Explainer**.

### How It Works

**Auto-detection:** The webpage monitors the match video and automatically pauses when a VAR moment occurs. No button. No user input. The AI activates itself.

**IBM Granite via watsonx.ai:** Every VAR event is sent to IBM Granite (`ibm/granite-4-h-small`) via the watsonx.ai API. Granite analyzes the decision, applies the relevant FIFA law, evaluates the danger context, and produces a structured explanation in seconds.

**Dual audience modes:**
- **Fan Mode** — plain language, clear numbers, no jargon. Designed for any supporter regardless of football knowledge.
- **Pro Mode** — full analytical breakdown including SHAP factor decomposition, confidence percentile, distance from goal, body part tracking, and law reference. Designed for analysts, coaches, and media professionals.

**Interactive Fan Chat:** Users can ask follow-up questions in natural language — *"Why was this offside?"*, *"What does Law 11 mean?"*, *"How close was it?"* — and Granite answers using the exact context of the current decision.

**23 Languages:** The full interface, AI responses, VAR explanations, and chat are available in English, Spanish, French, Portuguese, Arabic, German, Japanese, Chinese, Italian, Dutch, Turkish, Polish, Korean, Hindi, Swedish, Danish, Norwegian, Finnish, Hungarian, Romanian, Croatian, Czech, and Russian.

**VAR History:** Every decision from the 2022 FIFA World Cup is logged and searchable — filterable by decision type, team, match, and game phase.

**Teams Dashboard:** Per-team statistics with drill-down into offsides, yellow cards, and red cards by match, showing patterns across the tournament.

**Laws DB:** Every FIFA law visualized with pitch diagrams and plain-language explanations — no rulebook required.

### Technical Architecture

```
User (browser) → index.html (single-file frontend, localhost:3000)
                      ↓
              FastAPI Backend (localhost:8000 / Render)
                      ↓
           watsonx.ai API → IBM Granite ibm/granite-4-h-small
                      ↓
        StatsBomb Open Data (WC2022 match_id 3857300)
```

| Component | Technology |
|---|---|
| Frontend | Single-file HTML · Vanilla JS · Chart.js |
| Backend | Python · FastAPI · deployed on Render |
| AI Model | IBM Granite (`ibm/granite-4-h-small`) via watsonx.ai |
| ML Scoring | Random Forest Classifier · SHAP attribution |
| Data source | StatsBomb Open Data · WC2022 · match_id 3857300 |
| Demo detection | Autopilot VAR trigger on local `clip.mp4` |
| Deployment | Streamlit at evar-companion.streamlit.app |

### How We Get the 3cm Offside Margin

The precision of the Detection page comes directly from StatsBomb Open Data.

StatsBomb tracked **29 body points per player at 50 frames per second** for the Argentina vs Saudi Arabia match. The system calculates the exact position of each player's tracked offside point — the furthest legal body part — relative to the second-to-last defender at the precise frame the ball leaves the passer's foot.

For Messi's disallowed goal at minute 21: the tracked offside point was **3 centimetres past the line**. That number is not an estimate. It is derived directly from the StatsBomb event data (match_id 3857300).

This level of precision is only publicly available for select matches in the StatsBomb Open Data release — which is why the Argentina vs Saudi Arabia match is used as the demo simulation.

### ML Danger Scoring

Each offside event is scored for danger using a Random Forest Classifier trained on:
- Distance from goal at moment of pass
- Pitch zone (Central / Half-space / Wide)
- Offside margin in centimetres
- Match phase and game minute
- Pass angle and length

SHAP values are computed per event so fans and analysts can see exactly which factors drove the AI score — not just what the model decided, but why.

---

## Why This Solution Matters

### For the Casual Fan
VAR decisions affect match results. Fans deserve to understand what happened and why — not just that the flag went up. E-VAR makes every decision readable for any supporter, regardless of football knowledge or language.

### For Journalists and Analysts
Real-time access to danger scores, offside margins, law references, and SHAP breakdowns — all in one place, available seconds after the decision.

### For the Game
Transparency builds trust. When fans understand a decision — even one that goes against their team — they are more likely to accept it. The communication gap after VAR is not a technology problem. It is an explanation problem. E-VAR fixes that.

### In the Context of the FIFA World Cup
The World Cup is the most-watched sporting event on the planet. The 2022 tournament in Qatar saw record VAR usage — 560+ decisions across 64 matches. Billions of viewers across every continent watched calls they could not decode. E-VAR Companion is built specifically for this global, multilingual audience.

### Scalability
The architecture is designed to scale from a single demo match to a full live tournament:
- Any match feed can be connected via the detection layer
- Any language can be added to the i18n layer
- The Granite API call is stateless and horizontally scalable
- The frontend is a single HTML file — zero infrastructure for the viewer
- Partnership with FIFA, StatsBomb, or a broadcaster unlocks full event-level data for every match

---

## Current Limitations

We want to be transparent about the boundaries of the current demo.

**Player-level data:** Granular positional tracking — exact offside margins in centimetres, precise body-part coordinates per player — was not publicly released by FIFA for all 64 matches. StatsBomb Open Data provides this level of detail for the Argentina vs Saudi Arabia match only. For all other matches, E-VAR uses aggregate WC2022 tournament data. The AI framework and explanations work for every match; the precision of the numbers improves with a richer data feed.

**Live video:** The auto-detection demo runs on a local pre-recorded clip (`clip.mp4`). Integration with a live broadcast feed requires a rights agreement with a broadcaster or FIFA.

**Next step:** A data partnership with FIFA, StatsBomb, or a broadcast rights holder would unlock event-level tracking for every match, making E-VAR Companion fully production-ready for live tournaments.

---

## Running the Detection Demo

The Detection page autopilot requires a local copy of the match clip, which is **not included in this repository for copyright reasons** — the footage is licensed to FIFA and broadcasters.

To enable the Detection page locally:
1. Obtain a licensed copy of the Argentina vs Saudi Arabia FIFA World Cup 2022 match broadcast
2. Rename the file to `clip.mp4`
3. Place it in the root folder of the project
4. Run the app — the autopilot triggers automatically at minute 21 and minute 28

> The full Detection experience is demonstrated in the [demo video](YOUR_YOUTUBE_LINK_HERE).
> All other features — Fan Chat, History, Teams, Laws DB, Settings — work fully without the clip.

---

## Running Locally

```bash
# Clone the repository
git clone https://github.com/Aruki417/evar-companion.git
cd evar-companion

# Install backend dependencies
pip install -r requirements.txt

# Set your watsonx.ai credentials
export WATSONX_API_KEY=your_key_here
export WATSONX_SPACE_ID=your_space_id_here
export WATSONX_MODEL_ID=ibm/granite-4-h-small

# Start the FastAPI backend (invisible engine — runs on port 8000)
uvicorn evar_backend:app --port 8000

# In a second terminal — serve the frontend on port 3000
npx serve . --port 3000
```

Then open `http://localhost:3000` in your browser.

> **Note:** Port 8000 is the backend engine — it runs silently in the terminal.
> Port 3000 is the webpage you interact with and record.
> You never need to open localhost:8000 in your browser.

---

## Live Demo

**Streamlit app:** [evar-companion.streamlit.app](https://evar-companion.streamlit.app)
**Demo video:** [Watch Demo](https://drive.google.com/file/d/1_vV9ufHR4JpMKFfRmijwwhhY7jeZe0En/view?usp=sharing)

---

## Data Sources

- **StatsBomb Open Data** — event-level tracking for WC2022, match_id 3857300 (Argentina vs Saudi Arabia)
- **FIFA World Cup 2022 Official Statistics** — 64 matches · 172 goals · 214 yellow cards · 4 red cards · 224 VAR offside decisions

---

## Team

Built by Jessica Lan

---

## License

MIT License — see LICENSE for details.

---

*E-VAR Companion · Powered by IBM Granite via watsonx.ai · Built for the world's game*
