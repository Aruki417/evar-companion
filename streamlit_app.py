import streamlit as st
import os
import subprocess
import threading
import time

st.set_page_config(
    page_title="E-VAR Companion",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Start the FastAPI backend automatically if not already running
def start_backend():
    try:
        import httpx
        httpx.get("http://localhost:8000/health", timeout=2)
    except Exception:
        # Backend not running — start it
        env = os.environ.copy()
        # Pass Streamlit secrets to the backend process as env vars
        try:
            for k in ["WATSONX_API_KEY","WATSONX_SPACE_ID","WATSONX_MODEL_ID","WATSONX_URL"]:
                v = st.secrets.get(k,"")
                if v: env[k] = v
        except Exception:
            pass
        subprocess.Popen(
            ["uvicorn", "evar_backend:app", "--port", "8000"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        time.sleep(3)  # wait for startup

threading.Thread(target=start_backend, daemon=True).start()

# Hide Streamlit chrome
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
    iframe {border: none !important;}
</style>
""", unsafe_allow_html=True)

# Load and embed the HTML app
html_path = os.path.join(os.path.dirname(__file__), "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

st.components.v1.html(html_content, height=900, scrolling=True)

with st.sidebar:
    st.title("⚽ E-VAR Companion")
    st.markdown("**IBM SkillsBuild AI Builders Challenge 2026**")
    st.divider()
    st.markdown("""
### How to use
1. Go to **Detection** page
2. Press Play on the video
3. Watch it auto-pause at VAR moment
4. Read the IBM Granite explanation
5. Switch languages with the pills
6. Try **Fan Chat** to ask questions

### Tech stack
- IBM Granite on watsonx.ai
- StatsBomb Open Data (WC2022)
- RandomForest + SHAP (96% accuracy)
- FastAPI backend · 23 languages
    """)
    st.divider()
    st.caption("Built by Jessica · MSc Business Analytics")
