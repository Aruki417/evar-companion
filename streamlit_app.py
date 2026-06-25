"""
E-VAR Companion — Streamlit wrapper
Embeds the full HTML app so judges can access it via a public URL.
Run: streamlit run streamlit_app.py
"""
import streamlit as st
import subprocess
import threading
import os
import sys
import time

st.set_page_config(
    page_title="E-VAR Companion",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Hide Streamlit chrome for a cleaner look
st.markdown("""
<style>
    #MainMenu, header, footer {visibility: hidden;}
    .block-container {padding: 0 !important; margin: 0 !important; max-width: 100% !important;}
    iframe {border: none !important;}
</style>
""", unsafe_allow_html=True)

# Read the HTML app
html_path = os.path.join(os.path.dirname(__file__), "index.html")
with open(html_path, "r", encoding="utf-8") as f:
    html_content = f.read()

# Embed the full app — height covers a full screen
st.components.v1.html(html_content, height=900, scrolling=True)

# Sidebar — info for judges
with st.sidebar:
    st.title("⚽ E-VAR Companion")
    st.markdown("**IBM SkillsBuild AI Builders Challenge 2026**")
    st.divider()
    st.markdown("""
    ### What this is
    AI-powered VAR offside explainer for football fans.
    
    ### How it works
    1. Go to **Detection** page
    2. Press Play on the video
    3. Watch it auto-pause at the VAR moment
    4. Read the IBM Granite explanation
    5. Switch languages with the pills
    6. Try **Fan Chat** to ask questions
    
    ### Tech stack
    - IBM Granite on watsonx.ai
    - StatsBomb Open Data (WC2022)
    - RandomForest + SHAP (96% accuracy)
    - FastAPI backend
    - 23 languages
    
    ### Data
    224 real offside events  
    FIFA World Cup 2022  
    Match: ARG vs KSA  
    """)
    st.divider()
    st.caption("Built by Jessica · MSc Business Analytics")
