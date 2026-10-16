"""Minimal Streamlit demo UI for the assistant. (Milestone M5)

Talks to the FastAPI backend: upload docs, ask questions, inspect citations.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")
st.info("Milestone M5: wire this UI to the FastAPI backend (/ingest, /ask).")
