import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score,
)
import yfinance as yf

# ==========================================================
# 1. PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="Telkom Docs - Telkom Intelligence Platform",
    page_icon="📶",
    layout="wide",
    initial_sidebar_state="collapsed",
)

LOGO_URL = "https://upload.wikimedia.org/wikipedia/id/c/c4/Telkom_Indonesia_2013.svg"
RED, DARK_RED, GREY = "#EE2E24", "#B71C1C", "#6b7280"
GREEN, AMBER, BLUE, PURPLE = "#28a745", "#FF9800", "#2196F3", "#673AB7"
YEAR = pd.Timestamp.today().year

# ==========================================================
# 2. GLOBAL CSS  (warna & kartu = Dashboard 1, layout = Flutter Docs)
# ==========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, .stApp, .stMarkdown, p, h1, h2, h3, h4, h5, h6, label, input, textarea, button, li, td, th {
    font-family: 'Plus Jakarta Sans', sans-serif;
}
span[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

.stApp { background: #fafbfc; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
.main .block-container { padding-top: 1rem; padding-left: 2.2rem; padding-right: 2.2rem; max-width: 1320px; }

/* ---------- HEADER / NAVBAR ---------- */
.tk-header { background: #fff; border: 1px solid #f0f0f0; border-radius: 22px; overflow: hidden;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05); margin-bottom: 14px; }
.tk-topline { height: 5px; background: linear-gradient(90deg, #EE2E24, #B71C1C, #FF6F61, #EE2E24); background-size: 300% 100%;
    animation: slide 8s linear infinite; }
@keyframes slide { 0% {background-position: 0% 0;} 100% {background-position: 300% 0;} }
.tk-nav { display: flex; align-items: center; justify-content: space-between; padding: 14px 26px; gap: 20px; flex-wrap: wrap; }
.tk-nav-left { display: flex; align-items: center; gap: 28px; flex-wrap: wrap; }
.tk-brand { display: flex; align-items: center; gap: 12px; font-size: 22px; font-weight: 800; color: #202124; }
.tk-badge { background: #fdecea; color: #EE2E24; font-size: 13px; font-weight: 700; padding: 3px 12px; border-radius: 8px; }
.tk-links { display: flex; gap: 22px; font-size: 14px; font-weight: 600; color: #475569; }
.tk-links span { position: relative; cursor: pointer; padding: 4px 0; transition: color .2s; }
.tk-links span:hover { color: #EE2E24; }
.tk-links span::after { content: ''; position: absolute; left: 0; bottom: -2px; width: 0; height: 2px; background: #EE2E24; transition: width .25s; }
.tk-links span:hover::after { width: 100%; }
.tk-links span.active { color: #EE2E24; }
.tk-links span.active::after { width: 100%; }
.tk-nav-right { display: flex; align-items: center; gap: 14px; }
.tk-search { background: #f4f5f7; border-radius: 999px; padding: 8px 18px; font-size: 13px; color: #64748b; min-width: 170px; }
.tk-live { display: flex; align-items: center; gap: 8px; font-size: 12px; font-weight: 700; color: #28a745;
    background: #e9f7ee; padding: 7px 14px; border-radius: 999px; }
.tk-dot { width: 8px; height: 8px; background: #28a745; border-radius: 50%; animation: pulse 1.6s infinite; }
@keyframes pulse { 0% {box-shadow: 0 0 0 0 rgba(40,167,69,.6);} 70% {box-shadow: 0 0 0 9px rgba(40,167,69,0);} 100% {box-shadow: 0 0 0 0 rgba(40,167,69,0);} }
.tk-cta { background: linear-gradient(135deg, #EE2E24, #B71C1C); color: #fff !important; padding: 10px 22px; border-radius: 999px;
    font-weight: 700; font-size: 14px; box-shadow: 0 8px 18px rgba(238,46,36,.3); }

/* ---------- TICKER ---------- */
.tk-ticker { background: #1a1a1a; border-radius: 14px; overflow: hidden; margin-bottom: 14px; white-space: nowrap; }
.tk-track { display: inline-flex; gap: 48px; padding: 10px 0; animation: marquee 45s linear infinite; color: #ddd; font-size: 13px; font-weight: 500; }
.tk-track b { color: #ff8a80; }
@keyframes marquee { 0% {transform: translateX(0);} 100% {transform: translateX(-50%);} }

/* ---------- ANNOUNCEMENT BANNER ---------- */
.tk-announce { background: linear-gradient(90deg, #EE2E24 0%, #B71C1C 55%, #7f1d1d 100%); color: #fff; padding: 14px 24px;
    border-radius: 16px; font-size: 14px; font-weight: 500; margin-bottom: 26px; display: flex; justify-content: space-between;
    align-items: center; gap: 12px; flex-wrap: wrap; box-shadow: 0 10px 24px rgba(238,46,36,.25); }
.tk-announce a { color: #fff; font-weight: 800; text-decoration: underline; }

/* ---------- HERO ---------- */
.tk-hero-main { background: linear-gradient(135deg, #EE2E24 0%, #B71C1C 100%); padding: 54px 50px; border-radius: 30px; color: #fff;
    margin-bottom: 32px; box-shadow: 0 20px 40px rgba(238,46,36,.3); position: relative; overflow: hidden; }
.tk-hero-main::after { content: ''; position: absolute; inset: 0; background: url('https://www.transparenttextures.com/patterns/cubes.png'); opacity: .1; }
.tk-hero-main > * { position: relative; z-index: 2; }
.tk-hero-main h1 { font-size: 46px; font-weight: 800; margin: 0 0 8px 0; color: #fff; line-height: 1.15; }
.tk-hero-main p { font-size: 18px; opacity: .92; margin: 0 0 22px 0; color: #fff; max-width: 780px; }
.tk-pill { display: inline-block; background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.35); padding: 6px 16px;
    border-radius: 999px; font-size: 13px; font-weight: 600; margin-right: 8px; margin-bottom: 6px; color: #fff; }
.tk-hero { background: #fff; border: 1px solid #f0f0f0; border-left: 8px solid #EE2E24; border-radius: 22px; padding: 26px 32px;
    margin-bottom: 26px; box-shadow: 0 10px 20px rgba(0,0,0,.03); }
.tk-hero h2 { margin: 0 0 6px 0; font-size: 28px; font-weight: 800; color: #0f172a; }
.tk-hero p { margin: 0; color: #64748b; font-size: 15px; }

/* ---------- CARDS ---------- */
.metric-container { background: #fff; padding: 26px 22px; border-radius: 20px; border: 1px solid #f0f0f0; box-shadow: 0 10px 20px rgba(0,0,0,.03);
    transition: transform .3s ease; text-align: center; height: 100%; border-bottom: 4px solid transparent; }
.metric-container:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(0,0,0,.08); border-bottom: 4px solid #EE2E24; }
.metric-container h4 { margin: 0 0 6px 0; font-size: 15px; color: #475569; font-weight: 700; }
.metric-container h2 { margin: 6px 0; font-weight: 800; font-size: 30px; }
.metric-container p { margin: 0; font-size: 13px; color: #94a3b8; }

.doc-card { background: #fff6f5; border: 1px solid #fbd0cd; border-radius: 20px; padding: 24px; min-height: 150px;
    transition: all .3s ease; box-shadow: 0 6px 16px rgba(238,46,36,.04); }
.doc-card:hover { transform: translateY(-5px); box-shadow: 0 15px 30px rgba(238,46,36,.12); border-color: #EE2E24; }
.doc-card .ic { font-size: 26px; margin-bottom: 6px; }
.doc-card h3 { color: #EE2E24; font-size: 20px; font-weight: 700; margin: 0 0 8px 0; }
.doc-card p { color: #475569; font-size: 14px; line-height: 1.55; margin: 0; }

.info-card { background: #fff; padding: 28px; border-radius: 20px; border: 1px solid #eee; box-shadow: 0 10px 20px rgba(0,0,0,.02); height: 100%; }
.info-card h3, .info-card h4 { color: #EE2E24; margin-top: 0; }
.vision-card { background: #f8f9fa; padding: 28px; border-radius: 20px; border-left: 5px solid #EE2E24; height: 100%; }
.vision-card h4 { color: #EE2E24; margin-top: 0; }

.step-card { background: #fff; border: 1px solid #f0f0f0; border-radius: 20px; padding: 22px; box-shadow: 0 10px 20px rgba(0,0,0,.03); height: 100%; }
.step-no { display: inline-flex; width: 38px; height: 38px; align-items: center; justify-content: center; border-radius: 12px;
    background: linear-gradient(135deg, #EE2E24, #B71C1C); color: #fff; font-weight: 800; margin-bottom: 10px; }
.step-card h4 { margin: 0 0 6px 0; color: #0f172a; }
.step-card p { margin: 0; font-size: 14px; color: #64748b; }

.status-baik, .status-waspada, .status-buruk { color: #fff; padding: 25px; border-radius: 20px; font-weight: 800; text-align: center; font-size: 24px; height: 100%; }
.status-baik { background: linear-gradient(135deg, #28a745 0%, #20c997 100%); box-shadow: 0 10px 20px rgba(40,167,69,.3); }
.status-waspada { background: linear-gradient(135deg, #FFC107 0%, #FF9800 100%); box-shadow: 0 10px 20px rgba(255,193,7,.3); }
.status-buruk { background: linear-gradient(135deg, #DC3545 0%, #C62828 100%); box-shadow: 0 10px 20px rgba(220,53,69,.3); }
.status-baik small, .status-waspada small, .status-buruk small { display: block; font-size: 13px; font-weight: 600; opacity: .9; margin-bottom: 6px; }

.bar-wrap { background: #fff; border: 1px solid #f0f0f0; border-radius: 20px; padding: 22px 26px; box-shadow: 0 10px 20px rgba(0,0,0,.03); }
.bar-row { margin: 10px 0; font-size: 14px; font-weight: 600; color: #334155; }
.bar-bg { background: #f1f5f9; border-radius: 999px; height: 14px; margin-top: 6px; overflow: hidden; }
.bar-fill { height: 14px; border-radius: 999px; }

.section-title { font-size: 24px; font-weight: 800; color: #0f172a; margin: 30px 0 14px 0; }
.section-title span { color: #EE2E24; }

div[data-testid="stMetric"] { background: #fff; border: 1px solid #f0f0f0; padding: 20px; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,.03); }

/* ---------- BUTTONS ---------- */
.stButton > button, .stDownloadButton > button { border-radius: 999px; font-weight: 700; border: 1px solid #fbd0cd; color: #EE2E24; background: #fff; transition: all .2s; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color: #EE2E24; color: #fff; background: #EE2E24; }
.stButton > button[kind="primary"], .stButton > button[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #EE2E24, #B71C1C); color: #fff; border: none; padding: 0.55rem 1.6rem; box-shadow: 0 8px 18px rgba(238,46,36,.3); }
.stButton > button[kind="primary"]:hover { filter: brightness(1.08); }

/* ---------- TABS ---------- */
button[data-baseweb="tab"] { font-weight: 700; }
button[data-baseweb="tab"][aria-selected="true"] { color: #EE2E24; }
div[data-baseweb="tab-highlight"] { background-color: #EE2E24; }

/* ---------- LAYOUT: tanpa sidebar ---------- */
#MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"] { display: none !important; }
.block-container, [data-testid="stMainBlockContainer"] { padding: 1rem 2.2rem 0 2.2rem !important; max-width: 1320px !important; }
.stButton > button p { color: inherit; }

/* ---------- NAVBAR ATAS (menu = tombol asli, tampilan = header sebelumnya) ---------- */
.st-key-topnav, div.stElementContainer:has(> .st-key-topnav) { position: sticky; top: 8px; z-index: 1000; }
.st-key-topnav { position: sticky; overflow: hidden; background: rgba(255,255,255,.95); backdrop-filter: blur(12px); border: 1px solid #f0f0f0;
    border-radius: 22px; box-shadow: 0 10px 30px rgba(0,0,0,.06); padding: 18px 24px 14px 24px; }
.st-key-topnav::before { content: ''; position: absolute; left: 0; right: 0; top: 0; height: 5px;
    background: linear-gradient(90deg, #EE2E24, #B71C1C, #FF6F61, #EE2E24); background-size: 300% 100%; animation: slide 8s linear infinite; }
.st-key-topnav [data-testid="stHorizontalBlock"] { gap: 4px !important; align-items: center; flex-wrap: nowrap; }
.st-key-topnav [data-testid="stColumn"], .st-key-topnav [data-testid="column"] { min-width: 0 !important; }
.st-key-topnav [data-testid="stMarkdownContainer"] p { margin: 0; }
.st-key-topnav .stButton > button { width: 100%; border: none; background: transparent; color: #475569; font-weight: 600; font-size: 13.5px;
    padding: .35rem .2rem; min-height: 0; border-radius: 0; box-shadow: none; white-space: nowrap; filter: none; }
.st-key-topnav .stButton > button:hover { background: transparent; color: #EE2E24; box-shadow: inset 0 -2px 0 #EE2E24; }
.st-key-topnav .stButton > button[kind="primary"], .st-key-topnav .stButton > button[data-testid="stBaseButton-primary"] {
    background: transparent; color: #EE2E24; font-weight: 800; box-shadow: inset 0 -2px 0 #EE2E24; padding: .35rem .2rem; filter: none; }
.st-key-nav_cta button { background: linear-gradient(135deg, #EE2E24, #B71C1C) !important; color: #fff !important; border-radius: 999px !important;
    padding: .5rem 1rem !important; box-shadow: 0 8px 18px rgba(238,46,36,.3) !important; font-weight: 700 !important; }
.st-key-nav_cta button:hover { filter: brightness(1.08); }
.tk-live { width: fit-content; margin-left: auto; white-space: nowrap; }

/* ---------- FOOTER ---------- */
.st-key-tkfoot { position: relative; overflow: hidden; margin-top: 56px; border-radius: 30px 30px 0 0; padding: 0 48px;
    background: linear-gradient(160deg, #3a0b0b, #160606 60%, #7f1d1d); box-shadow: 0 -10px 40px rgba(0,0,0,.12); color: #d9b8b3; }
.st-key-tkfoot::before, .st-key-tkfoot::after { content: ''; position: absolute; border-radius: 50%; filter: blur(60px); opacity: .35; pointer-events: none; animation: fl 12s ease-in-out infinite; }
.st-key-tkfoot::before { width: 300px; height: 300px; background: #EE2E24; left: -80px; top: 80px; }
.st-key-tkfoot::after { width: 260px; height: 260px; background: #FF9800; right: -60px; bottom: 40px; animation-delay: -5s; }
.st-key-tkfoot > * { position: relative; z-index: 1; }
@keyframes fl { 50% { transform: translate(40px, -30px) scale(1.15); } }
.tk-footer-line { height: 6px; margin: 0 -48px; background: linear-gradient(90deg, #EE2E24, #B71C1C, #FF6F61, #FF9800, #EE2E24); background-size: 300% 100%; animation: slide 8s linear infinite; }
.tk-fcta { padding-top: 34px; }
.tk-fcta h3 { margin: 0; padding: 0; color: #fff !important; font-size: 26px; font-weight: 800; }
.tk-fcta h3 span { color: #ff6f61; }
.tk-fcta p { margin: 4px 0 0 0; color: #d9b8b3; font-size: 14px; }
.st-key-foot_b1 button, .st-key-foot_b2 button { width: 100%; font-weight: 800; border-radius: 999px; padding: .6rem 1rem; }
.st-key-foot_b1 button { background: linear-gradient(135deg, #EE2E24, #B71C1C) !important; color: #fff !important; border: none !important; box-shadow: 0 10px 22px rgba(238,46,36,.4); }
.st-key-foot_b2 button { background: transparent !important; color: #fff !important; border: 2px solid rgba(255,255,255,.4) !important; }
.st-key-foot_b2 button:hover { background: rgba(255,255,255,.12) !important; }
.ftop { text-align: center; margin: 30px 0 30px; }
.fbig { font-size: clamp(44px, 8vw, 92px); font-weight: 800; letter-spacing: 2px; line-height: 1.1;
    background: linear-gradient(90deg, #fff, #ffb4a8, #ffb74d, #fff); background-size: 250% 100%; -webkit-background-clip: text; background-clip: text; color: transparent; animation: slide 7s linear infinite; }
.fsub { margin: 6px 0 22px 0; color: #d9b8b3; }
.fst { display: flex; justify-content: center; gap: 14px; flex-wrap: wrap; }
.fst div { background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.15); border-radius: 18px; padding: 12px 26px; text-align: center; transition: .3s; }
.fst div:hover { transform: translateY(-5px); }
.fst b { display: block; color: #fff; font-size: 24px; }
.fst span { font-size: 12px; }
.st-key-footlinks [data-testid="stHorizontalBlock"] { border-top: 1px solid rgba(255,255,255,.12); padding-top: 30px; gap: 30px; }
.st-key-tkfoot h4 { color: #fff; font-size: 15px; margin: 0 0 10px 0; padding: 0; }
.st-key-tkfoot p { color: #d9b8b3; font-size: 14px; margin: 0 0 8px 0; }
.fl-logo { background: #fff; display: inline-block; padding: 10px 16px; border-radius: 14px; margin-bottom: 14px; }
.fl-logo img { height: 32px; display: block; }
.tags span { display: inline-block; font-size: 11px; font-weight: 700; padding: 4px 12px; border-radius: 99px; border: 1px solid rgba(255,255,255,.25); margin: 4px 6px 0 0; color: #f3d6d1; }
.st-key-tkfoot .stButton > button { background: none; border: none; color: #d9b8b3; justify-content: flex-start; padding: 2px 0; min-height: 0;
    font-weight: 500; font-size: 14px; box-shadow: none; border-radius: 0; margin-bottom: 4px; }
.st-key-tkfoot .stButton > button:hover { background: none; color: #fff; transform: translateX(4px); }
.fb { display: flex; justify-content: space-between; align-items: center; gap: 10px; flex-wrap: wrap; border-top: 1px solid rgba(255,255,255,.12);
    margin-top: 26px; padding: 18px 0 26px; font-size: 12.5px; color: #c9a7a2; }
.fb b { color: #ff8a80; }

@media (max-width: 1150px) {
    .st-key-topnav [data-testid="stHorizontalBlock"] { flex-wrap: wrap; }
    .st-key-topnav [data-testid="stColumn"], .st-key-topnav [data-testid="column"] { flex: 1 1 22% !important; }
    .tk-live { margin-left: 0; }
}
@media (max-width: 900px) {
    .tk-hero-main { padding: 32px 24px; } .tk-hero-main h1 { font-size: 30px; }
    .st-key-tkfoot { padding: 0 24px; } .tk-footer-line { margin: 0 -24px; }
}
@media (prefers-reduced-motion: reduce) { * { animation: none !important; transition: none !important; } }

/* ================= V3: logo menonjol, header, home, footer ================= */
.tk-brand { display: flex; align-items: center; height: 100%; }

.tk-logo { 
    display: inline-flex; 
    align-items: center; 
    justify-content: center; 
    background: #fff; 
    border-radius: 14px; 
    padding: 2px 8px; 
    transform: translateY(-10px) !important; /* <--- GESER NAIK/TURUN (Ganti angka ini untuk pasnya) */
}

.tk-brand .tk-logo img { 
    height: 50px !important; /* <--- KECILKAN/BESARKAN LOGO DI SINI */
    display: block; 
}

.st-key-topnav { background: #ffffff !important; box-shadow: 0 4px 20px rgba(0,0,0,.05); padding-top: 14px; padding-bottom: 12px; }   
.st-key-topnav::before { height: 6px; }
.tk-live { font-size: 11.5px; padding: 6px 10px; }

/* hero home */
.st-key-hero { position: relative; overflow: hidden; border-radius: 30px; padding: 46px 48px; color: #fff; margin-bottom: 34px;
    background: radial-gradient(circle at 12% 15%, rgba(255,111,97,.5), transparent 45%), radial-gradient(circle at 90% 90%, rgba(255,152,0,.32), transparent 40%),
                linear-gradient(135deg, #EE2E24 0%, #B71C1C 60%, #6d1414 100%);
    box-shadow: 0 24px 48px rgba(238,46,36,.32); }
.st-key-hero::before { content: ''; position: absolute; inset: 0; background-image: radial-gradient(rgba(255,255,255,.14) 1px, transparent 1px); background-size: 22px 22px; opacity: .5; pointer-events: none; }
.st-key-hero::after { content: ''; position: absolute; width: 380px; height: 380px; border-radius: 50%; background: rgba(255,255,255,.07); right: -110px; top: -150px; animation: fl 12s ease-in-out infinite; pointer-events: none; }
.st-key-hero > * { position: relative; z-index: 1; }
.st-key-hero p { color: #fff; }
.hero-logo { display: inline-block; background: #fff; padding: 12px 26px; border-radius: 18px; margin-bottom: 22px;
    box-shadow: 0 14px 34px rgba(0,0,0,.3), 0 0 0 6px rgba(255,255,255,.18); }
.hero-logo img { height: 70px; display: block; }
.st-key-hero h1.hero-h1 { font-size: clamp(32px, 4.2vw, 50px); font-weight: 800; line-height: 1.12; color: #fff; margin: 0 0 12px 0; padding: 0; }
.st-key-hero p.hero-p { font-size: 17px; opacity: .94; max-width: 560px; margin: 0 0 16px 0; }
.st-key-hero .stButton > button { width: 100%; padding: .65rem 1rem; font-weight: 800; border-radius: 999px; }
.st-key-hero_b1 button { background: #fff !important; color: #EE2E24 !important; border: none !important; box-shadow: 0 12px 26px rgba(0,0,0,.25); }
.st-key-hero_b1 button:hover { transform: translateY(-3px); }
.st-key-hero_b2 button { background: transparent !important; color: #fff !important; border: 2px solid rgba(255,255,255,.7) !important; }
.st-key-hero_b2 button:hover { background: rgba(255,255,255,.16) !important; }
.glass { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.35); backdrop-filter: blur(10px); border-radius: 24px; padding: 24px 26px; box-shadow: 0 18px 40px rgba(0,0,0,.2); }
.glass .gh { display: flex; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 700; color: #fff; }
.glass .gd { margin-left: auto; opacity: .8; font-weight: 600; }
.glass .gp { font-size: 40px; font-weight: 800; color: #fff; margin: 10px 0 4px; line-height: 1.1; }
.glass .gc { display: inline-block; font-weight: 800; padding: 4px 12px; border-radius: 999px; background: rgba(255,255,255,.2); color: #fff; font-size: 14px; }
.glass .gc.up { background: #e9f7ee; color: #15803d; }
.glass .gc.dn { background: #fdecea; color: #b91c1c; }
.glass .gc small { display: inline; font-weight: 600; opacity: .8; margin-left: 6px; color: inherit; }
.glass .gg { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 18px; border-top: 1px solid rgba(255,255,255,.25); padding-top: 14px; }
.glass small { display: block; font-size: 11px; opacity: .8; color: #fff; }
.glass .gg b { font-size: 13.5px; color: #fff; }

/* kartu eksplor home */
.ti { width: 46px; height: 46px; border-radius: 14px; background: linear-gradient(135deg, #fdecea, #fbd0cd); display: grid; place-items: center; font-size: 22px; margin-bottom: 10px; }
div[class*="st-key-tile"] { background: #fff; border: 1px solid #f0f0f0; border-radius: 20px; padding: 22px; box-shadow: 0 10px 20px rgba(0,0,0,.03); transition: .3s; border-bottom: 4px solid transparent; }
div[class*="st-key-tile"]:hover { transform: translateY(-5px); border-bottom-color: #EE2E24; box-shadow: 0 15px 30px rgba(238,46,36,.12); }
div[class*="st-key-tile"] h4 { margin: 0 0 6px 0; padding: 0; color: #0f172a; font-size: 17px; font-weight: 800; }
div[class*="st-key-tile"] p { margin: 0; font-size: 13.5px; color: #64748b; line-height: 1.55; min-height: 62px; }

/* footer */
.st-key-tkfoot { background-image: radial-gradient(rgba(255,255,255,.06) 1px, transparent 1px), linear-gradient(160deg, #3a0b0b, #160606 60%, #7f1d1d); background-size: 22px 22px, auto; }
.ftop .fl-plate { display: inline-block; background: #fff; padding: 14px 32px; border-radius: 20px; margin-bottom: 22px;
    box-shadow: 0 0 0 8px rgba(255,255,255,.08), 0 20px 50px rgba(238,46,36,.45); }
.ftop .fl-plate img { height: 54px; display: block; }
.fst b { background: linear-gradient(90deg, #fff, #ffb4a8); -webkit-background-clip: text; background-clip: text; color: transparent; }
.fok { display: inline-flex; align-items: center; gap: 8px; background: rgba(40,167,69,.15); color: #7ee2a0; border: 1px solid rgba(126,226,160,.35); padding: 4px 12px; border-radius: 999px; font-weight: 700; }
.fok .tk-dot { background: #7ee2a0; }

/* ================= V4: header & footer full-width (edge to edge) ================= */
:root { --px: max(35px, calc((100% - 1320px) / 2 + 35px)); }
.block-container, [data-testid="stMainBlockContainer"] { padding: 0 !important; max-width: 100% !important; }
.block-container > div[data-testid="stVerticalBlock"], [data-testid="stMainBlockContainer"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
.st-key-body { padding: 28px var(--px) 0 var(--px); }

.st-key-topnav, div.stElementContainer:has(> .st-key-topnav) { top: 0; }
.st-key-topnav { border-radius: 0; border: none; border-bottom: 1px solid #f0f0f0; padding: 20px var(--px) 14px var(--px); }
.tk-ticker { border-radius: 0; margin: 0; }
.tk-announce { border-radius: 0; margin: 0; box-shadow: none; padding: 14px var(--px); }

.st-key-tkfoot { border-radius: 0; padding: 0; margin-top: 56px; }
.st-key-tkfoot > * { box-sizing: border-box; }
.st-key-tkfoot > *:not(:first-child) { padding-left: var(--px); padding-right: var(--px); }
.tk-footer-line { margin: 0; }

/* ================= V5: tabel kustom ================= */
.tk-tbl-wrap { background: #fff; border: 1px solid #f0f0f0; border-radius: 20px; box-shadow: 0 10px 20px rgba(0,0,0,.04); overflow: hidden; overflow-x: auto; margin-bottom: 8px; }
table.tk-tbl { width: 100%; border-collapse: collapse; border: none; margin: 0; font-size: 14px; }
table.tk-tbl th { background: linear-gradient(135deg, #EE2E24, #B71C1C); color: #fff; text-align: left; padding: 13px 18px; font-weight: 700;
    font-size: 12.5px; letter-spacing: .4px; text-transform: uppercase; border: none; white-space: nowrap; }
table.tk-tbl th.num, table.tk-tbl td.num { text-align: right; }
table.tk-tbl td { padding: 12px 18px; border: none; border-bottom: 1px solid #f3f4f6; color: #334155; background: #fff; font-variant-numeric: tabular-nums; }
table.tk-tbl tr:nth-child(even) td { background: #fff8f7; }
table.tk-tbl tr:hover td { background: #fdecea; }
table.tk-tbl tr:last-child td { border-bottom: none; }
table.tk-tbl td.num { font-weight: 600; }
table.tk-tbl td.mut { color: #94a3b8; }
.tk-cell { display: flex; align-items: center; justify-content: flex-end; gap: 10px; }
.tk-hb { flex: 0 0 70px; height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden; }
.tk-hbf { height: 8px; border-radius: 999px; background: linear-gradient(90deg, #ff8a80, #EE2E24); }
.tk-pl { display: inline-block; font-size: 12px; font-weight: 800; padding: 4px 14px; border-radius: 999px; letter-spacing: .3px; }
.tk-pl.low, .tk-pl.baik { background: #e9f7ee; color: #15803d; }
.tk-pl.medium, .tk-pl.waspada { background: #fff4e0; color: #b45309; }
.tk-pl.high, .tk-pl.buruk { background: #fdecea; color: #b91c1c; }

/* ================= V6: tombol Previous / Next ================= */
.st-key-pagenav { margin-top: 40px; padding-top: 22px; border-top: 1px solid #f0f0f0; }
.st-key-pagenav [data-testid="stHorizontalBlock"] { align-items: center; }
.st-key-pagenav [data-testid="stColumn"]:last-child .stButton { display: flex; justify-content: flex-end; }
.st-key-pagenav .stButton > button { padding: .55rem 1.6rem; white-space: nowrap; }
</style>
""", unsafe_allow_html=True)


# ==========================================================
# 3. HELPERS
# ==========================================================
def html(s: str):
    """Render HTML tanpa indentasi supaya tidak dianggap code-block oleh markdown."""
    st.markdown("\n".join(l.strip() for l in s.strip().splitlines() if l.strip()), unsafe_allow_html=True)


def go(page_name: str):
    st.session_state.page = page_name


def rp(x):
    return "Rp " + f"{x:,.0f}".replace(",", ".")


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#e5e7eb")
    ax.tick_params(colors=GREY, labelsize=9)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.yaxis.label.set_color(GREY)
    ax.xaxis.label.set_color(GREY)


def new_fig(w=7.5, h=3.3):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("white")
    style_ax(ax)
    return fig, ax


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


def metric_card(title, value, sub="", color=RED):
    return (f'<div class="metric-container"><h4>{title}</h4>'
            f'<h2 style="color:{color};">{value}</h2><p>{sub}</p></div>')


def risk_card(label, title="🚨 Risk Level Prediction"):
    css = "status-baik" if label == "LOW" else "status-waspada" if label == "MEDIUM" else "status-buruk"
    return f'<div class="{css}"><small>{title}</small>{label}</div>'


def doc_card(icon, title, desc):
    return f'<div class="doc-card"><div class="ic">{icon}</div><h3>{title}</h3><p>{desc}</p></div>'


def page_hero(title, desc):
    html(f'<div class="tk-hero"><h2>{title}</h2><p>{desc}</p></div>')


def section(title_plain, title_red=""):
    html(f'<div class="section-title">{title_plain} <span>{title_red}</span></div>')


def pill(label):
    key = str(label).lower()
    if key in ("low", "medium", "high", "baik", "waspada", "buruk"):
        return f'<span class="tk-pl {key}">{label}</span>'
    return f"<b>{label}</b>"


def html_table(df, fmt=None, bar_cols=(), index_label=""):
    """Tabel HTML bertema Telkom (header merah, zebra, hover, pill label, bar mini)."""
    fmt = fmt or {}
    head = f'<th>{index_label}</th>' + "".join(f'<th class="num">{c}</th>' for c in df.columns)
    rows = []
    for idx, r in df.iterrows():
        tds = [f"<td>{pill(idx)}</td>"]
        for c in df.columns:
            v = r[c]
            if pd.isna(v):
                tds.append('<td class="num mut">–</td>')
                continue
            txt = fmt.get(c, "{:,.3f}").format(v)
            if c in bar_cols:
                mx = float(df[c].max()) or 1.0
                w = max(0.0, min(100.0, 100.0 * float(v) / mx))
                tds.append(f'<td class="num"><div class="tk-cell"><div class="tk-hb"><div class="tk-hbf" style="width:{w:.0f}%"></div></div><span>{txt}</span></div></td>')
            else:
                tds.append(f'<td class="num">{txt}</td>')
        rows.append("<tr>" + "".join(tds) + "</tr>")
    return f'<div class="tk-tbl-wrap"><table class="tk-tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


@st.cache_data(ttl=3600, show_spinner=False)

def market_snapshot():

    try:

        df = yf.download("TLKM.JK", period="1y", progress=False)

        if isinstance(df.columns, pd.MultiIndex):

            df.columns = df.columns.get_level_values(0)

        df = df.dropna()

        close = df["Close"]

        return {

            "df": df[["Close", "Volume"]],

            "last": float(close.iloc[-1]),

            "chg": float(close.iloc[-1] / close.iloc[-2] - 1) * 100,

            "hi": float(close.max()),

            "lo": float(close.min()),

            "vol": float(df["Volume"].tail(20).mean()),

            "date": df.index[-1],

        }

    except Exception:

        return None





def mape(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = np.where(y_true == 0, 1e-10, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


@st.cache_resource(show_spinner=False)
def train_pipeline(cutoff):
    """Regresi volatilitas (ML FIX TELKOM - regresi) + klasifikasi level risiko (ML FIX TELKOM - klasifikasi)."""

    # ---- Data ----
    data = yf.download("TLKM.JK", start="2005-01-01", auto_adjust=True, progress=False)
    data.reset_index(inplace=True)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    cut = pd.to_datetime(cutoff)

    # =====================================================================
    # A. REGRESI VOLATILITAS
    # =====================================================================
    weekly = data.resample("W-FRI", on="Date").agg(agg)
    weekly.dropna(inplace=True)
    weekly = weekly[weekly.index <= cut]  # filter cutoff dashboard

    weekly["Return"] = weekly["Close"].pct_change()
    weekly["Volatility"] = weekly["Return"].rolling(4).std()
    weekly["Volatility_smooth"] = weekly["Volatility"].rolling(2).mean()  # TARGET
    weekly["Vol_lag1"] = weekly["Volatility_smooth"].shift(1)
    weekly["Vol_lag2"] = weekly["Volatility_smooth"].shift(2)
    weekly["Range"] = (weekly["High"] - weekly["Low"]) / weekly["Close"]
    weekly["Return_1"] = weekly["Return"].shift(1)
    weekly["Vol_momentum"] = weekly["Vol_lag1"] - weekly["Vol_lag2"]
    weekly["Volume_change"] = np.log1p(weekly["Volume"]).diff()

    # kolom tambahan hanya untuk grafik harga (tidak dipakai model)
    weekly["MA4"] = weekly["Close"].rolling(4).mean()
    weekly["MA8"] = weekly["Close"].rolling(8).mean()

    weekly.replace([np.inf, -np.inf], np.nan, inplace=True)
    weekly.dropna(inplace=True)

    reg_cols = ["Vol_lag1", "Vol_lag2", "Range", "Return_1", "Vol_momentum", "Volume_change"]
    X = weekly[reg_cols]
    y = weekly["Volatility_smooth"]

    train_size = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

    rf_reg = RandomForestRegressor(
        n_estimators=700,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.5,
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
    )
    rf_reg.fit(X_train, y_train)

    train_pred = rf_reg.predict(X_train)
    test_pred = rf_reg.predict(X_test)

    reg_metrics = {
        "n_train": len(X_train), "n_test": len(X_test),
        "train_mae": mean_absolute_error(y_train, train_pred), "test_mae": mean_absolute_error(y_test, test_pred),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, train_pred))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_pred))),
        "train_r2": r2_score(y_train, train_pred), "test_r2": r2_score(y_test, test_pred),
        "train_mape": mape(y_train, train_pred), "test_mape": mape(y_test, test_pred),
    }

    weekly["Vol_Predicted"] = np.nan
    weekly.loc[X_test.index, "Vol_Predicted"] = test_pred

    # ---- Forecast minggu depan (fitur yang sama dengan model) ----
    latest = weekly.iloc[[-1]].copy()
    last_vol = float(latest["Volatility_smooth"].iloc[0])
    next_row = pd.DataFrame({
        "Vol_lag1": [last_vol],
        "Vol_lag2": [float(latest["Vol_lag1"].iloc[0])],
        "Range": [float(latest["Range"].iloc[0])],
        "Return_1": [float(latest["Return"].iloc[0])],
        "Vol_momentum": [last_vol - float(latest["Vol_lag1"].iloc[0])],
        "Volume_change": [float(latest["Volume_change"].iloc[0])],
    })[reg_cols]
    next_vol = float(rf_reg.predict(next_row)[0])

    # =====================================================================
    # B. KLASIFIKASI RISIKO (identik dengan ML FIX TELKOM - klasifikasi)
    # =====================================================================
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    cdata = data[data["Date"] < pd.Timestamp(today)]  # setara end=today
    cw = cdata.resample("W-FRI", on="Date").agg(agg).dropna()
    cw = cw[cw.index <= cut]  # filter cutoff dashboard

    cw["Return"] = cw["Close"].pct_change()
    cw["Volatility"] = cw["Return"].rolling(4).std()
    cw["Volatility_lag1"] = cw["Volatility"].shift(1)

    cw["MA4"] = cw["Close"].rolling(4).mean()
    cw["MA8"] = cw["Close"].rolling(8).mean()

    cw["Range"] = (cw["High"] - cw["Low"]) / cw["Close"]
    cw["Trend"] = cw["MA4"] - cw["MA8"]

    cw["Return_1"] = cw["Return"].shift(1)
    cw["Return_2"] = cw["Return"].shift(2)

    # RSI 14
    delta = cw["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    cw["RSI"] = 100 - (100 / (1 + rs))

    cw.dropna(inplace=True)

    # ---- Clustering & risk labeling ----
    scaler = StandardScaler()
    cluster_features = cw[["Volatility", "Volatility_lag1", "Range", "Trend", "RSI"]]
    X_scaled = scaler.fit_transform(cluster_features)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cw["Risk_Cluster"] = kmeans.fit_predict(X_scaled)

    cluster_stats = cw.groupby("Risk_Cluster")[["Volatility", "Range", "Trend", "RSI"]].mean()
    norm_stats = (cluster_stats - cluster_stats.min()) / (cluster_stats.max() - cluster_stats.min())
    cluster_stats["Risk_Score"] = (
        0.45 * norm_stats["Volatility"] +
        0.25 * norm_stats["Range"] +
        0.15 * (1 - norm_stats["Trend"]) +
        0.15 * norm_stats["RSI"]
    )
    sorted_clusters = cluster_stats["Risk_Score"].sort_values()
    risk_map = {
        sorted_clusters.index[0]: "LOW",
        sorted_clusters.index[1]: "MEDIUM",
        sorted_clusters.index[2]: "HIGH",
    }
    cw["Risk_Label"] = cw["Risk_Cluster"].map(risk_map)

    # ---- Data preparation ----
    cls_cols = ["Volatility_lag1", "MA4", "MA8", "Range", "Trend", "RSI"]
    X_cls = cw[cls_cols]
    y_cls = cw["Risk_Label"]

    le = LabelEncoder()
    y_enc = le.fit_transform(y_cls)

    Xc_train, Xc_test, yc_train, yc_test = train_test_split(X_cls, y_enc, test_size=0.2, shuffle=False)

    # ---- Random Forest Classifier ----
    rf_clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=7,
        min_samples_leaf=10,
        random_state=42,
        class_weight={0: 8, 1: 1, 2: 1},
    )
    rf_clf.fit(Xc_train, yc_train)

    # ---- Evaluation ----
    yc_train_pred = rf_clf.predict(Xc_train)
    yc_test_pred = rf_clf.predict(Xc_test)
    train_acc = accuracy_score(yc_train, yc_train_pred)
    test_acc = accuracy_score(yc_test, yc_test_pred)

    actual_counts = pd.Series(le.inverse_transform(yc_test)).value_counts()
    pred_counts = pd.Series(le.inverse_transform(yc_test_pred)).value_counts()
    dist_df = pd.DataFrame({"Actual": actual_counts, "Predicted": pred_counts}).fillna(0).astype(int)
    dist_df = dist_df.reindex([c for c in ["LOW", "MEDIUM", "HIGH"] if c in dist_df.index])

    cls_imp = pd.DataFrame({"Feature": X_cls.columns, "Importance": rf_clf.feature_importances_})
    cls_imp = cls_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # ---- Prediksi level risiko terbaru ----
    pred_risk = le.inverse_transform(rf_clf.predict(X_cls.iloc[[-1]]))[0]

    return {
        "next_vol": next_vol, "last_vol": last_vol, "risk": pred_risk,
        "weekly": weekly, "cls_weekly": cw, "le": le,
        "reg_metrics": reg_metrics,
        "cls": {
            "y_test": yc_test, "y_test_pred": yc_test_pred,
            "train_acc": float(train_acc), "test_acc": float(test_acc),
            "dist_df": dist_df, "imp_df": cls_imp,
        },
        "cls_acc": float(test_acc),
        "reg_imp": pd.Series(rf_reg.feature_importances_, index=reg_cols).sort_values(),
        "cls_imp": pd.Series(rf_clf.feature_importances_, index=cls_cols).sort_values(),
        "asof": latest.index[0], "last_close": float(latest["Close"].iloc[0]),
        "cutoff": str(cutoff),
    }


@st.cache_data(show_spinner=False)

def load_repository_data():

    today = pd.Timestamp.today().strftime("%Y-%m-%d")

    data = yf.download("TLKM.JK", start="2000-01-01", end=today, progress=False)

    data.reset_index(inplace=True)

    if isinstance(data.columns, pd.MultiIndex):

        data.columns = data.columns.get_level_values(0)

    w = data.resample("W-FRI", on="Date").agg(

        {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}

    ).dropna()

    w["Return"] = w["Close"].pct_change()

    w["Volatility"] = w["Return"].rolling(4).std()

    w["Volatility_lag1"] = w["Volatility"].shift(1)

    w["MA4"] = w["Close"].rolling(4).mean()

    w["MA8"] = w["Close"].rolling(8).mean()

    w["Range"] = (w["High"] - w["Low"]) / w["Close"]

    w["Trend"] = w["MA4"] - w["MA8"]

    w.dropna(inplace=True)

    w["Rolling_Return"] = w["Return"].rolling(4).mean()

    q75, q25 = w["Volatility"].quantile(0.75), w["Volatility"].quantile(0.25)



    def lab(r):

        if r["Volatility"] > q75 and r["Rolling_Return"] < 0 and r["Trend"] < 0:

            return "BURUK"

        if r["Volatility"] < q25 and r["Rolling_Return"] > 0 and r["Trend"] > 0:

            return "BAIK"

        return "WASPADA"



    w = w.dropna()

    w["Risk_Label"] = w.apply(lab, axis=1)

    return w


# ==========================================================
# 5. NAVIGATION STATE
# ==========================================================
P_HOME, P_START = "Overview", "Quick start & Setup"
P_ML, P_PERF = "ML Forecasting Engine", "Model Performance Analytics"
P_NET, P_B2B = "Network & ESG Intelligence", "B2B Enterprise Analytics"
P_AI, P_DATA = "AI Assistant (TelkomGPT)", "Data Repository"

PAGES = [P_HOME, P_ML, P_PERF, P_DATA]
NAV_LABEL = {
    P_HOME: "Home", P_ML: "Forecasting", P_PERF: "Performance",P_DATA: "Data Repository",}

if "page" not in st.session_state:
    st.session_state.page = P_HOME
if "chat" not in st.session_state:
    st.session_state.chat = []

page = st.session_state.page
res = st.session_state.get("res")
snap = market_snapshot()


def page_nav(current):
    """Tombol Sebelumnya / Selanjutnya berdasarkan urutan di PAGES."""
    if current not in PAGES:
        return
    i = PAGES.index(current)
    prev_p = PAGES[i - 1] if i > 0 else None
    next_p = PAGES[i + 1] if i < len(PAGES) - 1 else None
    with st.container(key="pagenav"):
        c_prev, c_mid, c_next = st.columns([1.7, 2, 1.9], vertical_alignment="center")
        if prev_p:
            c_prev.button(f"← Sebelumnya: {NAV_LABEL[prev_p]}", key=f"pn_prev_{i}", on_click=go, args=(prev_p,))
        if next_p:
            c_next.button(f"Selanjutnya: {NAV_LABEL[next_p]} →", key=f"pn_next_{i}", type="primary", on_click=go, args=(next_p,))

# ==========================================================
# 6. HEADER  (navbar atas + ticker berjalan + announcement)
# ==========================================================
if snap:
    arrow = "▲" if snap["chg"] >= 0 else "▼"
    live_txt = f"TLKM.JK <b>{rp(snap['last'])}</b> {arrow} {snap['chg']:+.2f}%"
    hi_lo = f"52W High <b>{rp(snap['hi'])}</b> · 52W Low <b>{rp(snap['lo'])}</b>"
else:
    live_txt = "TLKM.JK <b>market data offline</b>"
    hi_lo = "Yahoo Finance API"

ticker_items = [
    f"{live_txt}", f" {hi_lo}", " Random Forest · K-Means Risk Engine",
    " Always The Best", " 5G · Fiber · Data Center · Cloud", " Enterprise Grade Intelligence",
]
ticker_html = "".join(f"<span>{t}</span>" for t in ticker_items)

with st.container(key="topnav"):
    nav_cols = st.columns([2.3] + [1.0] * len(PAGES) + [1.35, 1.2], vertical_alignment="center")
    with nav_cols[0]:
        html(f'<div class="tk-brand"><div class="tk-logo"><img src="{LOGO_URL}" alt="Telkom Indonesia"/></div></div>')
    for col, p in zip(nav_cols[1:1 + len(PAGES)], PAGES):
        col.button(NAV_LABEL[p], key=f"nav_{p}", type="primary" if page == p else "secondary", on_click=go, args=(p,))
    with nav_cols[-2]:
        html('<div class="tk-live"><span class="tk-dot"></span> Engine Online</div>')
    
html(f"""
<div class="tk-ticker"><div class="tk-track">{ticker_html}{ticker_html}</div>
""")


# ==========================================================
# 8. FOOTER
# ==========================================================
def render_footer():
    with st.container(key="tkfoot"):
        html('<div class="tk-footer-line"></div>')
        c0, c1, c2 = st.columns([3, 1.1, 1.1], vertical_alignment="center")
        with c0:
            html("""
            <div class="tk-fcta"><h3>Siap membangun keputusan yang <span>lebih cerdas?</span></h3>
            <p>Mulai dari pipeline ML, atau langsung ambil datanya lewat repository.</p></div>
            """)
        

        html(f"""
        <div class="ftop">
        
        
        <p class="fsub">Intelligence Platform · Always The Best</p>
        <div class="fst">
        
        <div><b>2</b><span>Model ML</span></div>
        <div><b>{YEAR - 2005}+</b><span>Tahun data</span></div>
        <div><b>W-FRI</b><span>Resampling mingguan</span></div>
        </div>
        </div>
        """)

        with st.container(key="footlinks"):
            f0, f1, f2, f3 = st.columns([1.8, 1, 1, 1.2])
            with f0:
                html(f"""
                <h4>Telkom Intelligence Platform</h4>
                <p>Sistem peringatan dini risiko keuangan berbasis Machine Learning untuk saham TLKM.JK.</p>
                <div class="tags"><span>Random Forest</span><span>K-Means</span><span>Streamlit</span><span>Yahoo Finance</span></div>
                """)
            with f1:
                html("<h4>Platform</h4>")
                for p, lbl in [(P_ML, "ML Forecasting"), (P_PERF, "Model Performance")]:
                    st.button(lbl, key=f"ft_{p}", on_click=go, args=(p,))
            with f2:
                html("<h4>Resources</h4>")
                for p, lbl in [(P_DATA, "Data Repository")]:
                    st.button(lbl, key=f"ft_{p}", on_click=go, args=(p,))
            with f3:
                html("""
                <h4>Company</h4>
                <p>PT Telkom Indonesia (Persero) Tbk</p>
                <p>Jakarta, Indonesia</p>
                <p>Vision 2030 · Digital Telco</p>
                """)

        html(f"""
        <div class="fb">
        <span>© {YEAR} <b>PT Telkom Indonesia (Persero) Tbk</b> — Financial Risk Management Dashboard v3.5</span>
        <span class="fok"><span class="tk-dot"></span>Engine Online</span><span>Data bersifat analitik dan bukan saran investasi · Built with Streamlit</span>
        </div>
        """)


# Isi halaman berada di wadah "body" yang dipusatkan; header & footer tetap selebar layar.
_body = st.container(key="body")
_body.__enter__()

# ==========================================================
# 9. PAGES
# ==========================================================

# ---------------------- OVERVIEW ----------------------
if page == P_HOME:
    # --- Hero: teks + panel harga live ---
    if snap:
        up = snap["chg"] >= 0
        glass = f"""
        <div class="glass">
        <div class="gh"><span class="tk-dot"></span>Live · TLKM.JK<span class="gd">{snap['date'].strftime("%d %b %Y")}</span></div>
        <div class="gp">{rp(snap['last'])}</div>
        <div class="gc {'up' if up else 'dn'}">{'▲' if up else '▼'} {snap['chg']:+.2f}%<small>vs previous close</small></div>
        <div class="gg">
        <div><small>52W High</small><b>{rp(snap['hi'])}</b></div>
        <div><small>52W Low</small><b>{rp(snap['lo'])}</b></div>
        <div><small>Avg Vol 20D</small><b>{snap['vol'] / 1e6:,.1f} M</b></div>
        </div></div>
        """
    else:
        glass = """
        <div class="glass">
        <div class="gh">Telkom Intelligence Platform</div>
        <div class="gp">TLKM.JK</div>
        <div class="gc">Market data offline</div>
        <div class="gg">
        <div><small>Model ML</small><b>2 model</b></div>
        <div><small>Data</small><b>2005 – kini</b></div>
        <div><small>Resampling</small><b>W-FRI</b></div>
        </div></div>
        """

    with st.container(key="hero"):
        hl, hr = st.columns([1.3, 1], gap="large", vertical_alignment="center")
        with hl:
            html(f"""
            <div class="hero-logo"><img src="{LOGO_URL}" alt="Telkom Indonesia"/></div>
            <h1 class="hero-h1">Peringatan dini risiko saham, berbasis data.</h1>
            <p class="hero-p">Prediksi volatilitas dan level risiko TLKM.JK mingguan dengan Random Forest.</p>
            
            """)
            b1, b2, _ = st.columns([1.2, 1, 0.6])
            b1.button("Jalankan ML Pipeline", key="hero_b1", on_click=go, args=(P_ML,))
            
        with hr:
            html(glass)

    # --- Jelajahi platform ---
    section("Jelajahi", "Platform")
    tiles = [
        ("", "ML Forecasting", "Latih model Random Forest untuk volatilitas dan level risiko.", P_ML),
        ("", "Model Performance", "MAE, RMSE, R², confusion matrix, dan akurasi model.", P_PERF),
        ("", "Data Repository", "Unduh dataset risk regime.", P_DATA),
    ]
    for col, i in zip(st.columns(3), range(3)):
        ic, t, d, target = tiles[i]  # Indentasi diperbaiki (ditambah 4 spasi)
        with col:
            with st.container(key=f"tile{i}"):
                html(f'<h4>{t}</h4><p>{d}</p>')
                st.button("Buka halaman →", key=f"tilb_{i}", on_click=go, args=(target,))

    # --- Grafik harga ---
    if snap:
        section("Live Market", "Pulse")
        fig, ax = new_fig(11, 3.2)
        close = snap["df"]["Close"]
        ax.plot(close.index, close.values, color=RED, linewidth=2)
        ax.plot(close.index, close.rolling(20).mean(), color=GREY, linewidth=1.2, linestyle="--", label="MA20")
        ax.fill_between(close.index, close.values, close.min(), color=RED, alpha=0.08)
        ax.set_title("TLKM.JK — Harga Penutupan 1 Tahun Terakhir", loc="left", fontsize=12, fontweight="bold", color="#0f172a")
        ax.legend(frameon=False)
        show(fig)

    # --- Profil ---
    st.write("")
    ca, cv = st.columns([1.5, 1])
    ca.markdown("""
    <div class="info-card"><h3> Corporate Profile</h3>
    <p>PT Telkom Indonesia (Persero) Tbk adalah perusahaan informasi dan komunikasi serta penyedia jasa dan jaringan
    telekomunikasi secara lengkap di Indonesia.</p>
    <p><strong>Tahun Berdiri:</strong> 1965 | <strong>Fokus:</strong> Connectivity, Platform, Services.</p></div>
    """, unsafe_allow_html=True)
    cv.markdown("""
    <div class="vision-card"><h4> Vision 2030</h4>
    <p><i>"Menjadi digital telco pilihan utama untuk memajukan masyarakat."</i></p><hr>
    <h4> Mission</h4>
    <p>Memberikan layanan digital berkualitas tinggi untuk mendorong ekonomi bangsa.</p></div>
    """, unsafe_allow_html=True)

# ---------------------- QUICK START ----------------------
elif page == P_START:
    page_hero("Quick Start Guide", "Learn how to connect your Python workspace to Telkom Intelligence API endpoints.")
    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="step-card"><div class="step-no">1</div><h4>Install</h4><p>Pasang library yang dibutuhkan dashboard.</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="step-card"><div class="step-no">2</div><h4>Run</h4><p>Jalankan aplikasi Streamlit dari terminal.</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="step-card"><div class="step-no">3</div><h4>Integrate</h4><p>Gunakan SDK / REST API untuk integrasi data.</p></div>', unsafe_allow_html=True)

    section("1. Install", "dependencies")
    st.code("pip install streamlit pandas numpy matplotlib scikit-learn yfinance", language="bash")
    section("2. Jalankan", "dashboard")
    st.code("streamlit run Belajar_dashboard3.py", language="bash")
    section("3. Contoh SDK", "(ilustrasi)")
    st.code('''# Install Telkom Intelligence SDK
pip install telkom-intelligence-sdk pandas

# Initialize Client
from telkom_intel import MarketClient

client = MarketClient(api_key="TLKM_SECRET_KEY")
df = client.get_weekly_risk(ticker="TLKM.JK")
print(df.head())''', language="python")
    st.caption("SDK di atas hanya contoh ilustrasi dokumentasi.")

# ---------------------- ML FORECASTING ----------------------
elif page == P_ML:
    page_hero(" ML Forecasting & Risk Regime",
              "Predict next-week price volatility and risk classification for TLKM.JK.")

    cc1, cc2 = st.columns([1, 2])
    cutoff = cc1.date_input(" Cutoff Date Filter", value=pd.Timestamp.today().date(), max_value=pd.Timestamp.today().date())
    cc2.write("")
    if cc2.button(" Run Machine Learning", type="primary"):
        try:
            with st.spinner("Executing models..."):
                st.session_state.res = train_pipeline(cutoff)
            st.success("Pipeline executed successfully!")
        except Exception as e:
            st.error(f"Pipeline gagal dijalankan: {e}")
        res = st.session_state.get("res")

    if res is None:
        st.info(" Pilih tanggal lalu klik **Run Machine Learning Pipeline** untuk melihat hasil.")
    else:
        st.caption(f"Data terakhir: minggu {res['asof'].strftime('%d %b %Y')} · Close {rp(res['last_close'])} · Cutoff {res['cutoff']}")
        k1, k2, k3 = st.columns(3)
        k1.markdown(metric_card(" Forecast Volatility (Next Week)", f"{res['next_vol']:.4f}", "Random Forest Regression", BLUE), unsafe_allow_html=True)
        k2.markdown(risk_card(res["risk"]), unsafe_allow_html=True)

        w = res["weekly"]
        t1, t2, t3, t4 = st.tabs([" Volatility", " Price & Trend", " Risk Timeline", " Feature Importance"])

        with t1:
            c1, c2 = st.columns([1, 2])
            with c1:
                fig, ax = new_fig(4, 3.4)
                ax.plot(["Minggu Ini", "Minggu Depan"], [res["last_vol"], res["next_vol"]], marker="o", linewidth=2.5, color=RED, markersize=9)
                ax.set_title("Prediksi Volatility", loc="left", fontsize=11, fontweight="bold")
                ax.set_ylabel("Volatility")
                show(fig)
            with c2:
                ev = w.dropna(subset=["Vol_Predicted"])
                fig, ax = new_fig(8, 3.4)
                ax.plot(ev.index, ev["Volatility_smooth"], color=GREY, label="Actual", linewidth=1.5)
                ax.plot(ev.index, ev["Vol_Predicted"], color=RED, label="Predicted", linewidth=1.8)
                ax.set_title("Actual vs Predicted (Test Set)", loc="left", fontsize=11, fontweight="bold")
                ax.legend(frameon=False)
                show(fig)

        with t2:
            p = w.tail(156)
            fig, ax = new_fig(11, 3.6)
            ax.plot(p.index, p["Close"], color="#0f172a", linewidth=1.6, label="Close")
            ax.plot(p.index, p["MA4"], color=RED, linewidth=1.3, label="MA4")
            ax.plot(p.index, p["MA8"], color=AMBER, linewidth=1.3, label="MA8")
            ax.set_title("Harga Mingguan & Moving Average (±3 tahun)", loc="left", fontsize=11, fontweight="bold")
            ax.legend(frameon=False, ncol=3)
            show(fig)

        with t3:
            cw = res["cls_weekly"]
            p = cw.tail(260)
            cmap = {"LOW": GREEN, "MEDIUM": AMBER, "HIGH": "#DC3545"}
            fig, ax = new_fig(11, 3.6)
            ax.plot(p.index, p["Close"], color="#cbd5e1", linewidth=1)
            for lab, col in cmap.items():
                sub = p[p["Risk_Label"] == lab]
                ax.scatter(sub.index, sub["Close"], s=18, color=col, label=lab)
            ax.set_title("Risk Regime pada Harga (±5 tahun)", loc="left", fontsize=11, fontweight="bold")
            ax.legend(frameon=False, ncol=3)
            show(fig)
            cnt = cw["Risk_Label"].value_counts()
            r1, r2, r3 = st.columns(3)
            r1.markdown(metric_card("LOW Weeks", int(cnt.get("LOW", 0)), "Risiko rendah", GREEN), unsafe_allow_html=True)
            r2.markdown(metric_card("MEDIUM Weeks", int(cnt.get("MEDIUM", 0)), "Risiko menengah", AMBER), unsafe_allow_html=True)
            r3.markdown(metric_card("HIGH Weeks", int(cnt.get("HIGH", 0)), "Risiko tinggi", "#DC3545"), unsafe_allow_html=True)

        with t4:
            f1, f2 = st.columns(2)
            with f1:
                fig, ax = new_fig(5, 3.4)
                ax.barh(res["reg_imp"].index, res["reg_imp"].values, color=RED)
                ax.set_title("Regressor Volatilitas", loc="left", fontsize=11, fontweight="bold")
                show(fig)
            with f2:
                fig, ax = new_fig(5, 3.4)
                ax.barh(res["cls_imp"].index, res["cls_imp"].values, color=DARK_RED)
                ax.set_title("Classifier Level Risiko", loc="left", fontsize=11, fontweight="bold")
                show(fig)
        st.caption("⚠️ Hasil model bersifat analitik dan bukan saran investasi.")

# ---------------------- MODEL PERFORMANCE ----------------------
elif page == P_PERF:
    page_hero(" Model Performance Analytics", "Validation results, R² scores, MAE, RMSE, and Confusion Matrix for trained models.")
    if res is None:
        st.info(" Silakan jalankan **ML Forecasting Engine** terlebih dahulu.")
        st.button("Ke ML Forecasting Engine →", on_click=go, args=(P_ML,))
    else:
        weekly, le = res["weekly"], res["le"]

        tab_reg, tab_cls = st.tabs(["Regression Evaluation", "Classification Performance"])

        # ---------------- TAB 1: REGRESSION EVALUATION ----------------
        with tab_reg:
            section("Historical Volatility:", "Actual vs Predicted")
            ev = weekly.dropna(subset=["Vol_Predicted"])
            fig, ax = new_fig(11, 3.4)
            ax.plot(ev.index, ev["Volatility_smooth"], color=GREY, label="Actual Volatility", linewidth=1.5)
            ax.plot(ev.index, ev["Vol_Predicted"], color=RED, label="Predicted Volatility (RF)", linewidth=1.8)
            ax.legend(frameon=False)
            show(fig)

            m = res["reg_metrics"]
            st.caption(f"Data split (time-based, tanpa shuffle): Train {m['n_train']} minggu · Test {m['n_test']} minggu")

            section("Regression", "Evaluation Metrics")
            a, b, c, d = st.columns(4)
            a.markdown(metric_card("MAE", f"{m['test_mae']:.6f}", "Mean Absolute Error", BLUE), unsafe_allow_html=True)
            b.markdown(metric_card("RMSE", f"{m['test_rmse']:.6f}", "Root Mean Squared Error", AMBER), unsafe_allow_html=True)
            c.markdown(metric_card("R² Score", f"{m['test_r2']:.4f}", "Goodness of Fit", GREEN), unsafe_allow_html=True)
            d.markdown(metric_card("MAPE", f"{m['test_mape']:.2f}%", "Mean Absolute % Error", PURPLE), unsafe_allow_html=True)

        # ---------------- TAB 2: CLASSIFICATION PERFORMANCE ----------------
        with tab_cls:
            section("Risk Classification", "Performance")
            cls = res["cls"]
            s1, s2 = st.columns(2)
            s1.markdown(metric_card("Test Accuracy", f"{cls['test_acc'] * 100:.2f}%", "Data uji", GREEN), unsafe_allow_html=True)
            s2.markdown(metric_card("Test Samples", f"{len(cls['y_test'])}", "minggu data uji", AMBER), unsafe_allow_html=True)
            st.write("")

            yt, yp = cls["y_test"], cls["y_test_pred"]
            labels_idx = list(range(len(le.classes_)))
            cm = confusion_matrix(yt, yp, labels=labels_idx)
            col_cm, col_rep = st.columns([1, 1.3])
            with col_cm:
                fig, ax = plt.subplots(figsize=(5, 4))
                ConfusionMatrixDisplay(cm, display_labels=le.classes_).plot(ax=ax, cmap=plt.cm.Reds, colorbar=False)
                ax.set_title("Confusion Matrix - Risk Classification", fontsize=11, fontweight="bold")
                show(fig)
            with col_rep:
                st.write("#### Classification Report (Test)")
                rep = classification_report(yt, yp, labels=labels_idx, target_names=le.classes_, output_dict=True, zero_division=0)
                rdf = pd.DataFrame({k: rep[k] for k in list(le.classes_) + ["macro avg", "weighted avg"]}).T
                rdf = rdf[["precision", "recall", "f1-score", "support"]]
                acc_row = pd.DataFrame({"precision": [np.nan], "recall": [np.nan], "f1-score": [rep["accuracy"]],
                                        "support": [rdf.loc[list(le.classes_), "support"].sum()]}, index=["accuracy"])
                n_cls = len(le.classes_)
                rdf = pd.concat([rdf.iloc[:n_cls], acc_row, rdf.iloc[n_cls:]])
                html(html_table(rdf, fmt={"precision": "{:.3f}", "recall": "{:.3f}", "f1-score": "{:.3f}", "support": "{:,.0f}"},
                                bar_cols=("f1-score",), index_label="Class"))

# ---------------------- NETWORK & ESG ----------------------
elif page == P_NET:
    page_hero(" Network Performance & ESG Metrics",
              "Telemetry of Telkom 5G BTS rollout, fiber optics uptime, and ESG sustainability ratings (data ilustrasi).")
    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(metric_card("5G BTS Active Nodes", "24,500", "+12% YoY", RED), unsafe_allow_html=True)
    m2.markdown(metric_card("IndiHome Fiber Uptime", "99.98%", "+0.02%", GREEN), unsafe_allow_html=True)
    m3.markdown(metric_card("ESG Sustainability Rating", "Low Risk (18.4)", "Top 10 Asian Telco", BLUE), unsafe_allow_html=True)
    m4.markdown(metric_card("Data Center Capacity", "62 MW", "+8% YoY", PURPLE), unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        yrs = ["2021", "2022", "2023", "2024", "2025", "2026"]
        bts = [4.2, 8.9, 13.6, 18.2, 21.8, 24.5]
        fig, ax = new_fig(6, 3.6)
        ax.bar(yrs, bts, color=[RED if y == "2026" else "#f5a09b" for y in yrs])
        ax.set_title("5G BTS Rollout (ribu node)", loc="left", fontsize=11, fontweight="bold")
        show(fig)
    with c2:
        months = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]
        up = [99.94, 99.95, 99.96, 99.95, 99.97, 99.98, 99.97, 99.98, 99.99, 99.98, 99.98, 99.98]
        fig, ax = new_fig(6, 3.6)
        ax.plot(months, up, marker="o", color=RED, linewidth=2)
        ax.set_ylim(99.9, 100)
        ax.set_title("Fiber Uptime Bulanan (%)", loc="left", fontsize=11, fontweight="bold")
        show(fig)

    section("Cakupan", "regional")
    st.dataframe(pd.DataFrame({
        "Region": ["Sumatera", "Jawa", "Kalimantan", "Sulawesi", "Bali-Nusa Tenggara", "Maluku-Papua"],
        "5G Coverage (%)": [38, 71, 29, 33, 46, 14],
        "Fiber Homepass (juta)": [4.1, 12.6, 1.8, 2.2, 1.3, 0.6],
        "ESG Score": [21.2, 16.9, 22.5, 20.4, 18.8, 24.1],
    }))
    st.caption("Seluruh angka pada halaman ini adalah data ilustrasi, bukan data resmi perusahaan.")

# ---------------------- B2B ----------------------
elif page == P_B2B:
    page_hero(" Enterprise & B2B Portfolio", "Telkomsel Enterprise, Indibiz, and Data Center expansion metrics (simulasi).")
    k1, k2, k3, k4 = st.columns(4)
    k1.markdown(metric_card("Indibiz SME", "1.2 jt", "pelanggan aktif", RED), unsafe_allow_html=True)
    k2.markdown(metric_card("Enterprise Clients", "8,450", "korporasi", BLUE), unsafe_allow_html=True)
    k3.markdown(metric_card("Churn Probability", "2.7%", "rata-rata 12 bulan", AMBER), unsafe_allow_html=True)
    k4.markdown(metric_card("Cloud Growth", "+18%", "YoY", GREEN), unsafe_allow_html=True)

    st.write("")
    rng = np.random.default_rng(42)
    q = pd.date_range(end=pd.Timestamp.today().to_period("Q").start_time, periods=20, freq="QS")
    data_b2b = pd.DataFrame({
        "Indibiz SME": 100 + np.cumsum(rng.normal(3, 2, 20)),
        "Enterprise Solutions": 140 + np.cumsum(rng.normal(4, 3, 20)),
        "Data Center Cloud": 60 + np.cumsum(rng.normal(5, 2.5, 20)),
    }, index=q)

    fig, ax = new_fig(11, 3.8)
    ax.stackplot(data_b2b.index, data_b2b.T.values, labels=data_b2b.columns, colors=[RED, DARK_RED, "#f5a09b"], alpha=0.9)
    ax.set_title("Corporate Segment Revenue (Simulation, indeks)", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    show(fig)

    section("Churn risk", "per segmen")
    st.dataframe(pd.DataFrame({
        "Segment": ["Indibiz SME", "Enterprise Solutions", "Data Center Cloud", "Managed Services"],
        "Churn Probability (%)": [3.9, 1.8, 1.2, 2.6],
        "Revenue Share (%)": [28, 41, 18, 13],
        "Risk": ["MEDIUM", "LOW", "LOW", "LOW"],
    }))
    st.caption("Data simulasi untuk keperluan demonstrasi dashboard.")

# ---------------------- AI ASSISTANT ----------------------
elif page == P_AI:
    page_hero("TelkomGPT Financial Assistant",
              "Ask natural language questions regarding TLKM stock performance and risk indicators.")

    def tg_answer(q, r):
        ql = q.lower()
        if any(k in ql for k in ["help", "bantu", "bisa apa"]):
            return "Tanya saya tentang **risk level**, **volatilitas**, atau **akurasi model**."
        if r is None:
            return "Belum ada hasil model. Jalankan **ML Forecasting Engine** dulu agar saya bisa menjawab berdasarkan prediksi terbaru."
        if "risk" in ql or "risiko" in ql:
            return (f"Prediksi level risiko minggu depan: **{r['risk']}** "
                    f"(data hingga minggu {r['asof'].strftime('%d %b %Y')}).")
        if "vol" in ql:
            trend = "naik" if r["next_vol"] > r["last_vol"] else "turun"
            return (f"Volatilitas minggu ini **{r['last_vol']:.4f}** dan diprediksi **{trend}** menjadi "
                    f"**{r['next_vol']:.4f}** minggu depan.")
        if any(k in ql for k in ["akurasi", "accuracy", "performa"]):
            return (f"Regresi volatilitas (test): R² **{r['reg_metrics']['test_r2']:.4f}**, "
                    f"MAE **{r['reg_metrics']['test_mae']:.6f}**. "
                    f"Akurasi klasifikasi risiko: **{r['cls_acc'] * 100:.2f}%**.")
        return (f"Ringkasan: risiko **{r['risk']}**, volatilitas next week **{r['next_vol']:.4f}**.")

    qa = st.columns(3)
    quick = ["Risk level minggu depan?", "Prediksi volatilitas?", "Berapa akurasi model?"]
    for col, qq in zip(qa, quick):
        if col.button(qq, key=f"q_{qq}"):
            st.session_state.chat.append(("user", qq))
            st.session_state.chat.append(("assistant", tg_answer(qq, res)))

    prompt = st.chat_input("Ask a question about Telkom market data, e.g. What is the predicted risk level for next week?")
    if prompt:
        st.session_state.chat.append(("user", prompt))
        st.session_state.chat.append(("assistant", tg_answer(prompt, res)))

    if not st.session_state.chat:
        st.info("Mulai percakapan dengan pertanyaan di atas atau ketik pertanyaan Anda.")
    for role, msg in st.session_state.chat:
        with st.chat_message(role):
            st.markdown(msg)

# ---------------------- DATA REPOSITORY ----------------------
else:
    page_hero(" Data Repository",
              "Pusat data historis, fitur Machine Learning, dan informasi risiko saham TLKM sebagai dasar sistem peringatan dini.")
    try:
        with st.spinner("Loading repository data..."):
            repo = load_repository_data()
    except Exception as e:
        st.error(f"Gagal memuat data: {e}")
        repo = None

    if repo is not None:
        rng_opt = st.select_slider("Rentang data", ["1 tahun", "3 tahun", "5 tahun", "10 tahun", "Semua"], value="Semua")
        yrs = {"1 tahun": 1, "3 tahun": 3, "5 tahun": 5, "10 tahun": 10}.get(rng_opt)
        view = repo if yrs is None else repo[repo.index >= repo.index.max() - pd.DateOffset(years=yrs)]

        cnt = view["Risk_Label"].value_counts()
        r1, r2, r3, r4 = st.columns(4)
        r1.markdown(metric_card("Total Weeks", len(view), "baris data", BLUE), unsafe_allow_html=True)
        r2.markdown(metric_card("BAIK", int(cnt.get("BAIK", 0)), "minggu", GREEN), unsafe_allow_html=True)
        r3.markdown(metric_card("WASPADA", int(cnt.get("WASPADA", 0)), "minggu", AMBER), unsafe_allow_html=True)
        r4.markdown(metric_card("BURUK", int(cnt.get("BURUK", 0)), "minggu", "#DC3545"), unsafe_allow_html=True)
        st.write("")

        section("Risk Regime", "Dataset")
        st.caption("Label risiko berbasis indikator multi-faktor (volatilitas, rolling return, trend).")

        fmt = {c: "{:,.0f}" for c in ["Open", "High", "Low", "Close", "MA4", "MA8", "Volume"] if c in view.columns}
        fmt.update({c: "{:.4f}" for c in ["Return", "Volatility", "Volatility_lag1", "Range", "Rolling_Return"] if c in view.columns})
        fmt["Trend"] = "{:,.2f}"

        def _label_style(v):
            return {"BAIK": "background-color:#e9f7ee;color:#15803d;font-weight:700;text-align:center",
                    "WASPADA": "background-color:#fff4e0;color:#b45309;font-weight:700;text-align:center",
                    "BURUK": "background-color:#fdecea;color:#b91c1c;font-weight:700;text-align:center"}.get(v, "")

        sty = view.style.format(fmt).format_index("{:%d %b %Y}")
        sty = (sty.map if hasattr(sty, "map") else sty.applymap)(_label_style, subset=["Risk_Label"])
        sty = sty.background_gradient(cmap="Reds", subset=["Volatility"], vmin=0)
        st.dataframe(sty, height=480)
        st.download_button(
            "⬇ Download Risk Regime Data",
            view[["Volatility", "Rolling_Return", "Trend", "Risk_Label"]].to_csv(index=True).encode("utf-8"),
            "TLKM_risk_regime_multifactor.csv", "text/csv",
        )

page_nav(page)

_body.__exit__(None, None, None)
render_footer()
