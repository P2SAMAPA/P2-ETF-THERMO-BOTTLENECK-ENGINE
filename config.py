"""
Configuration for P2-ETF-THERMO-BOTTLENECK engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-thermo-bottleneck-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

# --- Macro features for Transfer Entropy ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Thermo-Bottleneck parameters ---
WINDOWS = [60, 120, 252]                     # rolling estimation windows (days)
LOOKAHEAD = 5                                # days ahead for collapse verification
VIB_LATENT_DIM = 2                           # bottleneck size
VIB_BETA = 0.5                               # IB trade‑off parameter
TE_LAG = 1                                   # lag for transfer entropy
MEP_BINS = 20                                # bins for KL estimation
COLLAPSE_THRESHOLD = 1.0                     # score above which warning issued
TOP_N_DESTAB = 3                             # number of highest‑risk ETFs to display

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
