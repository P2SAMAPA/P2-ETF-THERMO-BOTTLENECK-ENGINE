# P2-ETF-THERMO-BOTTLENECK

**Loss of order detection** using Information Bottleneck, Transfer Entropy, and Maximum Entropy Production rate. Predicts trend collapse (sharp drawdowns or reversals).

## Features

- **Variational Information Bottleneck** (compression efficiency) – measures whether past returns predict future.
- **Transfer Entropy** from macro (VIX, DXY, etc.) to ETF – decreasing external influence signals instability.
- **MEP rate** (KL divergence forward/backward) – quantifies time‑asymmetry.
- **Collapse Score** = -IB - TE + MEP. High score → collapse imminent.
- Tests 60, 120, 252‑day windows, selects the one maximising correlation of collapse score with future drawdown.
- Outputs per‑universe collapse warning and top risk ETFs.

## Data

Uses `P2SAMAPA/fi-etf-macro-signal-master-data`.  
Results stored in `P2SAMAPA/p2-etf-thermo-bottleneck-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-THERMO-BOTTLENECK.git
cd P2-ETF-THERMO-BOTTLENECK
pip install -r requirements.txt

Tishby, Pereira, Bialek (1999) – Information Bottleneck.

Schreiber (2000) – Transfer Entropy.

Dewar (2003) – Maximum Entropy Production.
