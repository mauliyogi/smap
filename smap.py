import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import ta
import io, time, warnings
import plotly.express as px
from math import ceil

warnings.filterwarnings("ignore", category=FutureWarning)

# === PAGE CONFIG ===
st.set_page_config(page_title="Smart Money Stock Screener", layout="wide", initial_sidebar_state="expanded")

# === HEADER ===
st.title("📊 Smart Money Stock Screener Dashboard")
st.markdown("""
Analisis **Smart Money** mendeteksi potensi **akumulasi institusional** berdasarkan kombinasi indikator teknikal seperti
volume, momentum, volatilitas, dan kekuatan relatif terhadap pasar.
Gunakan tombol **🔄 Refresh** untuk mengambil data baru dan **📤 Export** untuk mengunduh hasilnya.
""")

# === SETTINGS ===
period = "6mo"
interval = "1d"
batch_size = 50
tickers_df = pd.read_excel("tickers.xlsx", header=None)
idx_tickers = tickers_df[0].dropna().tolist()

# === BUTTONS ===
col1, col2 = st.columns(2)
refresh = col1.button("🔄 Refresh Data (Recalculate)")
export = col2.button("📤 Export to Excel")

# === SMARTSCORE DESCRIPTION ===
st.sidebar.header("💡 SmartScore System")
st.sidebar.markdown("""
**SmartScore** adalah skor kumulatif (0–15+) berdasarkan indikator berikut. Setiap indikator mencerminkan minat institusional atau momentum harga:

---

### 1. CMF_Pos (Chaikin Money Flow Positif)
- **Mengukur:** Aliran dana bersih ke saham dalam 20 periode.
- **Rumus:** Σ[(Close-Low)-(High-Close)]*Volume / (High-Low)*Volume
- **Interpretasi:** CMF > 0.1 menunjukkan aliran institusional yang kuat.

### 2. RVOL_High (Relative Volume)
- **Mengukur:** Volume saat ini dibanding rata-rata 20 hari.
- **Rumus:** Volume / Rata-rata(Volume_20)
- **Interpretasi:** RVOL > 2 menandakan aktivitas tidak biasa, sering sebelum breakout.

### 3. ADX_Strong (Average Directional Index)
- **Mengukur:** Kekuatan tren.
- **Interpretasi:** ADX > 25 = tren kuat.

### 4. DI_Trend (+DI vs -DI)
- **Mengukur:** Arah tren.
- **Interpretasi:** +DI > -DI = tren naik.

### 5. OBV_Up (On-Balance Volume Slope)
- **Mengukur:** Tren akumulasi volume beli/jual.
- **Interpretasi:** Slope positif = akumulasi.

### 6. RS_Strength (Relative Strength vs IHSG)
- **Mengukur:** Kinerja saham dibanding IHSG.
- **Interpretasi:** RS > 1 = saham mengungguli pasar.

### 7. Vol_Contract (Kontraksi Volatilitas)
- **Mengukur:** Volatilitas terkini vs rata-rata 60 hari.
- **Interpretasi:** Volatilitas menyusut sering mendahului breakout.

### 8. Breakout
- **Mengukur:** Harga menutup di atas tertinggi 20 hari sebelumnya.
- **Interpretasi:** Mengonfirmasi potensi breakout.

### 9. Volume_Pattern
- **Mengukur:** Pola “Dry-up lalu surge” volume.
- **Interpretasi:** Volume awal rendah diikuti lonjakan = akumulasi institusional.

### 10. Force_Pos (Force Index)
- **Mengukur:** Momentum pergerakan harga berbobot volume.
- **Interpretasi:** Positif = tekanan beli.

### 11. MFI_Strong (Money Flow Index)
- **Mengukur:** Volume + momentum harga.
- **Interpretasi:** MFI > 60 = tekanan beli kuat.

### 12. RSI_Trend_Pos (RSI Trend)
- **Mengukur:** Momentum relatif periode sebelumnya.
- **Interpretasi:** RSI > 50 & naik = tren bullish.

### 13. Above_VWAP (Harga di atas VWAP)
- **Mengukur:** Harga dibanding VWAP.
- **Interpretasi:** Harga di atas VWAP = kekuatan bullish.

### 14. ADL_Up (Accumulation/Distribution Line Slope)
- **Mengukur:** Aliran uang kumulatif (beli vs jual).
- **Interpretasi:** Slope positif = akumulasi oleh smart money.

---

**Klasifikasi Label berdasarkan SmartScore:**

| Rentang Skor | Label |
|--------------|-------|
| ≥ 12         | 🚀 Institutional Breakout |
| 9–11         | 📈 Strong Accumulation |
| 6–8          | 💰 Early Accumulation |
| < 6          | 🕓 Netral / Lemah |
""")

@st.cache_data(ttl=3600)
def run_screener():
    results, failed = [], []
    ihsg = yf.download("^JKSE", period=period, interval=interval, progress=False)
    ihsg_close = ihsg["Close"].squeeze()
    total_batches = ceil(len(idx_tickers) / batch_size)
    progress = st.progress(0)

    for batch_num in range(total_batches):
        batch = idx_tickers[batch_num * batch_size:(batch_num + 1) * batch_size]
        try:
            batch_data = yf.download(batch, period=period, interval=interval, progress=False, auto_adjust=False, group_by="ticker")
        except Exception as e:
            failed.extend(batch)
            continue

        for symbol in batch:
            try:
                if symbol not in batch_data or batch_data[symbol].empty:
                    failed.append(symbol)
                    continue
                data = batch_data[symbol].copy()
                if len(data) < 40:
                    failed.append(symbol)
                    continue

                # === Indicators ===
                data["OBV"] = ta.volume.on_balance_volume(data["Close"], data["Volume"])
                data["OBV_Slope"] = data["OBV"].diff(5)
                data["CMF"] = ta.volume.chaikin_money_flow(data["High"], data["Low"], data["Close"], data["Volume"])
                data["ADX"] = ta.trend.adx(data["High"], data["Low"], data["Close"], 14)
                data["DI_pos"] = ta.trend.adx_pos(data["High"], data["Low"], data["Close"], 14)
                data["DI_neg"] = ta.trend.adx_neg(data["High"], data["Low"], data["Close"], 14)
                data["RVOL"] = data["Volume"] / data["Volume"].rolling(20).mean()
                data["MFI"] = ta.volume.money_flow_index(data["High"], data["Low"], data["Close"], data["Volume"], 14)
                data["RSI"] = ta.momentum.rsi(data["Close"], 14)
                data["RSI_Trend"] = data["RSI"].diff(5)
                data["VWAP"] = ta.volume.volume_weighted_average_price(data["High"], data["Low"], data["Close"], data["Volume"], 14)
                data["ADL"] = ta.volume.acc_dist_index(data["High"], data["Low"], data["Close"], data["Volume"])
                data["ADL_Slope"] = data["ADL"].diff(5)
                data["Range"] = (data["High"] - data["Low"]) / data["Close"]
                data["Volatility"] = data["Range"].rolling(10).std()
                data["Volatility_Contraction"] = data["Volatility"] < data["Volatility"].rolling(60).mean() * 0.8
                data["Breakout"] = data["Close"] > data["Close"].rolling(20).max().shift(1)
                data["Vol_DryUp"] = data["Volume"] < data["Volume"].rolling(20).mean() * 0.5
                data["Vol_Surge"] = data["Volume"] > data["Volume"].rolling(20).mean() * 2
                data["ForceIndex"] = data["Close"].diff() * data["Volume"]
                data["Force_Pos"] = data["ForceIndex"].rolling(10).mean() > 0

                # === Relative to IHSG ===
                ihsg_aligned = ihsg_close.reindex(data.index).fillna(method="ffill")
                data["RS"] = data["Close"] / ihsg_aligned
                data["RS_MA"] = data["RS"].rolling(20).mean()
                data["RS_Strength"] = data["RS"] > data["RS_MA"]

                latest = data.iloc[-1]

                indicators = {
                    "CMF_Pos": latest["CMF"] > 0.1,
                    "RVOL_High": latest["RVOL"] > 2,
                    "ADX_Strong": latest["ADX"] > 25,
                    "DI_Trend": latest["DI_pos"] > latest["DI_neg"],
                    "OBV_Up": latest["OBV_Slope"] > 0,
                    "RS_Strength": latest["RS_Strength"],
                    "Vol_Contract": latest["Volatility_Contraction"],
                    "Breakout": latest["Breakout"],
                    "Volume_Pattern": latest["Vol_Surge"] and data["Vol_DryUp"].rolling(10).sum().iloc[-2] > 3,
                    "Force_Pos": latest["Force_Pos"],
                    "MFI_Strong": latest["MFI"] > 60,
                    "RSI_Trend_Pos": latest["RSI"] > 50 and latest["RSI_Trend"] > 0,
                    "Above_VWAP": latest["Close"] > latest["VWAP"],
                    "ADL_Up": latest["ADL_Slope"] > 0,
                }

                score = sum(int(v) for v in indicators.values())
                if score >= 12:
                    label = "🚀 Institutional Breakout"
                elif score >= 9:
                    label = "📈 Strong Accumulation"
                elif score >= 6:
                    label = "💰 Early Accumulation"
                else:
                    label = "🕓 Neutral/Weak"

                # === Append full numeric + flags ===
                results.append({
                    "Ticker": symbol,
                    "Close": round(float(latest["Close"]), 2),
                    "CMF": latest["CMF"],
                    "RVOL": latest["RVOL"],
                    "ADX": latest["ADX"],
                    "DI_pos": latest["DI_pos"],
                    "DI_neg": latest["DI_neg"],
                    "OBV_Slope": latest["OBV_Slope"],
                    "RS": latest["RS"],
                    "Volatility": latest["Volatility"],
                    "MFI": latest["MFI"],
                    "RSI": latest["RSI"],
                    "RSI_Trend": latest["RSI_Trend"],
                    "VWAP": latest["VWAP"],
                    "ADL_Slope": latest["ADL_Slope"],
                    **indicators,
                    "SmartScore": score,
                    "Label": label
                })
            except Exception:
                failed.append(symbol)
                continue
        progress.progress((batch_num + 1) / total_batches)

    df = pd.DataFrame(results)

    df = df.round(2)

    if not df.empty:
        df = df.sort_values("SmartScore", ascending=False)


    if not df.empty:
        df = df.sort_values("SmartScore", ascending=False)
    return df, failed

# === EXECUTION ===
if refresh or "df_cache" not in st.session_state:
    with st.spinner("⏳ Running Smart Money Screener..."):
        df, failed = run_screener()
        st.session_state.df_cache = df
        st.session_state.failed = failed
else:
    df = st.session_state.df_cache
    failed = st.session_state.failed

# === EXPORT ===
if export and not df.empty:
    towrite = io.BytesIO()
    df.to_excel(towrite, index=False, engine='openpyxl')
    towrite.seek(0)
    st.download_button("💾 Download SmartMoney Results", data=towrite, file_name="smartmoney_results.xlsx")

# === TABLE + COLOR ===
# === COLOR STYLE FOR POSITIVE/NEGATIVE NUMBERS ===
def color_numbers(val):
    if isinstance(val, (int, float)):
        if val > 0:
            return "color: #00FF00; font-weight: bold;"  # green for positive
        elif val < 0:
            return "color: #FF4C4C; font-weight: bold;"  # red for negative
        else:
            return "color: white;"  # neutral for zero
    return ""

# === BOOLEAN STYLE (optional) ===
def color_indicator(val):
    if val is True:
        return "color: green; font-weight: bold;"
    elif val is False:
        return "color: gray;"
    else:
        return ""

# === DISPLAY TABLES ===
if not df.empty:
    st.subheader("📋 Indicator Calculations and Smart Scores")

    # Columns to hide (boolean checks)
    indicator_cols = [
        "CMF_Pos", "RVOL_High", "ADX_Strong", "DI_Trend", "OBV_Up", "RS_Strength",
        "Vol_Contract", "Breakout", "Volume_Pattern", "Force_Pos", "MFI_Strong",
        "RSI_Trend_Pos", "Above_VWAP", "ADL_Up"
    ]

    # Keep only numeric & key info for display
    display_cols = [c for c in df.columns if c not in indicator_cols]

    # Columns that should NOT be colored
    neutral_cols = ["Close", "SmartScore"]

    # Columns to apply color to (all numerics except neutral ones)
    numeric_cols = [c for c in display_cols if c not in neutral_cols and df[c].dtype in ["float64", "int64"]]

    # === Apply styling & rounding ===
    styled_df = (
        df[display_cols]
        .reset_index(drop=True)
        .style
        .format(precision=2)  # ✅ limit to 2 decimals visually
        .applymap(color_numbers, subset=numeric_cols)  # ✅ apply color to numeric cols only
    )

    # Show styled table
    st.dataframe(styled_df, use_container_width=True)


    # === FILTERS ===
    st.subheader("🎯 Filter dan Analisis")
    col1, col2, col3 = st.columns(3)
    min_score = col1.slider("Min SmartScore", 0, 14, 6)
    label_filter = col2.multiselect("Pilih Label", options=df["Label"].unique(), default=df["Label"].unique())
    min_rsi = col3.slider("Minimal RSI", 0, 100, 40)

    filtered = df[
        (df["SmartScore"] >= min_score) &
        (df["Label"].isin(label_filter)) &
        (df["RSI"] >= min_rsi)
    ]

    # Drop boolean check columns
    filtered = filtered.drop(columns=indicator_cols, errors="ignore")

    # Round decimals
    filtered = filtered.round(2)

    # Apply same color styling but keep "Close" and "SmartScore" neutral
    # === Proper styling and rounding ===
    styled_filtered = (
        filtered.reset_index(drop=True)
        .style
        .format(precision=2)  # 👈 ensures all numbers show 2 decimals
        .applymap(color_numbers, subset=[c for c in filtered.columns if c not in neutral_cols])
    )
    st.dataframe(styled_filtered, use_container_width=True)


    # === PLOTS ===
    st.subheader("📊 Visual Analytics")

    tab1, tab2, tab3 = st.tabs(["SmartScore Distribution", "RSI vs MFI Bubble", "Top SmartScore Stocks"])

    with tab1:
        fig_hist = px.histogram(df, x="SmartScore", color="Label", nbins=14,
                                title="Distribusi SmartScore", template="plotly_dark")
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab2:
        fig_scatter = px.scatter(df, x="RSI", y="MFI", size="SmartScore", color="Label",
                                 hover_data=["Ticker", "SmartScore", "CMF", "RVOL", "ADX"],
                                 title="RSI vs MFI (Bubble = SmartScore)", template="plotly_dark")
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab3:
        fig_bar = px.bar(df.head(20), x="Ticker", y="SmartScore", color="Label",
                         title="Top 20 Stocks by SmartScore", template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown(f"✅ Total Saham Teranalisa: **{len(df)}** | ⚠️ Gagal: **{len(failed)}**")

else:
    st.warning("Belum ada data — tekan **🔄 Refresh Data** untuk memulai.")
