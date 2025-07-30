import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
from pathlib import Path
# ------------------------------------------------------------------
# Utilidad para colorear numéricos fuera de rango (>5 % por defecto)
# ------------------------------------------------------------------
try:
    highlight_large_diff  # noqa: F401
except NameError:
    def highlight_large_diff(val):
        if pd.isna(val):
            return ""
        return "background-color: red" if abs(val) > 5 else ""




def highlight_large_diff(val):
    if pd.isna(val):
        return ""
    return "background-color: red" if abs(val) > 5 else ""

# ────────────────────────────────────────────────────────────────
def anomalias_liq_page(key_prefix: str = "ff"):           # ← prefijo por defecto CAMBIADO
    """Detección de anomalías en precios_liq_ff.csv (Forever Fresh)"""

    st.header("🔎 Detección de anomalías — Liquidaciones FF")

    DEFAULT_FILE = "precios_liq_ff.csv"
    uploaded = st.file_uploader(
        "Subir precios_liq_ff.csv",
        type=["csv"],
        key=f"{key_prefix}_upl"
    )

    # ---------- 1. Cargar datos ----------
    @st.cache_data(show_spinner=False)
    def _load(path_or_buf):
        return pd.read_csv(path_or_buf, parse_dates=["sale date"])

    if uploaded is not None:
        df = _load(uploaded)
        st.success(f"Cargado **{uploaded.name}**")
    elif Path(DEFAULT_FILE).is_file():
        df = _load(DEFAULT_FILE)
        st.info(f"Se cargó automáticamente **{DEFAULT_FILE}**")
    else:
        st.error("No se encontró precios_liq_ff.csv y no subiste ninguno.")
        st.stop()

    NUM_COLS   = ["Sales price in RMB"]
    GROUP_COLS = ["type_Children", "Calibre", "brand", "custom"]

    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df["is_anomaly_if"] = False
    df["if_score"]      = None

    # ---------- 2. Filtros (con formulario) ----------
    with st.form(f"{key_prefix}_form_filtros"):
        cols = st.columns(4)
        variety = cols[0].selectbox("VAR COM",
            ["(Todas)"] + sorted(df["type_Children"].dropna().unique()))
        size    = cols[1].selectbox("CALIBRE",
            ["(Todas)"] + sorted(df["Calibre"].dropna().unique()))
        brand   = cols[2].selectbox("ETIQUETA",
            ["(Todas)"] + sorted(df["brand"].dropna().unique()))
        cust    = cols[3].selectbox("Customers",
            ["(Todas)"] + sorted(df["custom"].dropna().unique()))

        filtros_ok = st.form_submit_button("Guardar cambios")

    if not filtros_ok:
        st.info("Ajusta los filtros y pulsa **Guardar cambios** para aplicarlos.")
        st.stop()

    m = pd.Series(True, index=df.index)
    if variety != "(Todas)": m &= df["type_Children"] == variety
    if size    != "(Todas)": m &= df["Calibre"]       == size
    if brand   != "(Todas)": m &= df["brand"]         == brand
    if cust    != "(Todas)": m &= df["custom"]        == cust

    df_f = df.loc[m].copy()
    st.success(f"🔎 Filtrado aplicado: {len(df_f)} filas")

    # ---------- 3. Parámetros modelo ----------
    with st.expander("⚙️ Parámetros del modelo"):
        min_rows     = st.number_input("Mínimo de filas por grupo", 5, 200, 15,
                                       key=f"{key_prefix}_minr")
        contamination= st.slider("Contaminación esperada (%)", 1, 20, 2,
                                       key=f"{key_prefix}_cont")/100
        n_estimators = st.number_input("N.º árboles", 50, 400, 200, step=50,
                                       key=f"{key_prefix}_nest")
    IF_PARAMS = dict(n_estimators=int(n_estimators),
                     contamination=contamination,
                     random_state=42)

    def marcar(grp):
        if grp[NUM_COLS].notna().all(axis=1).sum() < min_rows:
            return grp
        X = grp.loc[grp[NUM_COLS].notna().all(axis=1), NUM_COLS]
        iso = IsolationForest(**IF_PARAMS).fit(X)
        preds  = iso.predict(X)
        scores = iso.decision_function(X)
        grp.loc[X.index, "is_anomaly_if"] = preds == -1
        grp.loc[X.index, "if_score"]      = scores
        return grp

    if st.button("Entrenar modelo", key=f"{key_prefix}_run"):
        res = df_f.groupby(GROUP_COLS, group_keys=False).apply(marcar).reset_index(drop=True)
        anomalies = res[res["is_anomaly_if"]]

        st.subheader("Resumen")
        st.write(f"Total: {len(res):,} | Anómalas: {len(anomalies):,} "
                 f"({len(anomalies)/len(res):.2%})")

        if not anomalies.empty:
            cols_show = GROUP_COLS + NUM_COLS + ["if_score"]
            st.dataframe(
                anomalies[cols_show].style.applymap(highlight_large_diff, subset=NUM_COLS),
                use_container_width=True,
            )
            st.download_button("Descargar CSV",
                anomalies[cols_show].to_csv(index=False).encode(),
                file_name="anomalies_precios_liq.csv",
                mime="text/csv",
                key=f"{key_prefix}_csv",
            )
        else:
            st.success("✅ No se detectaron anomalías.")
