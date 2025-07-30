# ──────────────────────────────────────────────────────────────
# NEW TAB: "Detección de anomalías GESCOM"  (filtro BROKER + col. FOLIO)
# ----------------------------------------------------------------
# Versión actualizada: además del filtro BROKER ya incorporado, la
# tabla/CSV final ahora muestra la columna FOLIO asociada a cada fila
# anómala para que puedas identificar rápidamente el registro.
# ----------------------------------------------------------------
import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# Re‑use highlight util from main script if it exists
try:
    highlight_large_diff  # noqa: F401
except NameError:
    def highlight_large_diff(val):
        if pd.isna(val):
            return ""
        return "background-color: red" if abs(val) > 5 else ""


def anomalias_gescom_page(df_arrivals: pd.DataFrame, key_prefix: str = "ges"):
    """Interactive anomaly detection for the GESCOM arrivals dataframe.

    Parameters
    ----------
    df_arrivals : pd.DataFrame
        DataFrame cargado a partir del CSV de GESCOM (arribos).
    key_prefix : str, optional
        Prefijo para los keys de los widgets Streamlit, by default "ges".
    """

    st.header("🔎 Detección de anomalías en GESCOM (CIF)")

    # ------------------------------------------------------------------
    # 1️⃣ Columnas numéricas & de agrupamiento
    # ------------------------------------------------------------------
    NUM_COLS   = ["CIF"]  # métrica numérica principal
    GROUP_COLS = [
        "ESPECIE", "ZONA", "VARIEDAD COMERCIAL","CALIBRE", "ETIQUETA", "ENVASE", "BROKER"
    ]
    EXTRA_COLS = ["FOLIO"]  # columnas que queremos conservar/mostrar

    # Validación de columnas obligatorias
    df = df_arrivals.copy()
    for col in NUM_COLS + GROUP_COLS + EXTRA_COLS:
        if col not in df.columns:
            st.error(f"La columna '{col}' no existe en el dataframe de GESCOM.")
            st.stop()

    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df["is_anomaly_if"] = False
    df["if_score"]      = None

    # ------------------------------------------------------------------
    # 2️⃣ Filtros dinámicos (incluye BROKER)
    # ------------------------------------------------------------------
    cols = st.columns(7)
    fruit   = cols[0].selectbox("ESPECIE",   ["(Todas)"] + sorted(df["ESPECIE"].dropna().unique()),   key=f"{key_prefix}_esp")
    region  = cols[1].selectbox("ZONA",      ["(Todas)"] + sorted(df["ZONA"].dropna().unique()),      key=f"{key_prefix}_zone")
    size    = cols[2].selectbox("CALIBRE",   ["(Todas)"] + sorted(df["CALIBRE"].dropna().unique()),   key=f"{key_prefix}_cal")
    variety    = cols[3].selectbox("VARIEDAD COMERCIAL",   ["(Todas)"] + sorted(df["VARIEDAD COMERCIAL"].dropna().unique()),   key=f"{key_prefix}_VARIEDAD COMERCIAL")
    brand   = cols[4].selectbox("ETIQUETA",  ["(Todas)"] + sorted(df["ETIQUETA"].dropna().unique()),  key=f"{key_prefix}_brand")
    codigo  = cols[5].selectbox("ENVASE", ["(Todas)"] + sorted(df["ENVASE"].dropna().unique()), key=f"{key_prefix}_code")
    broker  = cols[6].selectbox("BROKER",    ["(Todas)"] + sorted(df["BROKER"].dropna().unique()),    key=f"{key_prefix}_broker")

    mask = pd.Series(True, index=df.index)
    if fruit  != "(Todas)": mask &= df["ESPECIE"]   == fruit
    if region != "(Todas)": mask &= df["ZONA"]      == region
    if size   != "(Todas)": mask &= df["CALIBRE"]   == size
    if brand  != "(Todas)": mask &= df["ETIQUETA"]  == brand
    if codigo != "(Todas)": mask &= df["ENVASE"] == codigo
    if broker != "(Todas)": mask &= df["BROKER"]    == broker
    if variety != "(Todas)": mask &= df["VARIEDAD COMERCIAL"]    == variety

    filtered_df = df.loc[mask].copy()
    st.write("Filas después del filtro:", len(filtered_df))

    # ------------------------------------------------------------------
    # 3️⃣ Parámetros del modelo IsolationForest
    # ------------------------------------------------------------------
    with st.expander("⚙️ Parámetros del modelo", expanded=False):
        min_rows     = st.number_input("Mínimo de filas por grupo", 5, 200, 15, key=f"{key_prefix}_minrows")
        contamination= st.slider("Contaminación esperada (%)", 1, 20, 2, key=f"{key_prefix}_cont")/100
        n_estimators = st.number_input("N.º árboles", 50, 400, 200, step=50, key=f"{key_prefix}_nest")

    IF_PARAMS = dict(n_estimators=int(n_estimators), contamination=contamination, random_state=42)

    # Helper ------------------------------------------------------------
    def marcar_anomalias(grp: pd.DataFrame) -> pd.DataFrame:
        if grp[NUM_COLS].notna().all(axis=1).sum() < min_rows:
            return grp  # grupo demasiado pequeño
        mask_ok = grp[NUM_COLS].notna().all(axis=1)
        X = grp.loc[mask_ok, NUM_COLS]
        iso = IsolationForest(**IF_PARAMS).fit(X)
        preds  = iso.predict(X)          # -1 → anomalía
        scores = iso.decision_function(X)
        grp.loc[mask_ok, "is_anomaly_if"] = preds == -1
        grp.loc[mask_ok, "if_score"]      = scores
        return grp

    # ------------------------------------------------------------------
    # 4️⃣ Entrenamiento y visualización
    # ------------------------------------------------------------------
    if st.button("Entrenar modelo", key=f"{key_prefix}_train"):
        result_df = (
            filtered_df
            .groupby(GROUP_COLS, group_keys=False)
            .apply(marcar_anomalias)
            .reset_index(drop=True)
        )
        anomalies = result_df[result_df["is_anomaly_if"]]

        st.subheader("Resumen")
        st.write(
            f"Total: {len(result_df):,} | Anómalas: {len(anomalies):,} "
            f"({len(anomalies)/len(result_df):.2%})"
        )

        if not anomalies.empty:
            cols_to_show = EXTRA_COLS + GROUP_COLS + NUM_COLS + ["if_score"]
            with st.expander("Ver anomalías"):
                st.dataframe(
                    anomalies[cols_to_show]
                    .style.applymap(highlight_large_diff, subset=NUM_COLS),
                    use_container_width=True,
                )

            st.download_button(
                label="Descargar anomalías (CSV)",
                data=anomalies[cols_to_show].to_csv(index=False).encode(),
                file_name="anomalies_gescom.csv",
                mime="text/csv",
                key=f"{key_prefix}_dl_csv",
            )
        else:
            st.success("✅ No se detectaron anomalías con los parámetros actuales.")
