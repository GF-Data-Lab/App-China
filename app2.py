import streamlit as st
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI

# -------------------------------------------------------------
# Configuración de archivos estáticos (no se permite subir)
# -------------------------------------------------------------
ARRIVALS_FILE = Path("GESCOM (1).csv")          # Arribos
PRICES_FILE   = Path("precios_mercado.csv")     # Precios de mercado

# Umbral de alerta para diferencia porcentual
ALERT_THRESHOLD = 5  # %

# -------------------------------------------------------------
# 1. Configuración de Azure OpenAI
# -------------------------------------------------------------
endpoint         = 'https://aif-prod-eu2-001.cognitiveservices.azure.com/'
subscription_key = 'AHwZuK6uZewwI4wrNTFrkv3krJfhbilfy4iQnLaRlCpJ0OBx8XFWJQQJ99BFACHYHv6XJ3w3AAAAACOG0KIa'
api_version      = "2024-12-01-preview"
deployment       = "gpt-4o"

client = AzureOpenAI(
    api_version    = api_version,
    azure_endpoint = endpoint,
    api_key        = subscription_key,
)

# ————— Función de resumir un chunk de texto —————
@st.cache_data(show_spinner=False)
def resumen_chunk(text_chunk: str, max_tokens: int = 1024, temperature: float = 0.7) -> str:
    resp = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "Eres un analista de datos que resume tendencias."},
            {"role": "user",   "content": f"Resume en un párrafo lo que muestran estos datos:\n\n{text_chunk}"}
        ],
        model       = deployment,
        max_tokens  = max_tokens,
        temperature = temperature,
        top_p       = 1.0,
    )
    return resp.choices[0].message.content.strip()

# ————— Función map-reduce sobre DataFrame —————
@st.cache_data(show_spinner=False)
def resumir_df(df: pd.DataFrame, chunk_size: int = 200) -> str:
    """
    Espera un df con columnas 'FOLIO', 'a' (precio promedio) y 'b' (CIF).
    Genera mini-resúmenes por trozo y luego un resumen global.
    """
    mini_resumenes = []

    # Map
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        lines = []
        for _, row in chunk.iterrows():
            diff = row["b"] - row["a"]
            lines.append(f"Folio {row['FOLIO']}: CIF={row['b']:.2f}, PrecioProm={row['a']:.2f} (diff={diff:.2f})")
        texto = "\n".join(lines)
        mini = resumen_chunk(texto)
        mini_resumenes.append(mini)

    # Reduce
    combinado = "\n\n".join(mini_resumenes)
    prompt_final = (
        "A partir de estos resúmenes parciales, genera un informe único "
        "que sintetice las principales tendencias entre CIF y precio promedio:\n\n"
        + combinado
    )
    return resumen_chunk(prompt_final, max_tokens=2048, temperature=0.5)

# ————— Carga de datos —————
@st.cache_data(show_spinner=True)
def load_data():
    if not ARRIVALS_FILE.exists() or not PRICES_FILE.exists():
        st.error(
            "No se encontraron los archivos requeridos:\n\n"
            f"• {ARRIVALS_FILE.resolve()}\n"
            f"• {PRICES_FILE.resolve()}"
        )
        st.stop()

    df_arrivals = pd.read_csv(ARRIVALS_FILE, parse_dates=["ArrivalDate.Date"])
    df_prices   = pd.read_csv(PRICES_FILE,   parse_dates=["FECHA"])
    return df_arrivals, df_prices


def highlight_large_diff(val):
    if pd.isna(val):
        return ""
    return "background-color: red" if abs(val) > ALERT_THRESHOLD else ""


def get_daily_differences(row: pd.Series, precios_df: pd.DataFrame) -> pd.DataFrame:
    day     = row["ArrivalDate.Date"]
    zona    = row["ZONA"]
    specie  = row["ESPECIE"]
    size    = row["CALIBRE"]
    codigo  = row["CODIGO_PM"]
    brand   = row["ETIQUETA"]
    cif     = row["CIF"]

    start = day
    end   = day + pd.Timedelta(days=7)
    fechas_rango = pd.date_range(start=start, end=end, freq="D")
    daily_rows = []

    for fecha in fechas_rango:
        mask = (
            (precios_df["FECHA"] == fecha) &
            (precios_df["ZONA"]      == zona) &
            (precios_df["ESPECIE"]   == specie) &
            (precios_df["CALIBRE"]   == size) &
            (precios_df["CODIGO_PM"] == codigo) &
            (precios_df["ETIQUETA"]  == brand)
        )
        df_dia = precios_df.loc[mask]
        if not df_dia.empty:
            amount_avg = df_dia["AverageAmount"].mean()
            diff_abs   = cif - amount_avg
            diff_pct   = (diff_abs / amount_avg * 100) if amount_avg else None
        else:
            amount_avg = diff_abs = diff_pct = None

        daily_rows.append({
            "FechaPrecio": fecha.date(),
            "CIF": cif,
            "AmountAvg": amount_avg,
            "Diff_Abs": diff_abs,
            "Diff_Pct": diff_pct,
        })

    return pd.DataFrame(daily_rows)


def enviar_alertas_por_folio(df_filtered: pd.DataFrame, precios_df: pd.DataFrame):
    """Recorre cada folio del df filtrado y muestra/print las filas en alerta."""
    any_alerts = False
    for folio in sorted(df_filtered["FOLIO"].unique()):
        df_folio = df_filtered[df_filtered["FOLIO"] == folio]
        daily_tables = [get_daily_differences(row, precios_df) for _, row in df_folio.iterrows()]
        df_daily = pd.concat(daily_tables, ignore_index=True)

        # Convertimos Diff_Pct a numérico, forzando NaN para valores no convertibles (None)
        diff_pct_numeric = pd.to_numeric(df_daily["Diff_Pct"], errors="coerce")
        df_alerts = df_daily[diff_pct_numeric.abs() > ALERT_THRESHOLD]

        if not df_alerts.empty:
            any_alerts = True
            st.warning(f"🚨 Alertas para FOLIO {folio}")
            st.dataframe(
                df_alerts.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
                use_container_width=True,
            )
            # También imprime en la consola (útil al correr `streamlit run`)
            print(f"--- ALERTAS FOLIO {folio} ---")
            print(df_alerts.to_string(index=False))

    if not any_alerts:
        st.success("✅ No se encontraron alertas para ningún folio.")

# -------------------------------------------------------------
# 3. App principal
# -------------------------------------------------------------

def main():
    st.set_page_config(page_title="Comparación CIF vs Precios Promedio", layout="wide")
    st.title("Comparación CIF vs Precios Promedio")

    st.sidebar.success("Usando archivos estáticos:")
    st.sidebar.code(f"{ARRIVALS_FILE}\n{PRICES_FILE}")

    # 1. Carga data
    df, precios_dia_ff = load_data()

    # 2. Selección de fecha
    fechas_disponibles = sorted(df["ArrivalDate.Date"].dt.date.unique())
    if not fechas_disponibles:
        st.error("No hay fechas válidas en GESCOM.csv.")
        st.stop()

    fecha_sel = st.selectbox("Selecciona la fecha de llegada", fechas_disponibles)
    fecha_sel_dt = pd.to_datetime(fecha_sel)
    df_filtered = df[df["ArrivalDate.Date"] == fecha_sel_dt]

    st.write(f"### Registros para {fecha_sel}: {len(df_filtered)} fila(s)")
    st.dataframe(df_filtered, use_container_width=True)
    if df_filtered.empty:
        st.warning("No se encontraron arribos para esa fecha.")
        st.stop()

    # 2.1 Botón de alertas globales por folio
    if st.button("🚨 Enviar alerta por cada folio"):
        with st.spinner("Buscando alertas…"):
            enviar_alertas_por_folio(df_filtered, precios_dia_ff)

    # 3. Tabla de diferencia diaria por FOLIO específico
    folios = sorted(df_filtered["FOLIO"].unique())
    folio_sel = st.selectbox("Selecciona el FOLIO", folios)
    if folio_sel:
        df_folio = df_filtered[df_filtered["FOLIO"] == folio_sel]
        st.subheader(f"Diferencia diaria para FOLIO {folio_sel}")
        daily_tables = [get_daily_differences(row, precios_dia_ff) for _, row in df_folio.iterrows()]
        df_daily = pd.concat(daily_tables, ignore_index=True)
        st.dataframe(df_daily.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
                     use_container_width=True)
        csv_daily = df_daily.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Descargar CSV diferencia diaria",
                           data=csv_daily,
                           file_name=f"dif_diaria_folio_{folio_sel}.csv",
                           mime="text/csv")

    # 4. Resumen de coincidencias y mismatches — TODOS los folios
    resumen = []
    mismatches = []
    folios = df_filtered["FOLIO"].unique()
    progress = st.progress(0, text="Procesando folios… (resumen)")
    for i, folio in enumerate(folios, start=1):
        df_current = df_filtered[df_filtered["FOLIO"] == folio]
        for _, row in df_current.iterrows():
            # … mismo código de filtrado y llena resumen[] y mismatches[] …
            # (mantén tu lógica original aquí para coincidencias/mismatches)
            pass
        progress.progress(i/len(folios), text=f"Procesado folio {folio} ({i}/{len(folios)})")
    progress.empty()

    df_resumen = None
    if resumen:
        df_resumen = pd.DataFrame(resumen).drop_duplicates()
        st.subheader("Resumen de diferencias (coincidencias) — TODOS los folios")
        st.dataframe(df_resumen.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
                     use_container_width=True)
        csv = df_resumen.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Descargar CSV con coincidencias",
                           data=csv,
                           file_name="resumen_comparacion.csv",
                           mime="text/csv")

    if mismatches:
        st.subheader("Folios sin coincidencias y campos que no hacen match")
        df_mismatch = pd.DataFrame(mismatches)
        st.dataframe(df_mismatch, use_container_width=True)
        csv_bad = df_mismatch.to_csv(index=False).encode("utf-8")
        st.download_button("💾 Descargar CSV de faltantes",
                           data=csv_bad,
                           file_name="faltantes_campos.csv",
                           mime="text/csv")
    elif not resumen:
        st.info("No se encontraron coincidencias ni faltantes que mostrar.")

    # ← Generar resumen LLM sólo al hacer click
    if df_resumen is not None:
        if st.button("📝 Generar resumen LLM"):
            with st.spinner("Generando resumen LLM…"):
                # Renombramos para que resumir_df calcule diff = b - a
                df_for_summary = df_resumen.rename(columns={"AmountAvg": "a", "CIF": "b"})
                summary_text = resumir_df(df_for_summary, chunk_size=200)
            st.subheader(f"🗓 Resumen LLM para la fecha {fecha_sel}")
            st.write(summary_text)


if __name__ == "__main__":
    main()

