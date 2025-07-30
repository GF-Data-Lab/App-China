import streamlit as st
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from models.alert_sender import AlertSender

# -------------------------------------------------------------
# Configuración de archivos estáticos por defecto (pestaña «Comparación actual»)
# -------------------------------------------------------------
ARRIVALS_FILE = Path("GESCOM (1).csv")          # Arribos
PRICES_FILE   = Path("precios_mercado.csv")     # Precios de mercado

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

# Umbral para disparar alertas (±% sobre la diferencia porcentual)
ALERT_THRESHOLD = 5

# -------------------------------------------------------------
# 2. Utilidades generales
# -------------------------------------------------------------

def _dedup_arrivals(df: pd.DataFrame) -> pd.DataFrame:
    """Quita filas duplicadas dentro de un mismo FOLIO para evitar tablas repetidas."""
    return df.drop_duplicates(subset=[
        "ZONA", "ESPECIE", "CALIBRE", "CODIGO_PM", "ETIQUETA", "CIF"
    ])

# -------------------------------------------------------------
# 3. Helpers LLM (resumen) – se mantienen como en el script original
# -------------------------------------------------------------

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

@st.cache_data(show_spinner=False)
def resumir_df(df: pd.DataFrame, chunk_size: int = 200) -> str:
    """Map‑reduce para resumir diferencias a nivel global."""
    mini_resumenes = []
    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start : start + chunk_size]
        lines = []
        for _, row in chunk.iterrows():
            diff = row["b"] - row["a"]
            lines.append(f"Folio {row['FOLIO']}: CIF={row['b']:.2f}, PrecioProm={row['a']:.2f} (diff={diff:.2f})")
        mini_resumenes.append(resumen_chunk("\n".join(lines)))

    prompt_final = (
        "A partir de estos resúmenes parciales, genera un informe único que sintetice las principales tendencias entre CIF y precio promedio:\n\n"
        + "\n\n".join(mini_resumenes)
    )
    return resumen_chunk(prompt_final, max_tokens=2048, temperature=0.5)

# -------------------------------------------------------------
# 4. Cálculo de diferencias diarias y alertas – sin cambios de lógica
# -------------------------------------------------------------

def highlight_large_diff(val):
    if pd.isna(val):
        return ""
    return "background-color: red" if abs(val) > ALERT_THRESHOLD else ""


def get_daily_differences(row: pd.Series, precios_df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve un dataframe con la diferencia diaria CIF vs mercado para los 7 días siguientes."""
    day     = row["ArrivalDate.Date"]
    zona    = row["ZONA"]
    specie  = row["ESPECIE"]
    size    = row["CALIBRE"]
    codigo  = row["CODIGO_PM"]
    brand   = row["ETIQUETA"]
    cif     = row["CIF"]

    fechas_rango = pd.date_range(start=day, end=day + pd.Timedelta(days=7), freq="D")
    registros = []
    for fecha in fechas_rango:
        mask = (
            (precios_df["FECHA"]      == fecha) &
            (precios_df["ZONA"]       == zona)  &
            (precios_df["ESPECIE"]    == specie) &
            (precios_df["CALIBRE"]    == size)   &
            (precios_df["CODIGO_PM"]  == codigo) &
            (precios_df["ETIQUETA"]   == brand)
        )
        df_dia = precios_df.loc[mask]
        if not df_dia.empty:
            amount_avg = df_dia["AverageAmount"].mean()
            diff_abs   = cif - amount_avg
            diff_pct   = (diff_abs / amount_avg * 100) if amount_avg else None
        else:
            amount_avg = diff_abs = diff_pct = None

        registros.append({
            "FechaPrecio": fecha.date(),
            "CIF": cif,
            "AmountAvg": amount_avg,
            "Diff_Abs": diff_abs,
            "Diff_Pct": diff_pct,
        })

    return pd.DataFrame(registros)


def _evalua_alertas(df_daily: pd.DataFrame) -> pd.DataFrame:
    diff_pct_num = pd.to_numeric(df_daily["Diff_Pct"], errors="coerce")
    return df_daily[diff_pct_num.abs() > ALERT_THRESHOLD]


def enviar_alertas_por_folio(df_filtered: pd.DataFrame, precios_df: pd.DataFrame):
    any_alerts = False
    dict_alerts = {}
    for folio in sorted(df_filtered["FOLIO"].unique()):
        df_folio_raw     = df_filtered[df_filtered["FOLIO"] == folio]
        df_folio_unique  = _dedup_arrivals(df_folio_raw)

        daily_tables = [get_daily_differences(row, precios_df) for _, row in df_folio_unique.iterrows()]
        df_daily     = pd.concat(daily_tables, ignore_index=True)
        df_daily     = df_daily.drop_duplicates(subset=["FechaPrecio", "CIF", "AmountAvg"])

        df_alerts = _evalua_alertas(df_daily)
        if not df_alerts.empty:
            any_alerts = True
            st.warning(f"🚨 Alertas para FOLIO {folio}")
            st.dataframe(
                df_alerts.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
                use_container_width=True,
            )
            print(f"--- ALERTAS FOLIO {folio} ---")
            print(df_alerts.to_string(index=False))
            dict_alerts[folio] = df_alerts

    if not any_alerts:
        st.success("✅ No se encontraron alertas para ningún folio.")
    else:
        sender = AlertSender(dict_alerts, 'manuel.munozm@garcesfruit.com')
        sender.send_alert()

# -------------------------------------------------------------
# 5. Carga de datos (estáticos o subidos)
# -------------------------------------------------------------

def load_data_from_path(arrivals_path: Path, prices_path: Path):
    """Lee los CSV desde rutas locales (pestaña Comparación actual)."""
    if not arrivals_path.exists() or not prices_path.exists():
        st.error(
            "No se encontraron los archivos requeridos:\n\n" +
            f"• {arrivals_path.resolve()}\n" +
            f"• {prices_path.resolve()}"
        )
        st.stop()
    df_arrivals = pd.read_csv(arrivals_path, parse_dates=["ArrivalDate.Date"])
    df_prices   = pd.read_csv(prices_path,   parse_dates=["FECHA"])
    return df_arrivals, df_prices


def load_data_from_upload(arrivals_upload, prices_upload):
    """Convierte los archivos subidos (BytesIO) en DataFrames."""
    df_arrivals = pd.read_csv(arrivals_upload, parse_dates=["ArrivalDate.Date"])
    df_prices   = pd.read_csv(prices_upload,   parse_dates=["FECHA"])
    return df_arrivals, df_prices

# -------------------------------------------------------------
# 6. Lógica de comparación encapsulada para reuso en ambas pestañas
# -------------------------------------------------------------

def comparacion_page(df_arrivals: pd.DataFrame, precios_df: pd.DataFrame, key_prefix: str = ""):
    """Renderiza toda la interfaz de comparación (reutilizable)."""
    # 1. Selección de fecha
    fechas_disponibles = sorted(df_arrivals["ArrivalDate.Date"].dt.date.unique())
    if not fechas_disponibles:
        st.error("No hay fechas válidas en el CSV de arribos.")
        return

    fecha_sel = st.selectbox(
        "Selecciona la fecha de llegada",
        fechas_disponibles,
        key=f"{key_prefix}_fecha_select"
    )
    fecha_sel_dt = pd.to_datetime(fecha_sel)
    df_filtered = df_arrivals[df_arrivals["ArrivalDate.Date"] == fecha_sel_dt]

    st.write(f"### Registros para {fecha_sel}: {len(df_filtered)} fila(s)")
    st.dataframe(df_filtered, use_container_width=True)
    if df_filtered.empty:
        st.warning("No se encontraron arribos para esa fecha.")
        return

    # Botón para enviar alertas por folio
    if st.button("🚨 Enviar alerta por cada folio", key=f"{key_prefix}_btn_alertas"):
        enviar_alertas_por_folio(df_filtered, precios_df)

    # 2. Tabla de diferencia diaria por FOLIO (selección individual)
    folios = sorted(df_filtered["FOLIO"].unique())
    folio_sel = st.selectbox(
        "Selecciona el FOLIO",
        folios,
        key=f"{key_prefix}_folio_select"
    )
    if folio_sel:
        df_folio = df_filtered[df_filtered["FOLIO"] == folio_sel]
        df_folio_unique = _dedup_arrivals(df_folio)
        st.subheader(f"Diferencia diaria para FOLIO {folio_sel}")
        daily_tables = [get_daily_differences(row, precios_df) for _, row in df_folio_unique.iterrows()]
        df_daily = pd.concat(daily_tables, ignore_index=True)
        df_daily = df_daily.drop_duplicates(subset=["FechaPrecio", "CIF", "AmountAvg"])
        st.dataframe(
            df_daily.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
            use_container_width=True,
        )
        csv_daily = df_daily.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar CSV diferencia diaria",
            data=csv_daily,
            file_name=f"dif_diaria_folio_{folio_sel}.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_daily",
        )

    # 3. Resumen de coincidencias y mismatches — TODOS los folios
    resumen = []
    mismatches = []
    folios = df_filtered["FOLIO"].unique()
    progress = st.progress(0, text="Procesando folios… (resumen)")
    for i, folio in enumerate(folios, start=1):
        df_current = df_filtered[df_filtered["FOLIO"] == folio]
        # → Agrega tu lógica original para coincidencias/mismatches aquí
        # (esta parte no se modifica en esta versión)
        pass
        progress.progress(i/len(folios), text=f"Procesado folio {folio} ({i}/{len(folios)})")
    progress.empty()

    df_resumen = None
    if resumen:
        df_resumen = pd.DataFrame(resumen).drop_duplicates()
        st.subheader("Resumen de diferencias (coincidencias) — TODOS los folios")
        st.dataframe(
            df_resumen.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
            use_container_width=True,
        )
        csv = df_resumen.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar CSV con coincidencias",
            data=csv,
            file_name="resumen_comparacion.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_resumen",
        )

    if mismatches:
        st.subheader("Folios sin coincidencias y campos que no hacen match")
        df_mismatch = pd.DataFrame(mismatches)
        st.dataframe(df_mismatch, use_container_width=True)
        csv_bad = df_mismatch.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar CSV de faltantes",
            data=csv_bad,
            file_name="faltantes_campos.csv",
            mime="text/csv",
            key=f"{key_prefix}_download_faltantes",
        )
    elif not resumen:
        st.info("No se encontraron coincidencias ni faltantes que mostrar.")

    # 4. Generar resumen LLM
    if df_resumen is not None:
        if st.button("📝 Generar resumen LLM", key=f"{key_prefix}_btn_resumen_llm"):
            with st.spinner("Generando resumen LLM…"):
                df_for_summary = df_resumen.rename(columns={"AmountAvg": "a", "CIF": "b"})
                summary_text = resumir_df(df_for_summary, chunk_size=200)
            st.subheader(f"🗓 Resumen LLM para la fecha {fecha_sel}")
            st.write(summary_text)

# -------------------------------------------------------------
# 7. App principal con interfaz multitab
# -------------------------------------------------------------
# -------------------------------------------------------------
# 4-bis. Helpers específicos para la lógica de liquidaciones
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def build_customer2zona(precios_df: pd.DataFrame) -> dict[str, str]:
    """Devuelve dicc. {customer -> zona} construido UNA sola vez."""
    return (
        precios_df[['Customers', 'ZONA']]
        .drop_duplicates()
        .set_index('Customers')['ZONA']
        .to_dict()
    )


# -------------------------------------------------------------
# Configuración: nombre de la columna que contiene el precio de la liquidación
#                en tu CSV de liquidaciones.
# Ajusta este valor si tu archivo usa otro encabezado.
# -------------------------------------------------------------
LIQ_PRICE_COL = "Sales price in RMB"  # ⇦ cámbialo según tu CSV

# -------------------------------------------------------------
# Reutilizamos el mismo umbral y función de resaltado del script original
# -------------------------------------------------------------
ALERT_THRESHOLD = 5  # ±%

def highlight_large_diff(val: float):
    if pd.isna(val):
        return ""
    return "background-color: red" if abs(val) > ALERT_THRESHOLD else ""

# ----------------------------------------------------------------------------
# Función principal (reemplaza la versión antigua de comparacion_liq_page)
# ----------------------------------------------------------------------------

def comparacion_liq_page(df_liq: pd.DataFrame, precios_dia_ff: pd.DataFrame):
    """Pestaña «Nueva comparación» con cálculo de diferencias % vs mercado."""

    # ----------------------- 1. Filtro por sale date -----------------------
    sale_dates = sorted(pd.to_datetime(df_liq["sale date"]).dt.date.unique())
    if not sale_dates:
        st.error("El CSV de liquidaciones no contiene fechas válidas en 'sale date'.")
        return

    sale_sel = st.selectbox("Selecciona la *sale date*", sale_dates)
    fecha_dt = pd.to_datetime(sale_sel)
    df_liq_sel = df_liq[df_liq["sale date"] == fecha_dt]

    st.write(f"### Registros para {sale_sel}: {len(df_liq_sel)} fila(s)")
    st.dataframe(df_liq_sel, use_container_width=True)

    if df_liq_sel.empty:
        st.warning("No hay registros para esa fecha.")
        return

    # -------------------- 2. Bucle por terna + comparación -----------------
    customer2zona = (
        precios_dia_ff[["Customers", "ZONA"]].drop_duplicates().set_index("Customers")["ZONA"].to_dict()
    )
    cols_key = ["Invoice number", "Cabinet number"]  # sale date ya está filtrada

    resultados = []           # acumularemos aquí filas con resultado de la comparación
    progress = st.progress(0, text="Procesando ternas…")
    total = len(df_liq_sel.groupby(cols_key))

    for i, (invoice, cabinet) in enumerate(df_liq_sel.groupby(cols_key).groups.keys(), 1):
        st.subheader(f"📄 Invoice {invoice} – Cabinet {cabinet}")
        grupo = df_liq_sel[
            (df_liq_sel["Invoice number"] == invoice) & (df_liq_sel["Cabinet number"] == cabinet)
        ]

        for _, row in grupo.iterrows():
            day      = pd.to_datetime(row["sale date"])
            customer = row["custom"]
            variety  = row["type_Children"].strip().lower()
            brand    = row["brand"].strip().lower()
            liq_price = row[LIQ_PRICE_COL]
            calibre = row['Calibre'].strip().lower()
            # Intentos: mismo día, día-1, … día-6
            df_rango = pd.DataFrame()
            for offset in range(0, 7):
                search_day = day - pd.Timedelta(days=offset)

                # ---- A) filtro por customer ----
                mask_cust = (
                    (precios_dia_ff["FECHA"] == search_day)
                    & (precios_dia_ff["ETIQUETA"].str.lower() == brand)
                    & (precios_dia_ff["VARIEDAD COMERCIAL"].str.lower() == variety)
                    & (precios_dia_ff["Customers"] == customer)
                    & (precios_dia_ff['CALIBRE'] == calibre)
                )
                df_rango = precios_dia_ff.loc[mask_cust]
                if not df_rango.empty:
                    break

                # ---- B) filtro por zona ----
                zona = customer2zona.get(customer)
                if zona is not None:
                    mask_zona = (
                        (precios_dia_ff["FECHA"] == search_day)
                        & (precios_dia_ff["ETIQUETA"].str.lower() == brand)
                        & (precios_dia_ff["VARIEDAD COMERCIAL"].str.lower() == variety)
                        & (precios_dia_ff["ZONA"] == zona) 
                        & (precios_dia_ff['CALIBRE'] == calibre)
                    )
                    df_rango = precios_dia_ff.loc[mask_zona]
                    if not df_rango.empty:
                        break  # éxito

            # ---------- Mostrar / acumular resultado ----------
            if df_rango.empty:
                st.warning(
                    f"⚠️ Sin coincidencias para **{customer} / {brand}** "
                    f"entre {day.date()} y {(day - pd.Timedelta(days=6)).date()}"
                )
            else:
                # Calcula promedio mercado y diferencias
                amount_avg = df_rango["AverageAmount"].mean()
                diff_abs   = liq_price - amount_avg
                diff_pct   = (diff_abs / amount_avg * 100) if amount_avg else None

                fila_result = {
                    "Invoice": invoice,
                    "Cabinet": cabinet,
                    "SaleDate": day.date(),
                    "Customer": customer,
                    "Variety": variety,
                    "Brand": brand,
                    "LiquidationPrice": liq_price,
                    "AmountAvg": amount_avg,
                    "Diff_Abs": diff_abs,
                    "Diff_Pct": diff_pct,
                }
                resultados.append(fila_result)

        progress.progress(i / total, text=f"Procesado {i}/{total}")

    progress.empty()

    # ------------------------ 3. Tabla + CSV global -----------------------
    if resultados:
        df_result = pd.DataFrame(resultados)
        st.subheader("Comparación Liquidación vs Mercado — TODAS las coincidencias")
        st.dataframe(
            df_result.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
            use_container_width=True,
        )

        csv_all = df_result.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Descargar resultados (CSV)",
            data=csv_all,
            file_name=f"comparacion_liq_{sale_sel}.csv",
            mime="text/csv",
        )

        # Muestra alertas en pantalla si hay diferencias grandes (> ALERT_THRESHOLD)
        df_alerts = df_result[pd.to_numeric(df_result["Diff_Pct"], errors="coerce").abs() > ALERT_THRESHOLD]
        if not df_alerts.empty:
            st.warning("🚨 Se encontraron diferencias mayores al umbral:")
            st.dataframe(
                df_alerts.style.applymap(highlight_large_diff, subset=["Diff_Pct"]),
                use_container_width=True,
            )
    else:
        st.info("No se encontraron coincidencias comparables para esa fecha.")


def main():
    st.set_page_config(page_title="Comparación CIF vs Precios Promedio", layout="wide")
    st.title("Comparador CIF vs Mercado — Modo Multitab")

    tab_actual, tab_nueva = st.tabs(["Comparación actual", "Nueva comparación"])

    # ------------------------------------------
    # Pestaña 1: Comparación con archivos por defecto
    # ------------------------------------------
    with tab_actual:
        st.sidebar.success("Usando archivos estáticos:")
        st.sidebar.code(f"{ARRIVALS_FILE}\n{PRICES_FILE}")

        df_arrivals, df_prices = load_data_from_path(ARRIVALS_FILE, PRICES_FILE)
        comparacion_page(df_arrivals, df_prices, key_prefix="orig")

    # ------------------------------------------
    # Pestaña 2: Comparación con archivos subidos por el usuario
    # ------------------------------------------
    with tab_nueva:
        st.sidebar.info("Sube CSV de liquidaciones y CSV de precios mercado (daily FF):")
        liq_upload   = st.file_uploader("📥 CSV liquidaciones",  type="csv", key="upload_liq")
        price_upload = st.file_uploader("📥 CSV precios mercado", type="csv", key="upload_dia_ff")

        if liq_upload and price_upload:
            df_liq       = pd.read_csv(liq_upload,   parse_dates=['sale date'])
            df_prec_dia  = pd.read_csv(price_upload, parse_dates=['FECHA'])
            comparacion_liq_page(df_liq, df_prec_dia)
        else:
            st.warning("Sube ambos archivos para iniciar la comparación.")



if __name__ == "__main__":
    main()
