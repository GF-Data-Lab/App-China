# ────────────────────────────
# 0. IMPORTS Y CONFIG GENERAL
# ────────────────────────────
import streamlit as st
import altair as alt
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from anomaly_gescom_tab import anomalias_gescom_page
from anomaly_forever import anomalias_liq_page
# ↓ solo si sigues usando Azure OpenAI en los otros tabs
from openai import AzureOpenAI
from models.alert_sender import AlertSender
import os
import streamlit_oauth as streamlit_auth_component
from streamlit_oauth import OAuth2Component  # noqa: F401
# ---- BLOQUE DE AUTENTICACIÓN (sin cambios) ----
AUTHORIZE_URL = os.environ.get(
    'AUTHORIZE_URL',
    "https://login.microsoftonline.com/46ae710d-4335-430b-b7c8-f87b925b1d44/oauth2/v2.0/authorize"
)
TOKEN_URL = os.environ.get(
    'TOKEN_URL',
    "https://login.microsoftonline.com/46ae710d-4335-430b-b7c8-f87b925b1d44/oauth2/v2.0/token"
)
REFRESH_TOKEN_URL = os.environ.get('REFRESH_TOKEN_URL', TOKEN_URL)
REVOKE_TOKEN_URL  = os.environ.get('REVOKE_TOKEN_URL', None) 

CLIENT_ID     = os.environ.get('CLIENT_ID', "a55dc350-8107-46dd-bd32-a46f921a65ba")
CLIENT_SECRET = os.environ.get('CLIENT_SECRET', "5x_8Q~aHSERSz5jTocAS2V42GnJ5DJPUQgRCjbOq")
REDIRECT_URI  = os.environ.get(
    'REDIRECT_URI',
    "https://chinaprice-b6caard6euf5g6cj.brazilsouth-01.azurewebsites.net/"
)
SCOPE = os.environ.get('SCOPE', "User.Read")

oauth2 = OAuth2Component(
    CLIENT_ID,
    CLIENT_SECRET,
    AUTHORIZE_URL,
    TOKEN_URL,
    REFRESH_TOKEN_URL,
    REVOKE_TOKEN_URL
)

if 'token' not in st.session_state:
    st.session_state['token'] = None

# ---- PANTALLA DE LOGIN ----
if st.session_state['token'] is None:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("garces_data_analytics.png", width=300)
        st.markdown(
            "<h3 style='text-align: center;'>Inicia sesión para continuar</h3>",
            unsafe_allow_html=True
        )
        with st.spinner("Esperando autenticación…"):
            result = oauth2.authorize_button(
                "🟦 Iniciar sesión con Microsoft",
                REDIRECT_URI,
                SCOPE
            )
        if result and 'token' in result:
            st.session_state.token = result.get('token')
            st.rerun()
    st.stop()   # <<< Sale si no hay token
# Configuración global de la página
st.set_page_config(page_title="Comparador & Anomalías", layout="wide")

# ────────────────────────────
# 1. CONSTANTES COMUNES
# ────────────────────────────
# Pestaña «Comparación actual»
ARRIVALS_FILE = Path("GESCOM (1).csv")
PRICES_FILE   = Path("precios_mercado.csv")
LIQ_PRICE_COL = "Sales price in RMB"  # ⇦ cámbialo según tu CSV
# Parámetros de Azure OpenAI (idénticos a tu código original)
endpoint         = 'https://aif-prod-eu2-001.cognitiveservices.azure.com/'
subscription_key = 'AHwZuK6uZewwI4wrNTFrkv3krJfhbilfy4iQnLaRlCpJ0OBx8XFWJQQJ99BFACHYHv6XJ3w3AAAAACOG0KIa'
api_version      = "2024-12-01-preview"
deployment       = "gpt-4o"
client = AzureOpenAI(
    api_version    = api_version,
    azure_endpoint = endpoint,
    api_key        = subscription_key,
)

# Umbral para alertas en % (lo usas en 2 pestañas → lo mantenemos global)
ALERT_THRESHOLD = 5

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
    """Devuelve un dataframe con las diferencias CIF vs mercado
    usando **promedio** y **moda** para los 7 días siguientes.

    Columns devueltas:
      - AmountAvg / Diff_Abs / Diff_Pct         → promedio del mercado
      - AmountMode / Diff_Abs_Mode / Diff_Pct_Mode → moda del mercado
    """
    day     = row["ArrivalDate.Date"]
    zona    = row["ZONA"]
    specie  = row["ESPECIE"]
    size    = row["CALIBRE"]
    codigo  = row["CODIGO_PM"]
    brand   = row["ETIQUETA"]
    cif     = row["CIF"]
    folio      = row.get("FOLIO")
    container  = row.get("CONTENEDOR", row.get("CONTAINER"))
    broker     = row.get("BROKER")

    fechas_rango = pd.date_range(start=day, end=day + pd.Timedelta(days=7), freq="D")
    registros = []
    for fecha in fechas_rango:
        mask = (
            (precios_df["FECHA"]     == fecha) &
            (precios_df["ZONA"]      == zona)  &
            (precios_df["ESPECIE"]   == specie) &
            (precios_df["CALIBRE"]   == size)   &
            (precios_df["CODIGO_PM"] == codigo) &
            (precios_df["ETIQUETA"]  == brand)
        )
        df_dia = precios_df.loc[mask]
        if not df_dia.empty:
            amount_avg: Optional[float] = df_dia["AverageAmount"].mean()

            # --- Moda ---
            moda_series = df_dia["AverageAmount"].mode(dropna=True)
            amount_mode: Optional[float] = moda_series.iloc[0] if not moda_series.empty else None

            # --- Diferencias vs promedio ---
            diff_abs_avg = cif - amount_avg
            diff_pct_avg = (diff_abs_avg / amount_avg * 100) if amount_avg else None

            # --- Diferencias vs moda ---
            diff_abs_mode = cif - amount_mode if amount_mode is not None else None
            diff_pct_mode = (diff_abs_mode / amount_mode * 100) if amount_mode else None
        else:
            amount_avg = amount_mode = diff_abs_avg = diff_pct_avg = diff_abs_mode = diff_pct_mode = None

        registros.append({
            "FOLIO": folio,
            "CONTENEDOR": container,
            "BROKER": broker,
            "FechaPrecio": fecha.date(),
            "CIF": cif,
            "AmountAvg": amount_avg,
            "AmountMode": amount_mode,
            "Diff_Abs": diff_abs_avg,
            "Diff_Pct": diff_pct_avg,
            "Diff_Abs_Mode": diff_abs_mode,
            "Diff_Pct_Mode": diff_pct_mode,
        })

    return pd.DataFrame(registros)


def _evalua_alertas(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Filtra las filas cuya diferencia en % (promedio o moda) excede el umbral."""
    cond_avg  = pd.to_numeric(df_daily["Diff_Pct"], errors="coerce").abs()  > ALERT_THRESHOLD
    cond_mode = pd.to_numeric(df_daily["Diff_Pct_Mode"], errors="coerce").abs() > ALERT_THRESHOLD
    return df_daily[cond_avg | cond_mode]


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
        sender = AlertSender(dict_alerts, 'juan.latife@garcesfruit.com')
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
# ────────────────────────────
# 2. PESTAÑA 1  «Comparación actual»
# ────────────────────────────
# (se copia tu función comparacion_page tal cual, añadiendo `key_prefix="orig"`)

# ……………  << aquí pega tu función comparacion_page() intacta >>  ……………
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

# ────────────────────────────
# 3. PESTAÑA 2  «Nueva comparación»
# ────────────────────────────
# (se copia tu función comparacion_liq_page tal cual, añadiendo key_prefix="new")

# ……………  << aquí pega tu función comparacion_liq_page() intacta >>  ……………
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
            codigo = row['CODIGO_PM']
            liq_price = row[LIQ_PRICE_COL]
            codigo_real = row['Packaging specification name']
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
                    & (precios_dia_ff['CODIGO_PM'] == codigo)
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
                        & (precios_dia_ff['CODIGO_PM'] == codigo)
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
                    "Calibre": calibre,
                    "Brand": brand,
                    "Codigo": codigo,
                    "Codigo real": codigo_real,
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

# ────────────────────────────
# 4. PESTAÑA 3  «Detección de anomalías»
# ────────────────────────────
def anomalias_page(key_prefix: str = "anom"):
    """
    Replica el flujo de tu script de anomalías dentro de su propio tab.
    Para mantener el ejemplo breve, sólo se resaltan los puntos clave que cambian:
      • Todos los widgets llevan key=f"{key_prefix}_..."  
      • Se reutiliza `st.cache_data` con nombre único
    """
    st.header("🔎 Detección de anomalías en precios")

    DEFAULT_FILE = "precios_ff.csv"
    uploaded_file = st.file_uploader(
        "Subir archivo (CSV/Excel)", type=["csv", "xlsx"],
        key=f"{key_prefix}_uploader",
        help="Si no subes nada, se usará forever_todo.csv en el directorio actual.")

    # ---------- 1. Cargar datos ----------
    @st.cache_data(show_spinner=False)
    def _load_anom_data(file):
        if isinstance(file, (Path, str)):
            return pd.read_excel(file) if str(file).lower().endswith(".xlsx") else pd.read_csv(file, sep=";")
        return pd.read_excel(file) if file.name.lower().endswith(".xlsx") else pd.read_csv(file)

    if uploaded_file is not None:
        df = _load_anom_data(uploaded_file)
        st.success(f"Archivo cargado: **{uploaded_file.name}**")
    else:
        if Path(DEFAULT_FILE).is_file():
            df = _load_anom_data(DEFAULT_FILE)
            st.info(f"Se cargó automáticamente **{DEFAULT_FILE}**")
        else:
            st.error("No se encontró archivo por defecto y no subiste ninguno.")
            st.stop()
    df = pd.read_csv("precios_ff.csv")
    NUM_COLS   = ["AmountMAX", "AmountMIN", "AverageAmount"]
    GROUP_COLS = ["ESPECIE", "ZONA", "VARIEDAD COMERCIAL", "CALIBRE", "Customers", "ETIQUETA"]
    df[NUM_COLS] = df[NUM_COLS].apply(pd.to_numeric, errors="coerce")
    df["is_anomaly_if"] = False
    df["if_score"]      = None

    # ---------- 2. Filtros ----------
    with st.form("filtros_anomalias"):                 # ⬅️ NUEVO contenedor
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        fruit     = col1.selectbox("ESPECIE",   ["(Todas)"] + sorted(df["ESPECIE"].dropna().unique()))
        region    = col2.selectbox("ZONA",      ["(Todas)"] + sorted(df["ZONA"].dropna().unique()))
        variety   = col3.selectbox("VARIEDAD COMERCIAL",
                                   ["(Todas)"] + sorted(df["VARIEDAD COMERCIAL"].dropna().unique()))
        brand     = col4.selectbox("ETIQUETA",  ["(Todas)"] + sorted(df["ETIQUETA"].dropna().unique()))
        size      = col5.selectbox("CALIBRE",   ["(Todas)"] + sorted(df["CALIBRE"].dropna().unique()))
        codigo    = col6.selectbox("CODIGO_PM", ["(Todas)"] + sorted(df["CODIGO_PM"].dropna().unique()))
        customer  = col7.selectbox("Customers", ["(Todas)"] + sorted(df["Customers"].dropna().unique()))

        filtros_ok = st.form_submit_button("Guardar cambios")  # ⬅️ EL BOTÓN

    # Solo aplica los filtros y continúa si el usuario los guardó
    if not filtros_ok:
        st.info("Ajusta los filtros y pulsa **Guardar cambios** para aplicarlos.")
        # st.stop()                            # evita recargas mientras elige filtros

    mask = pd.Series(True, index=df.index)
    if fruit    != "(Todas)": mask &= df["ESPECIE"]   == fruit
    if region   != "(Todas)": mask &= df["ZONA"]      == region
    if variety  != "(Todas)": mask &= df["VARIEDAD COMERCIAL"] == variety
    if size     != "(Todas)": mask &= df["CALIBRE"]   == size
    if brand    != "(Todas)": mask &= df["ETIQUETA"]  == brand
    if customer != "(Todas)": mask &= df["Customers"] == customer
    if codigo   != "(Todas)": mask &= df["CODIGO_PM"] == codigo

    filtered_df = df.loc[mask].copy()
    st.success(f"🔎 Filtrado aplicado: {len(filtered_df)} filas")


    # ---------- 3. Parámetros modelo ----------
    with st.expander("⚙️ Parámetros del modelo"):
        min_rows     = st.number_input("Mínimo de filas por grupo", 5, 200, 15, key=f"{key_prefix}_minrows")
        contamination= st.slider("Contaminación esperada (%)", 1, 20, 2, key=f"{key_prefix}_cont")/100
        n_estimators = st.number_input("N.º árboles", 50, 400, 200, step=50, key=f"{key_prefix}_nest")

    IF_PARAMS = dict(n_estimators=int(n_estimators), contamination=contamination, random_state=42)

    # ---------- 4. Función detección ----------
    def marcar_anomalias(grp: pd.DataFrame) -> pd.DataFrame:
        if grp[NUM_COLS].notna().all(axis=1).sum() < min_rows:
            return grp
        mask_ok = grp[NUM_COLS].notna().all(axis=1)
        X = grp.loc[mask_ok, NUM_COLS]
        iso = IsolationForest(**IF_PARAMS).fit(X)
        preds  = iso.predict(X)          # -1 anomalía
        scores = iso.decision_function(X)
        grp.loc[mask_ok, "is_anomaly_if"] = preds == -1
        grp.loc[mask_ok, "if_score"]      = scores
        return grp

    IF_PARAMS = dict(n_estimators=int(n_estimators), contamination=contamination, random_state=42)

    # Entrena el modelo al hacer click
    if st.button("Entrenar modelo", key=f"{key_prefix}_train"):
        result_df = (
            filtered_df
            .groupby(GROUP_COLS, group_keys=False)
            .apply(marcar_anomalias)
            .reset_index(drop=True)
        )
        anomalies = result_df[result_df["is_anomaly_if"]]

        # Resumen y conteo
        st.subheader("📊 Resumen de detección de anomalías")
        st.write(
            f"Total registros analizados: {len(result_df):,}  |  "
            f"Anómalos: {len(anomalies):,} ({len(anomalies)/len(result_df):.2%})"
        )

        # Mostrar tabla de anomalías directamente
        if not anomalies.empty:
            st.subheader("✅ Anomalías detectadas")
            st.dataframe(
                anomalies[GROUP_COLS + NUM_COLS + ["if_score"]],
                use_container_width=True
            )
        else:
            st.info("No se detectaron anomalías con los parámetros seleccionados.")

        # Expander con detalle adicional si se desea
        with st.expander("Ver anomalías (detalle)"):
            st.dataframe(
                anomalies[GROUP_COLS + NUM_COLS + ["if_score"]],
                use_container_width=True
            )

        # Descargar CSV
        st.download_button(
            "Descargar anomalías (CSV)",
            data=anomalies.to_csv(index=False).encode(),
            file_name="anomalies.csv",
            mime="text/csv",
            key=f"{key_prefix}_dl_csv"
        )

        # Gráfico interactivo
        result_df['FECHA'] = pd.to_datetime(result_df['FECHA'], errors='coerce')
        result_df = result_df.sort_values('FECHA')
        base = alt.Chart(result_df).encode(
            x=alt.X('FECHA:T', title='Fecha'),
            y=alt.Y('AverageAmount:Q', title='AverageAmount')
        )
        line = base.mark_line()
        anom = (
            base.transform_filter('datum.is_anomaly_if')
                .mark_circle(color='red', size=90)
                .encode(tooltip=GROUP_COLS + NUM_COLS + ['if_score'])
        )
        st.altair_chart(line + anom, use_container_width=True)

# ────────────────────────────
# 5. APP PRINCIPAL (multitab)
# ────────────────────────────
def main():
    st.title("🧮 Comparador CIF vs Mercado & Anomalías")

    tab_actual, tab_nueva, tab_anom, tab_ges, tab_forev = st.tabs(
        ["Liquidaciones GESCOM",
         "Liquidaciones Forever Fresh",
         "Detección de anomalías",
         "Detección de anomalías GESCOM",
         "Detección de anomalías ForeverFresh"]          
    )


    # ---- Tab 1 —— archivos por defecto -------------------
    with tab_actual:
        df_arrivals, df_prices = load_data_from_path(ARRIVALS_FILE, PRICES_FILE)
        comparacion_page(df_arrivals, df_prices, key_prefix="orig")

    # ---- Tab 2 —— uploads del usuario --------------------
    with tab_nueva:
        liq_upload   = "precios_liq_ff.csv"
        price_upload = "precios_ff.csv"
        if liq_upload and price_upload:
            df_liq      = pd.read_csv(liq_upload,   parse_dates=['sale date'])
            df_prec_dia = pd.read_csv(price_upload, parse_dates=['FECHA'])
            comparacion_liq_page(df_liq, df_prec_dia)
        else:
            st.info("Sube ambos archivos para iniciar la comparación.")

    # ---- Tab 3 —— anomalías ------------------------------
    with tab_anom:
        anomalias_page(key_prefix="anom")

    # ---- Tab 4 —— anomalías GESCOM ----------------------------
    with tab_ges:
        anomalias_gescom_page(df_arrivals, key_prefix="ges")
    
    with tab_forev:
        anomalias_liq_page(key_prefix="ff")



# Entry‑point
if __name__ == "__main__":
    main()
