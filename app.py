import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from groq import Groq

# --- 1. CONFIGURACIÓN DE PÁGINA (ESTILO EAFIT) ---
st.set_page_config(
    page_title="HDHI Health Intelligence | EAFIT",
    layout="wide",
    page_icon="🏥"
)

# Estilo CSS para colores institucionales (Azul y Dorado)
# Estilo CSS para tarjetas con etiquetas oscuras
st.markdown("""
    <style>
    /* Fondo de la página */
    .main { background-color: #f8f9fa; }
    
    /* Contenedor de la tarjeta de métrica */
    div[data-testid="stMetric"] {
        background-color: #ffffff; /* Fondo blanco para contraste */
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1); /* Sombra suave */
        border-left: 5px solid #003366; /* Borde lateral azul EAFIT */
    }

    /* Color de la etiqueta (el título pequeño) */
    [data-testid="stMetricLabel"] {
        color: #1f1f1f !important; /* Gris muy oscuro / Negro */
        font-weight: 600;
    }

    /* Color del valor (el número grande) */
    [data-testid="stMetricValue"] {
        color: #003366 !important; /* Azul EAFIT */
    }

    /* Títulos generales */
    h1, h2, h3 { color: #003366; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA Y LIMPIEZA DE DATOS ---
URL_DATA = "https://raw.githubusercontent.com/mlondono13/Proyecto-Final/main/HDHI%20Admission%20data.csv"

@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url, low_memory=False)
    log_limpieza = []
    
    # Limpieza de columnas numéricas
    float_cols = ['EF', 'HB', 'CREATININE', 'GLUCOSE', 'UREA', 'TLC', 'PLATELETS', 'AGE']
    for col in float_cols:
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        median_val = df[col].median()
        nulls_count = df[col].isna().sum()
        df[col] = df[col].fillna(median_val)
        if nulls_count > 0:
            log_limpieza.append(f"Columna {col}: Se imputaron {nulls_count} valores con la mediana ({median_val}).")

    # Fechas y Mortalidad
    df['D.O.A'] = pd.to_datetime(df['D.O.A'], dayfirst=True, errors='coerce')
    df['MORTALITY'] = df['OUTCOME'].apply(lambda x: 1 if x == 'DEAD' else 0)
    
    return df, log_limpieza

try:
    df_raw, logs = load_and_clean_data(URL_DATA)
except Exception as e:
    st.error(f"Error al cargar datos desde GitHub: {e}")
    st.stop()

# --- 3. BARRA LATERAL ---
with st.sidebar:
    st.title("Configuración")
    user_api_key = st.text_input("Groq API Key", type="password")
    
    st.divider()
    st.subheader("Filtros Globales")
    age_range = st.slider("Rango de Edad", int(df_raw['AGE'].min()), int(df_raw['AGE'].max()), (20, 80))
    gender = st.multiselect("Género", options=df_raw['GENDER'].unique(), default=df_raw['GENDER'].unique())
    
    df_filtered = df_raw[
        (df_raw['AGE'].between(age_range[0], age_range[1])) &
        (df_raw['GENDER'].isin(gender))
    ]

# --- 4. CUERPO PRINCIPAL ---
st.title("🏥 Sistema de Soporte a la Decisión: HDHI")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Negocio", "🧹 Limpieza", "🤖 IA Consultor"])

with tab1:
    col1, col2, col3 = st.columns(3)
    col1.metric("Pacientes", len(df_filtered))
    col2.metric("Mortalidad %", f"{(df_filtered['MORTALITY'].mean()*100):.1f}%")
    col3.metric("Estancia Media", f"{df_filtered['DURATION OF STAY'].mean():.1f} días")

    fig_age = px.histogram(df_filtered, x="AGE", color="OUTCOME", barmode="overlay",
                           title="Distribución de Edad", color_discrete_sequence=["#003366", "#D4AF37", "gray"])
    st.plotly_chart(fig_age, use_container_width=True)

# --- TAB 2: RESPUESTA A PREGUNTAS DE NEGOCIO ---
with tab2:
    st.header("🎯 Análisis Estratégico y de Riesgo")
    
    # FILA 1: Jerarquía de Riesgo (Sunburst)
    st.subheader("1. Análisis de Comorbilidad y Supervivencia")
    st.markdown("Este gráfico jerárquico permite ver cómo interactúan el género y la diabetes en el desenlace del paciente.")
    
    # Creamos una columna auxiliar para que el gráfico sea más legible
    df_sun = df_filtered.copy()
    df_sun['Diabetes'] = df_sun['DM'].map({1: 'Con Diabetes', 0: 'Sin Diabetes'})
    
    fig_sun = px.sunburst(
        df_sun, 
        path=['GENDER', 'Diabetes', 'OUTCOME'], 
        color='OUTCOME',
        color_discrete_map={'DISCHARGE': '#003366', 'DEAD': '#D4AF37'},
        title="Flujo de Riesgo: Género -> Diabetes -> Resultado"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    

    st.divider()

    # FILA 2: Mapa de Calor de Correlaciones
    st.subheader("2. Mapa de Calor: Correlaciones Clínicas")
    st.markdown("Identificación de relaciones entre biomarcadores (Hemoglobina, Creatinina, Edad) y la estancia hospitalaria.")
    
    # Seleccionamos variables numéricas relevantes
    cols_corr = ['AGE', 'HB', 'CREATININE', 'GLUCOSE', 'DURATION OF STAY', 'MORTALITY']
    corr_matrix = df_filtered[cols_corr].corr()
    
    fig_heat = px.imshow(
        corr_matrix, 
        text_auto=".2f", 
        aspect="auto",
        color_continuous_scale=[(0, "#D4AF37"), (0.5, "#ffffff"), (1, "#003366")],
        title="Matriz de Correlación de Factores Críticos"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

with tab3:
    st.header("Módulo de Limpieza")
    st.info("Este módulo cumple con el requerimiento de visibilidad del proceso ETL.")
    for l in logs:
        st.write(f"✔️ {l}")
    st.dataframe(df_filtered.head(50))

with tab4:
    if not user_api_key:
        st.info("Introduce tu API Key de Groq para habilitar el consultor.")
    else:
        client = Groq(api_key=user_api_key)
        prompt = st.text_input("Pregunta al consultor:")
        if prompt:
            context = f"Pacientes analizados: {len(df_filtered)}. Tasa muerte: {df_filtered['MORTALITY'].mean():.2%}"
            chat = client.chat.completions.create(
                messages=[{"role": "system", "content": "Eres un experto en salud pública de EAFIT."},
                          {"role": "user", "content": f"{context}. Pregunta: {prompt}"}],
                model="llama3-8b-8192")
            st.write(chat.choices[0].message.content)
