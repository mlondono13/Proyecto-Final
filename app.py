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
st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #003366; }
    div.stButton > button:first-child { background-color: #003366; color: white; }
    h1, h2, h3 { color: #003366; }
    </style>
    """, unsafe_allow_ Harris=True)

# --- 2. CARGA Y LIMPIEZA DE DATOS (MÓDULO ETL) ---
URL_DATA = "https://raw.githubusercontent.com/mlondono13/Proyecto-Final/main/HDHI%20Admission%20data.csv"

@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url, low_memory=False)
    
    # Registro de limpieza para el usuario
    log_limpieza = []
    
    # 1. Limpieza de columnas numéricas
    float_cols = ['EF', 'HB', 'CREATININE', 'GLUCOSE', 'UREA', 'TLC', 'PLATELETS', 'AGE']
    for col in float_cols:
        nulls_before = df[col].isna().sum()
        df[col] = pd.to_numeric(df[col].astype(str).str.strip(), errors='coerce')
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        if nulls_before > 0:
            log_limpieza.append(f"Columna {col}: Se imputaron {nulls_before} valores nulos con la mediana ({median_val}).")

    # 2. Formateo de Fechas
    df['D.O.A'] = pd.to_datetime(df['D.O.A'], dayfirst=True, errors='coerce')
    df['D.O.D'] = pd.to_datetime(df['D.O.D'], dayfirst=True, errors='coerce')
    
    # 3. Creación de variables objetivo
    df['MORTALITY'] = df['OUTCOME'].apply(lambda x: 1 if x == 'DEAD' else 0)
    
    return df, log_limpieza

df_raw, logs = load_and_clean_data(URL_DATA)

# --- 3. BARRA LATERAL (FILTROS Y API) ---
with st.sidebar:
    st.image("https://www.eafit.edu.co/LogoEAFIT.png", width=150) # Opcional si tienes la URL
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
st.markdown("Analítica avanzada para la gestión de admisiones hospitalarias y riesgo de mortalidad.")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard General", "🎯 Preguntas de Negocio", "🧹 Módulo de Limpieza", "🤖 IA Consultor"])

# --- TAB 1: KPI Y EXPLORACIÓN ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pacientes", len(df_filtered))
    col2.metric("Edad Media", f"{df_filtered['AGE'].mean():.1f} años")
    col3.metric("Tasa Mortalidad", f"{(df_filtered['MORTALITY'].mean()*100):.1f}%")
    col4.metric("Estancia Media", f"{df_filtered['DURATION OF STAY'].mean():.1f} días")

    c1, c2 = st.columns(2)
    with c1:
        fig_age = px.histogram(df_filtered, x="AGE", color="OUTCOME", title="Distribución de Edad por Resultado",
                               color_discrete_sequence=["#003366", "#D4AF37"])
        st.plotly_chart(fig_age, use_container_width=True)
    with c2:
        fig_stay = px.box(df_filtered, x="OUTCOME", y="DURATION OF STAY", title="Estancia Hospitalaria vs Resultado",
                          color="OUTCOME", color_discrete_sequence=["#003366", "#D4AF37"])
        st.plotly_chart(fig_stay, use_container_width=True)

# --- TAB 2: RESPUESTA A PREGUNTAS DE NEGOCIO ---
with tab2:
    st.header("🎯 Análisis Estratégico")
    
    # Pregunta 1: Factores de Riesgo
    st.subheader("1. ¿Qué condiciones preexistentes aumentan más el riesgo de muerte?")
    risk_cols = ['DM', 'HTN', 'CKD', 'HB', 'CAD']
    risk_analysis = df_filtered.groupby('OUTCOME')[risk_cols].mean().T
    fig_risk = px.bar(risk_analysis, barmode='group', title="Prevalencia de Enfermedades por Estado Final",
                      labels={'index': 'Condición', 'value': 'Proporción'},
                      color_discrete_sequence=["#003366", "#D4AF37"])
    st.plotly_chart(fig_risk, use_container_width=True)

    # Pregunta 2: Correlación Bioquímica
    st.subheader("2. Relación entre Creatinina, Glucosa y Mortalidad")
    fig_scatter = px.scatter(df_filtered, x="CREATININE", y="GLUCOSE", color="OUTCOME", 
                             size="DURATION OF STAY", hover_data=['AGE'],
                             title="Análisis de Biomarcadores",
                             color_discrete_sequence=["#003366", "#D4AF37"])
    st.plotly_chart(fig_scatter, use_container_width=True)

# --- TAB 3: MÓDULO DE LIMPIEZA (REQUISITO RÚBRICA) ---
with tab3:
    st.header("🧹 Proceso de ETL y Calidad de Datos")
    with st.expander("Ver bitácora de limpieza aplicada"):
        for log in logs:
            st.write(f"✅ {log}")
    
    st.subheader("Muestra de Datos Procesados")
    st.dataframe(df_filtered.head(10))

# --- TAB 4: CONSULTOR IA ---
with tab4:
    st.header("🤖 Consultor Senior con IA")
    if not user_api_key:
        st.warning("Ingrese su API Key en la barra lateral para activar la consultoría.")
    else:
        try:
            client = Groq(api_key=user_api_key)
            pregunta = st.text_input("Ej: ¿Cómo influye la Diabetes (DM) en la estancia de pacientes mayores de 70 años?")
            
            if pregunta:
                # Construcción de contexto robusto
                contexto = f"""
                Resumen estadístico actual:
                - Pacientes filtrados: {len(df_filtered)}
                - Mortalidad en este grupo: {df_filtered['MORTALITY'].mean()*100:.2f}%
                - Correlación CKD-Mortalidad: {df_filtered[['CKD', 'MORTALITY']].corr().iloc[0,1]:.2f}
                - Principales condiciones: {df_filtered[risk_cols].mean().to_dict()}
                """
                
                with st.spinner("Analizando..."):
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Eres un Consultor Senior en Ciencia de Datos Médicos para EAFIT. Sé técnico y estratégico."},
                            {"role": "user", "content": f"Contexto: {contexto}. Pregunta: {pregunta}"}
                        ],
                        model="llama3-8b-8192",
                    )
                    st.success("Análisis del Consultor:")
                    st.write(chat_completion.choices[0].message.content)
        except Exception as e:
            st.error(f"Error en la conexión con Groq: {e}")
