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
    .main { background-color: #f8f9fa; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
        border-left: 5px solid #003366;
    }
    [data-testid="stMetricLabel"] {
        color: #1f1f1f !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #003366 !important;
    }
    h1, h2, h3 { color: #003366; }
    </style>
    """, unsafe_allow_html=True)

# --- 2. CARGA Y LIMPIEZA DE DATOS ---
URL_DATA = "https://raw.githubusercontent.com/mlondono13/Proyecto-Final/main/HDHI%20Admission%20data.csv"

@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url, low_memory=False)
    log_limpieza = []
    
    # --- MAPEO DE GÉNERO (M/F -> Masculino/Femenino) ---
    df['GENDER'] = df['GENDER'].map({'M': 'Masculino', 'F': 'Femenino'})
    log_limpieza.append("Mapeo de Género: Se transformaron etiquetas 'M/F' a 'Masculino/Femenino'.")
    
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
    
    # El multiselect ahora mostrará Masculino/Femenino automáticamente
    gender_options = df_raw['GENDER'].unique()
    gender = st.multiselect("Género", options=gender_options, default=gender_options)
    
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
                            title="Distribución de Edad", color_discrete_sequence=["#003366", "#D4AF37", "gray"],
                            labels={"AGE": "Edad", "OUTCOME": "Resultado"})
    st.plotly_chart(fig_age, use_container_width=True)

# --- TAB 2: RESPUESTA A PREGUNTAS DE NEGOCIO ---
with tab2:
    st.header("🎯 Análisis Estratégico y de Riesgo")
    
    st.subheader("1. Análisis de Comorbilidad y Supervivencia")
    st.markdown("Este gráfico jerárquico permite ver cómo interactúan el género y la diabetes en el desenlace del paciente.")
    
    df_sun = df_filtered.copy()
    df_sun['Diabetes'] = df_sun['DM'].map({1: 'Con Diabetes', 0: 'Sin Diabetes'})
    
    fig_sun = px.sunburst(
        df_sun, 
        path=['GENDER', 'Diabetes', 'OUTCOME'], 
        color='OUTCOME',
        color_discrete_map={'DISCHARGE': '#003366', 'DEAD': '#D4AF37'},
        title="Flujo de Riesgo: Género -> Diabetes -> Resultado",
        labels={"GENDER": "Género", "OUTCOME": "Resultado"}
    )
    st.plotly_chart(fig_sun, use_container_width=True)

    st.divider()

    st.subheader("2. Mapa de Calor: Correlaciones Clínicas")
    st.markdown("Identificación de relaciones entre biomarcadores y la estancia hospitalaria.")
    
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
    st.header("📋 Informe Ejecutivo de Consultoría")
    st.markdown("Generación de informes estratégicos mediante IA.")

    if not user_api_key:
        st.info("🔑 Ingrese su Groq API Key en la barra lateral para habilitar la generación de informes.")
    else:
        if st.button("🚀 Generar Informe de Situación Actual"):
            try:
                client = Groq(api_key=user_api_key)
                
                mortalidad_tasa = df_filtered['MORTALITY'].mean() * 100
                estancia_media = df_filtered['DURATION OF STAY'].mean()
                preexistencias = ['DM', 'HTN', 'CKD', 'CAD']
                top_comorbilidad = df_filtered[preexistencias].mean().idxmax()
                porcentaje_critico = (df_filtered[top_comorbilidad].mean() * 100)
                
                contexto_informe = f"""
                DATOS DEL SEGMENTO FILTRADO:
                - Volumen de pacientes: {len(df_filtered)}
                - Tasa de mortalidad: {mortalidad_tasa:.1f}%
                - Estancia promedio: {estancia_media:.1f} días
                - Comorbilidad prevalente: {top_comorbilidad} (presente en el {porcentaje_critico:.1f}% de los casos)
                - Rango de edad analizado: {age_range[0]} - {age_range[1]} años
                """

                with st.spinner("El Consultor Senior está redactando el informe..."):
                    response = client.chat.completions.create(
                        messages=[
                            {
                                "role": "system", 
                                "content": "Eres un Consultor Senior en Analítica de Salud de la Universidad EAFIT. Redacta informes ejecutivos, técnicos y formales."
                            },
                            {
                                "role": "user", 
                                "content": f"Genera un informe profesional estructurado (Resumen, Riesgos, Recomendaciones) para este contexto:\n{contexto_informe}"
                            }
                        ],
                        model="llama-3.3-70b-versatile", 
                        temperature=0.3
                    )

                    st.success("Informe generado con éxito")
                    st.markdown("---")
                    st.markdown(response.choices[0].message.content)
                    st.markdown("---")
                    
            except Exception as e:
                st.error(f"Error al generar el informe: {e}")
