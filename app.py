import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HDHI Health Intelligence", layout="wide")

# --- FASE 1: FUNCIÓN DE LIMPIEZA (ETL) ---
@st.cache_data # Optimiza la carga de datos
def clean_data_final(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    # 1. Columnas críticas a FLOAT
    float_cols = ['EF', 'HB', 'CREATININE', 'GLUCOSE', 'UREA', 'TLC', 'PLATELETS']
    
    for col in float_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['', 'nan', '.', 'None'], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Imputación por mediana para evitar fallos en gráficas
        df[col] = df[col].fillna(df[col].median()).astype(float)

    # 2. Limpieza de Fechas
    df['D.O.A'] = pd.to_datetime(df['D.O.A'], dayfirst=True, errors='coerce')
    df['D.O.D'] = pd.to_datetime(df['D.O.D'], dayfirst=True, errors='coerce')
    
    # 3. Variable de estancia (Target)
    df['STAY_DAYS'] = (df['D.O.D'] - df['D.O.A']).dt.days.astype(float)
    df['STAY_DAYS'] = df['STAY_DAYS'].fillna(0.0)
    
    # 4. Limpieza de categorías para visualización
    df['GENDER'] = df['GENDER'].map({'M': 'Masculino', 'F': 'Femenino'})
    df['RURAL'] = df['RURAL'].map({'R': 'Rural', 'U': 'Urbano'})
    
    return df

# --- CARGA DE DATOS ---
try:
    df = clean_data_final('https://raw.githubusercontent.com/mlondono13/Proyecto-Final/main/HDHI%20Admission%20data.csv')
    st.sidebar.success("✅ Datos cargados y limpiados")
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar datos: {e}")
    st.stop()

# --- INTERFAZ DEL DASHBOARD ---
st.title("🏥 Sistema de Soporte a la Decisión - HDHI")
st.markdown("Análisis avanzado de datos clínicos y factores de riesgo.")

# Sidebar con filtros
st.sidebar.header("Filtros Globales")
genero = st.sidebar.multiselect("Género:", options=df['GENDER'].unique(), default=df['GENDER'].unique())
sector = st.sidebar.multiselect("Ubicación:", options=df['RURAL'].unique(), default=df['RURAL'].unique())

# Aplicar filtros
df_filtered = df[(df['GENDER'].isin(genero)) & (df['RURAL'].isin(sector))]

# --- LAYOUT DE PESTAÑAS ---
tab1, tab2, tab3 = st.tabs(["📊 Vista General", "🧬 Análisis Clínico", "🤖 Consultoría IA"])

with tab1:
    # Métricas principales
    col1, col2, col3 = st.columns(3)
    col1.metric("Pacientes Filtrados", len(df_filtered))
    col2.metric("Edad Promedio", f"{df_filtered['AGE'].mean():.1f} años")
    col3.metric("Estancia Media", f"{df_filtered['STAY_DAYS'].mean():.1f} días")
    
    # Visualización inicial
    st.subheader("Distribución de Edad y Estancia")
    fig = px.box(df_filtered, x="GENDER", y="AGE", color="OUTCOME", 
                 title="Distribución de Edad por Género y Resultado")
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Factores de Riesgo: Hemoglobina vs Creatinina")
    # Gráfico Multidimensional
    fig_clinico = px.scatter(df_filtered, x="HB", y="CREATININE", 
                             size="STAY_DAYS", color="EF",
                             hover_data=['AGE', 'DM', 'HTN'],
                             title="Relación HB vs Creatinina (Tamaño=Días, Color=Fracción Eyección)")
    st.plotly_chart(fig_clinico, use_container_width=True)

with tab3:
    st.info("Próximamente: Integración con LLM Groq para recomendaciones estratégicas.")
