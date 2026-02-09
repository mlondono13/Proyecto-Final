import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from groq import Groq

# --- 1. CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="HDHI Health Intelligence", layout="wide", page_icon="🏥")

# --- 2. CONSTANTES Y CARGA DE DATOS ---
URL_DATA = "https://raw.githubusercontent.com/mlondono13/Proyecto-Final/main/HDHI%20Admission%20data.csv"

@st.cache_data
def load_and_clean_data(url):
    df = pd.read_csv(url, low_memory=False)
    
    # Columnas críticas a convertir en FLOAT
    float_cols = ['EF', 'HB', 'CREATININE', 'GLUCOSE', 'UREA', 'TLC', 'PLATELETS']
    
    for col in float_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace(['', 'nan', '.', 'None', ' ', 'N/A'], np.nan)
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median()).astype(float)

    # Limpieza de Fechas
    df['D.O.A'] = pd.to_datetime(df['D.O.A'], dayfirst=True, errors='coerce')
    df['D.O.D'] = pd.to_datetime(df['D.O.D'], dayfirst=True, errors='coerce')
    
    # Cálculo de estancia (STAY_DAYS)
    df['STAY_DAYS'] = (df['D.O.D'] - df['D.O.A']).dt.days.astype(float)
    df['STAY_DAYS'] = df['STAY_DAYS'].fillna(0.0)
    
    # Ajuste para evitar error en Plotly size
    df['STAY_DAYS_VISUAL'] = df['STAY_DAYS'].apply(lambda x: x if x > 0 else 0.5)
    
    # Mapeo de categorías
    df['GENDER'] = df['GENDER'].map({'M': 'Masculino', 'F': 'Femenino'})
    df['RURAL'] = df['RURAL'].map({'R': 'Rural', 'U': 'Urbano'})
    
    return df

# Ejecutar carga
try:
    df = load_and_clean_data(URL_DATA)
except Exception as e:
    st.error(f"Error al conectar con GitHub: {e}")
    st.stop()

# --- 3. SIDEBAR (FILTROS Y API KEY) ---
st.sidebar.title("Configuración")

# Campo para que el usuario ingrese su propia API KEY
user_api_key = st.sidebar.text_input("Ingresa tu Groq API Key:", type="password")
st.sidebar.caption("Obtén tu llave en: https://console.groq.com/")

st.sidebar.divider()
st.sidebar.header("Filtros de Análisis")

genero_f = st.sidebar.multiselect("Género:", options=df['GENDER'].unique(), default=df['GENDER'].unique())
sector_f = st.sidebar.multiselect("Ubicación:", options=df['RURAL'].unique(), default=df['RURAL'].unique())

# Aplicar filtros
df_filtered = df[(df['GENDER'].isin(genero_f)) & (df['RURAL'].isin(sector_f))]

# --- 4. CUERPO PRINCIPAL ---
st.title("🏥 Dashboard Inteligente Hospitalario (HDHI)")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📊 Estadísticas Base", "🧬 Análisis Clínico", "🤖 Consultoría IA"])

# --- TAB 1: VISTA GENERAL ---
with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pacientes", f"{len(df_filtered):,}")
    col2.metric("Edad Media", f"{df_filtered['AGE'].mean():.1f} años")
    col3.metric("Estancia Media", f"{df_filtered['STAY_DAYS'].mean():.1f} d")
    col4.metric("HB Promedio", f"{df_filtered['HB'].mean():.1f}")

    st.subheader("Distribución de Edad")
    fig_age = px.histogram(df_filtered, x="AGE", color="GENDER", nbins=30, 
                           title="Pirámide Poblacional", barmode='overlay')
    st.plotly_chart(fig_age, use_container_width=True)

# --- TAB 2: ANÁLISIS CLÍNICO ---
with tab2:
    st.subheader("Relación Laboratorios vs Estancia")
    fig_scatter = px.scatter(
        df_filtered, x="HB", y="CREATININE", 
        size="STAY_DAYS_VISUAL", color="EF",
        hover_data={'STAY_DAYS': True, 'STAY_DAYS_VISUAL': False, 'AGE': True},
        title="Dispersión Clínica (Tamaño: Estancia, Color: Fracción Eyección)",
        color_continuous_scale="RdYlGn"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    cols_corr = ['AGE', 'HB', 'EF', 'CREATININE', 'GLUCOSE', 'STAY_DAYS', 'DM', 'HTN', 'CKD']
    corr_matrix = df_filtered[cols_corr].corr()
    fig_heat = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', title="Mapa de Correlación")
    st.plotly_chart(fig_heat, use_container_width=True)

# --- TAB 3: CONSULTORÍA IA ---
with tab3:
    st.header("🤖 Consultor Médico con IA")
    
    if not user_api_key:
        st.warning("⚠️ Por favor, ingresa tu Groq API Key en la barra lateral para usar la IA.")
    else:
        try:
            client = Groq(api_key=user_api_key)
            pregunta = st.text_input("Haz una pregunta sobre los datos:")
            
            if pregunta:
                # Preparamos contexto resumido
                contexto = f"""
                Pacientes: {len(df_filtered)}. Edad media: {df_filtered['AGE'].mean():.1f}. 
                Mortalidad total: {len(df_filtered[df_filtered['OUTCOME'] == 'DEAD'])}.
                Correlación DM con Estancia: {corr_matrix.loc['DM', 'STAY_DAYS']:.2f}.
                """
                
                with st.spinner("La IA está analizando los datos..."):
                    completion = client.chat.completions.create(
                        model="llama3-8b-8192",
                        messages=[
                            {"role": "system", "content": "Eres un experto médico. Responde basado en los datos proporcionados."},
                            {"role": "user", "content": f"Datos: {contexto}. Pregunta: {pregunta}"}
                        ]
                    )
                    st.markdown("### 💡 Respuesta del Consultor:")
                    st.write(completion.choices[0].message.content)
        except Exception as e:
            st.error(f"Error con la API Key: {e}")

# --- FOOTER ---
st.sidebar.markdown("---")
st.sidebar.write("Autor: Manuel Londoño")
