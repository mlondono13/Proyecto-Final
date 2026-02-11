# 🏥 HDHI Health Intelligence: Sistema de Soporte a la Decisión
**Proyecto Final - Fundamentos de Ciencia de Datos | Universidad EAFIT**

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-Llama_3.3-orange?style=for-the-badge)

## 📝 Descripción del Problema
Este proyecto surge de la necesidad de transformar datos clínicos complejos en estrategias accionables para el **Heart Design and Health Institute (HDHI)**. Como Consultores de Datos Senior, hemos desarrollado una plataforma que integra limpieza automatizada, análisis visual avanzado e inteligencia artificial generativa para mitigar riesgos de mortalidad y optimizar la estancia hospitalaria.

## 🎯 Preguntas de Negocio Resueltas
El dashboard responde dinámicamente a los siguientes interrogantes estratégicos:
1. **Factores de Riesgo:** ¿Cuáles comorbilidades tienen mayor peso estadístico en el desenlace de mortalidad?
2. **Correlaciones Clínicas:** ¿Existe una relación directa entre los niveles de creatinina/hemoglobina y el tiempo de estancia?
3. **Análisis de Segmentos:** ¿Cómo varía el riesgo según el género y la presencia de diabetes en pacientes críticos?

## 🚀 Instalación y Ejecución Local

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/tu-usuario/nombre-del-repositorio.git](https://github.com/tu-usuario/nombre-del-repositorio.git)
   cd nombre-del-repositorio
2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
3. **Ejecutar aplicación:**
     ```bash
       streamlit run app.py
-------
## 🌐 Link al Despliegue
Accede a la plataforma en vivo:

👉 https://proyectofinalfdc.streamlit.app/

----------
## 🛠️ Tecnologías y Metodología
ETL: Limpieza e imputación de nulos mediante medianas para asegurar la integridad estadística y evitar sesgos por valores atípicos.

EDA: Visualizaciones interactivas desarrolladas con Plotly, incluyendo diagramas de Sunburst para jerarquía de riesgos y Heatmaps de correlación clínica.

LLM Integration: Implementación de la API de Groq utilizando el modelo llama-3.3-70b-versatile para la generación automatizada de informes ejecutivos.

Gemini said
Aquí tienes el bloque completo en formato Markdown limpio y listo para copiar en tu archivo README.md:

-------------------
##👥 Créditos
Autor: Marcel Londoño Leon - Jerónimo Piedrahita Franco

Institución: Universidad EAFIT

Materia: Fundamentos de Ciencia de Datos

Dataset: HDHI Admission Data.
