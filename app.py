import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# --- CONFIGURACIÓN VISUAL MEJORADA ---
st.set_page_config(page_title="Clasificador Perros y Gatos IA", layout="centered", page_icon="🐾")

# Título y encabezado personalizado
st.markdown(
    """
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="color: #2E4053;">🐾 Clasificador de Imágenes: Perros vs Gatos</h1>
        <p style="font-size: 1.1em; color: #5D6D7E;">
            Modelo predictivo para la clase de IA — Campus Comayagua — 2026<br>
            <b>Desarrollado por: Richard Andino - 20231900184</b>
        </p>
    </div>
    <hr style="margin-top: 0; margin-bottom: 25px;">
    """, 
    unsafe_allow_html=True
)

st.write("Sube una imagen para que la red neuronal MobileNetV2 determine si se trata de un perro o un gato.")

# --- CONFIGURACIÓN DE RUTAS Y CONSTANTES ---
IMG_SIZE = (224, 224)
MODEL_DIR = Path("modelo_perros_gatos_mobilenet")
CLASS_PATH = MODEL_DIR / "class_names.json"
MODEL_PATHS = [
    MODEL_DIR / "perros_gatos_mobilenet.keras", 
    MODEL_DIR / "perros_gatos_mobilenet.h5"
]

# Diccionario de traducción por si tus carpetas del dataset están en inglés (ej: "cats", "dogs")
# Si tus carpetas ya se llamaban "Gatos" y "Perros", el sistema las dejará igual.
LABELS_ES = {
    "cats": "Gato",
    "dogs": "Perro",
    "Gatos": "Gato",
    "Perros": "Perro"
}

# --- CARGA OPTIMIZADA DE RECURSOS ---
@st.cache_resource
def cargar_modelo():
    for path in MODEL_PATHS:
        if path.exists():
            return tf.keras.models.load_model(path, compile=False)
    st.error("❌ No se encontró el modelo. Coloque la carpeta 'modelo_perros_gatos_mobilenet' junto a app.py.")
    st.stop()

@st.cache_data
def cargar_clases():
    if CLASS_PATH.exists():
        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return ["Gatos", "Perros"]

# --- PROCESAMIENTO DE IMÁGENES ---
def preparar_imagen(img):
    img = img.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    # Preprocesamiento idéntico al usado con MobileNetV2 en el entrenamiento
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predecir(img):
    preds = modelo.predict(preparar_imagen(img), verbose=0)[0]
    # Ordenar índices de mayor a menor probabilidad
    indices_ordenados = np.argsort(preds)[::-1]
    
    return [
        (LABELS_ES.get(clases[i], clases[i]), float(preds[i]) * 100)
        for i in indices_ordenados
    ]

# Inicializar modelo y etiquetas
modelo = cargar_modelo()
clases = cargar_clases()

# --- INTERFAZ DE USUARIO ---
archivo = st.file_uploader("Seleccione una imagen de un perrito o gatito", type=["jpg", "jpeg", "png"])

if archivo:
    # Crear dos columnas visuales: Izquierda para la imagen, Derecha para los resultados
    col1, col2 = st.columns([1, 1], gap="large")
    
    imagen = Image.open(archivo)
    
    with col1:
        st.markdown("### 📸 Imagen Analizada")
        st.image(imagen, use_container_width=True)
        
    with col2:
        st.markdown("### 🧠 Análisis de la IA")
        with st.spinner("Clasificando..."):
            resultados = predecir(imagen)
        
        # Mostrar el resultado principal de forma destacada usando un contenedor métrico
        clase_principal, prob_principal = resultados[0]
        
        st.metric(
            label="Predicción Principal", 
            value=f"{clase_principal}", 
            delta=f"{prob_principal:.2f}% de Certeza"
        )
        
        st.write("---")
        st.write("**Probabilidades detalladas:**")
        
        # Barras de progreso visuales para cada clase
        for clase, prob in resultados:
            st.write(f"**{clase}** ({prob:.2f}%)")
            st.progress(int(prob))
else:
    st.info("🐾 Por favor, cargue un archivo de imagen para iniciar la clasificación automática.")
