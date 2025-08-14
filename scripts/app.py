# scripts/app.py
import streamlit as st
from pathlib import Path
import sys
from datetime import datetime
from PIL import Image
import configparser
import pandas as pd
import altair as alt

# --- 1. Configuración de Paths e Importaciones ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.macro_estimator.training.predictor import Predictor
from src.macro_estimator.database_utils import Database

# --- 2. Carga de Recursos y Inicialización ---

# Crear directorios necesarios
DB_PATH = Path("data/app_data/user_data.db")
IMAGE_STORAGE_PATH = Path("data/app_data/uploaded_images")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
IMAGE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

# Inicializar la base de datos
db = Database(db_path=DB_PATH)

@st.cache_resource
def load_predictor(config_path="config/config.yaml") -> Predictor:
    """Carga el modelo y la configuración una sola vez y lo guarda en caché."""
    print("--- Cargando el modelo... (esto solo se ejecuta una vez) ---")
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return Predictor(config)
    except Exception as e:
        st.error(f"Error fatal al cargar el modelo: {e}")
        return None

# --- 3. Lógica de la Interfaz de Usuario ---

def render_login_page():
    """Muestra la página de login y registro."""
    st.title("Bienvenido a Macro Estimator AI 🍽️")
    
    login_tab, register_tab = st.tabs(["Iniciar Sesión", "Registrarse"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Usuario", key="login_user")
            password = st.text_input("Contraseña", type="password", key="login_pass")
            submitted = st.form_submit_button("Iniciar Sesión")
            if submitted:
                user_id = db.check_user(username, password)
                if user_id:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.session_state.user_id = user_id
                    st.success("¡Sesión iniciada con éxito!")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos.")

    with register_tab:
        with st.form("register_form"):
            new_username = st.text_input("Nuevo Usuario", key="reg_user")
            new_password = st.text_input("Nueva Contraseña", type="password", key="reg_pass")
            confirm_password = st.text_input("Confirmar Contraseña", type="password", key="reg_pass_confirm")
            reg_submitted = st.form_submit_button("Registrarse")
            if reg_submitted:
                if new_password == '' or new_username == '' or confirm_password == '':
                    st.error("El usuario o la contraseña no pueden estar vacíos.")
                elif new_password == confirm_password:
                    user_created = db.add_user(new_username, new_password)
                    if user_created and new_password != '':
                        st.success("¡Usuario creado con éxito! Por favor, inicia sesión.")
                    else:
                        st.error("El nombre de usuario ya existe o los campos están vacíos.")
                else:
                    st.error("Las contraseñas no coinciden.")

def render_main_app(predictor):
    """Muestra la aplicación principal después del login."""
    
    # --- Sidebar ---
    with st.sidebar:
        st.subheader(f"Bienvenido, {st.session_state.username}!")
        if st.button("Cerrar Sesión"):
            st.session_state.logged_in = False
            del st.session_state.username
            del st.session_state.user_id
            st.rerun()

    # --- Interfaz de Carga de Imagen ---
    st.title("Estima las Macros de tu Comida")
    st.markdown("Sube una foto de tu plato y la IA estimará su contenido nutricional.")
    
    uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None and predictor:
        image = Image.open(uploaded_file)
        st.image(image, caption="Imagen subida.", use_container_width=True)
        
        if st.button("Calcular Macros"):
            with st.spinner('Analizando la imagen...'):
                image_bytes = uploaded_file.getvalue()
                prediction = predictor.predict_from_bytes(image_bytes)
                
                # Guardar la imagen subida con un nombre único
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{st.session_state.username}_{timestamp_str}_{uploaded_file.name}"
                image_path = IMAGE_STORAGE_PATH / image_filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)

                # Guardar en la base de datos
                db.add_meal(st.session_state.user_id, datetime.now(), str(image_path), prediction)
                
                st.success("¡Análisis completado y guardado en tu historial!")
                
                # Mostrar resultados
                st.subheader("Resultados de la Estimación")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Calorías", f"{prediction['calories']:.0f} kcal")
                col2.metric("Carbos", f"{prediction['carb_grams']:.1f} g")
                col3.metric("Grasas", f"{prediction['fat_grams']:.1f} g")
                col4.metric("Proteína", f"{prediction['protein_grams']:.1f} g")
                
                # Gráfico de Torta para la última comida
                macros_df = pd.DataFrame({
                    'Macro': ['Grasas (g)', 'Carbos (g)', 'Proteína (g)'],
                    'Gramos': [prediction['fat_grams'], prediction['carb_grams'], prediction['protein_grams']]
                })
                pie_chart = alt.Chart(macros_df).mark_arc().encode(
                    theta=alt.Theta(field="Gramos", type="quantitative"),
                    color=alt.Color(field="Macro", type="nominal")
                ).properties(title="Desglose de Macronutrientes")
                st.altair_chart(pie_chart, use_container_width=True)

    # --- Sección de Historial y Gráficos ---
    st.header("Historial y Estadísticas")
    user_history = db.get_user_meals(st.session_state.user_id)
    
    if user_history.empty:
        st.info("Aún no has registrado ninguna comida. ¡Sube una imagen para empezar!")
    else:
        # Gráfico de barras del historial
        st.subheader("Ingesta Diaria (últimos 7 días)")
        history_last_7_days = user_history[user_history['timestamp'] > (datetime.now() - pd.Timedelta(days=7))]
        daily_summary = history_last_7_days.groupby(history_last_7_days['timestamp'].dt.date).sum(numeric_only=True)
        st.bar_chart(daily_summary[['calories', 'carb_grams', 'fat_grams', 'protein_grams']])
        
        # Historial detallado
        st.subheader("Registro de Comidas")
        for index, entry in user_history.iterrows():
            with st.expander(f"Comida del {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(str(entry['image_path']), use_container_width=True)
                with col2:
                    st.text(f"Calorías: {entry['calories']:.0f} kcal")
                    st.text(f"Carbohidratos: {entry['carb_grams']:.1f} g")
                    st.text(f"Grasas: {entry['fat_grams']:.1f} g")
                    st.text(f"Proteína: {entry['protein_grams']:.1f} g")

def main():
    """Función principal que renderiza la aplicación Streamlit."""
    st.set_page_config(page_title="Macro Estimator AI", page_icon="🍽️", layout="centered")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        predictor = load_predictor()
        render_main_app(predictor)
    else:
        render_login_page()

if __name__ == '__main__':
    main()