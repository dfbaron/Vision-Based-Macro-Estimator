# scripts/app.py
import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, date
import yaml
import pandas as pd
import altair as alt
from PIL import Image
import configparser

# --- 1. Configuración de Paths e Importaciones ---
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.macro_estimator.training.predictor import Predictor
from src.macro_estimator.database_utils import Database

# --- 2. Inicialización y Carga de Recursos en Caché ---
DB_PATH = Path("data/app_data/user_data.db")
IMAGE_STORAGE_PATH = Path("data/app_data/uploaded_images")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
IMAGE_STORAGE_PATH.mkdir(parents=True, exist_ok=True)

db = Database(db_path=DB_PATH)

@st.cache_resource
def load_predictor(config_path="config/config.yaml") -> Predictor:
    print("--- Cargando el modelo... (esto solo se ejecuta una vez) ---")
    try:
        config = configparser.ConfigParser()
        config.read(config_path)
        return Predictor(config)
    except Exception as e:
        st.error(f"Error fatal al cargar el modelo: {e}")
        return None

# --- 3. Componentes de la Interfaz de Usuario ---

def render_login_page():
    """Muestra la página de login y registro."""
    st.title("Bienvenido a Macro Estimator AI 🍽️")
    login_tab, register_tab = st.tabs(["Iniciar Sesión", "Registrarse"])
    # ... (código de la función de login, sin cambios)
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
                if new_password == confirm_password:
                    if db.add_user(new_username, new_password):
                        st.success("¡Usuario creado con éxito! Por favor, inicia sesión.")
                    else:
                        st.error("El nombre de usuario ya existe o los campos están vacíos.")
                else:
                    st.error("Las contraseñas no coinciden.")


def render_dashboard():
    """Muestra el panel de control con el progreso diario y gráficos."""
    st.header(f"Panel de Control de Hoy: {date.today().strftime('%B %d, %Y')}")

    # Cargar datos de hoy
    goals = db.get_user_goals(st.session_state.user_id)
    meals_today = db.get_user_meals_df(st.session_state.user_id)
    if meals_today.empty:
        st.info("No hay comidas registradas para hoy.")
        return
    
    meals_today = meals_today[meals_today['timestamp'].dt.date == date.today()]
    
    # Calcular totales
    totals = meals_today[['calories', 'fat_grams', 'carb_grams', 'protein_grams']].sum()

    # Mostrar métricas y barras de progreso
    st.subheader("Progreso Diario")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Calorías", f"{totals.calories:.0f} / {goals['calories']:.0f} kcal")
        st.progress(min(totals.calories / goals['calories'], 1.0))
    # ... (métricas y barras de progreso para grasas, carbos y proteínas) ...
    with col2:
        st.metric("Grasas", f"{totals.fat_grams:.1f} / {goals['fat_grams']:.1f} g")
        st.progress(min(totals.fat_grams / goals['fat_grams'], 1.0))
    with col3:
        st.metric("Carbohidratos", f"{totals.carb_grams:.1f} / {goals['carb_grams']:.1f} g")
        st.progress(min(totals.carb_grams / goals['carb_grams'], 1.0))
    with col4:
        st.metric("Proteínas", f"{totals.protein_grams:.1f} / {goals['protein_grams']:.1f} g")
        st.progress(min(totals.protein_grams / goals['protein_grams'], 1.0))

    # Gráfico de desglose de macros del día
    if not totals.empty and totals.sum() > 0:
        macros_df = pd.DataFrame({
            'Macro': ['Grasas (kcal)', 'Carbos (kcal)', 'Proteína (kcal)'],
            'Calorías': [totals.fat_grams * 9, totals.carb_grams * 4, totals.protein_grams * 4]
        })
        pie_chart = alt.Chart(macros_df).mark_arc(innerRadius=50).encode(
            theta=alt.Theta(field="Calorías", type="quantitative"),
            color=alt.Color(field="Macro", type="nominal", scale=alt.Scale(scheme='viridis'))
        ).properties(title="Desglose Calórico por Macronutriente Hoy")
        st.altair_chart(pie_chart, use_container_width=True)

def render_add_meal_page(predictor):
    """Muestra la página para añadir una nueva comida."""
    st.header("Registrar una Nueva Comida")
    
    with st.form("add_meal_form"):
        description = st.text_input("Descripción de la comida (ej. 'Almuerzo - Ensalada de pollo')")
        uploaded_file = st.file_uploader("Sube una imagen de tu comida", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("Analizar y Registrar Comida")

        if submitted and uploaded_file is not None and description:
            with st.spinner('Analizando la imagen...'):
                image = Image.open(uploaded_file)
                image_bytes = uploaded_file.getvalue()
                prediction = predictor.predict_from_bytes(image_bytes)
                
                # Guardar imagen y registro en la DB
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f"{st.session_state.username}_{timestamp_str}_{uploaded_file.name}"
                image_path = IMAGE_STORAGE_PATH / image_filename
                with open(image_path, "wb") as f:
                    f.write(image_bytes)
                db.add_meal(st.session_state.user_id, datetime.now(), str(image_path), description, prediction, 'AI Scan')
                
                st.success("¡Comida registrada con éxito!")
                st.image(image, use_container_width=True)
                st.write(prediction)
        elif submitted:
            st.warning("Por favor, añade una descripción y sube una imagen.")

def render_history_page():
    """Muestra el historial completo de comidas y gráficos de tendencia."""
    st.header("Historial y Tendencias")
    user_history = db.get_user_meals_df(st.session_state.user_id)

    if user_history.empty:
        st.info("No hay comidas registradas en tu historial.")
        return

    # Gráfico de tendencia
    st.subheader("Tendencia de Ingesta Calórica")
    daily_calories = user_history.set_index('timestamp').resample('D')['calories'].sum().reset_index()
    line_chart = alt.Chart(daily_calories).mark_line(point=True).encode(
        x='timestamp:T',
        y='calories:Q'
    ).properties(title="Calorías por Día")
    st.altair_chart(line_chart, use_container_width=True)

    # Historial detallado
    st.subheader("Registro Completo")
    for _, entry in user_history.iterrows():
        with st.expander(f"{entry['description']} - {entry['timestamp'].strftime('%Y-%m-%d %H:%M')}"):
            # ... (código para mostrar los detalles de la comida, como en la versión anterior)
            pass

def render_settings_page():
    """Muestra la página de configuración de metas."""
    st.header("Configuración de Metas")
    goals = db.get_user_goals(st.session_state.user_id)
    with st.form("goals_form"):
        st.write("Establece tus metas diarias de macronutrientes y calorías.")
        calories = st.number_input("Calorías (kcal)", value=goals.get('calories', 2000))
        fat = st.number_input("Grasas (g)", value=goals.get('fat_grams', 70))
        carbs = st.number_input("Carbohidratos (g)", value=goals.get('carb_grams', 250))
        protein = st.number_input("Proteínas (g)", value=goals.get('protein_grams', 150))
        submitted = st.form_submit_button("Guardar Metas")
        if submitted:
            new_goals = {'calories': calories, 'fat_grams': fat, 'carb_grams': carbs, 'protein_grams': protein}
            db.update_user_goals(st.session_state.user_id, new_goals)
            st.success("¡Metas actualizadas con éxito!")

# --- 4. Flujo Principal de la Aplicación ---
def main():
    st.set_page_config(page_title="Macro Estimator AI", page_icon="🍽️", layout="wide")

    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        predictor = load_predictor()
        
        # Menú de navegación en la barra lateral
        st.sidebar.title("Navegación")
        page = st.sidebar.radio("Ir a", ["Panel de Control", "Registrar Comida", "Historial", "Configuración"])
        
        if st.sidebar.button("Cerrar Sesión"):
            st.session_state.logged_in = False
            st.experimental_rerun()

        if page == "Panel de Control":
            render_dashboard()
        elif page == "Registrar Comida":
            render_add_meal_page(predictor)
        elif page == "Historial":
            render_history_page()
        elif page == "Configuración":
            render_settings_page()
    else:
        render_login_page()

if __name__ == '__main__':
    main()