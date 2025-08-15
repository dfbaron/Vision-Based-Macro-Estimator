# scripts/app.py
import streamlit as st
from pathlib import Path
import sys
from datetime import datetime, date, timedelta
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
    """
    Displays a user-friendly login and registration page that can be toggled.
    """
    # --- 1. Inicializar el estado de la vista (login o registro) ---
    if 'login_view' not in st.session_state:
        st.session_state.login_view = "Log In"

    st.title("Welcome to Vision Macro Estimator 🍽️")
    
    # --- 2. Lógica para cambiar entre vistas ---
    def switch_to_register():
        st.session_state.login_view = "Sign Up"
    
    def switch_to_login():
        st.session_state.login_view = "Log In"

    # --- Vista de Iniciar Sesión ---
    if st.session_state.login_view == "Log In":
        st.subheader("Log In to Your Account")
        with st.form("login_form"):
            username = st.text_input("Username", key="login_user", placeholder="Enter your username")
            password = st.text_input("Password", type="password", key="login_pass", placeholder="Enter your password")
            
            submitted = st.form_submit_button("➡️ Log In", type="primary", use_container_width=True)
            
            if submitted:
                if not username or not password:
                    st.warning("Please enter both username and password.")
                else:
                    user_id = db.check_user(username, password)
                    if user_id:
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_id = user_id
                        st.success("Logged in successfully! Redirecting...")
                        st.rerun()
                    else:
                        st.error("Incorrect username or password. Please try again.")

        # Botón para cambiar a la vista de registro
        st.markdown("---")
        st.write("Don't have an account yet?")
        # Usamos `on_click` para llamar a nuestra función que cambia el estado
        st.button("📝 Sign Up Here", on_click=switch_to_register, use_container_width=True)

    # --- Vista de Registrarse ---
    elif st.session_state.login_view == "Sign Up":
        st.subheader("Create a New Account")
        with st.form("register_form"):
            new_username = st.text_input("Choose a Username", key="reg_user", placeholder="e.g., nutrition_ninja")
            new_password = st.text_input("Create a Password", type="password", key="reg_pass", placeholder="Must be at least 6 characters")
            confirm_password = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
            
            reg_submitted = st.form_submit_button("🚀 Create Account", type="primary", use_container_width=True)
            
            if reg_submitted:
                if not all([new_username, new_password, confirm_password]):
                    st.warning("Please fill out all fields.")
                elif len(new_password) < 6:
                    st.warning("Password must be at least 6 characters long.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match. Please re-enter.")
                else:
                    if db.add_user(new_username, new_password):
                        st.success("Account created successfully! Switching to Log In...")
                        # Cambiar automáticamente a la vista de login después de un registro exitoso
                        switch_to_login()
                        st.rerun() # Opcional: recargar para limpiar el formulario
                    else:
                        st.error("That username is already taken. Please choose another one.")
        
        # Botón para volver a la vista de login
        st.markdown("---")
        st.write("Already have an account?")
        st.button("🔐 Log In Instead", on_click=switch_to_login, use_container_width=True)


def render_dashboard():
    """
    Displays a beautiful and informative daily dashboard with corrected metric labels.
    """
    st.header(f"Today's Dashboard: {date.today().strftime('%B %d, %Y')} 📅")

    # --- 1. Cargar Datos y Metas del Día ---
    user_id = st.session_state.user_id
    goals = db.get_user_goals(user_id)
    all_meals = db.get_user_meals_df(user_id)
    
    if all_meals.empty:
        st.info("👋 You haven't logged any meals yet today. Go to the 'Log a Meal' page to get started!")
        st.subheader("Your Daily Goals")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories", f"{goals.get('calories', 0):.0f} kcal")
        col2.metric("Carbs", f"{goals.get('carb_grams', 0):.1f} g")
        col3.metric("Fat", f"{goals.get('fat_grams', 0):.1f} g")
        col4.metric("Protein", f"{goals.get('protein_grams', 0):.1f} g")
        st.stop()

    meals_today = all_meals[all_meals['timestamp'].dt.date == date.today()]
    
    # --- 2. Calcular Totales y Restantes ---
    totals = meals_today[['calories', 'fat_grams', 'carb_grams', 'protein_grams']].sum()
    remaining = {
        'calories': goals['calories'] - totals.calories,
        'fat': goals['fat_grams'] - totals.fat_grams,
        'carbs': goals['carb_grams'] - totals.carb_grams,
        'protein': goals['protein_grams'] - totals.protein_grams
    }

    # --- 3. Mostrar Métricas de Progreso y Restantes (con la corrección) ---
    st.subheader("Daily Progress")
    
    tab1, tab2 = st.tabs(["📊 Progress Overview", "🔢 Remaining Macros"])

    with tab1:
        # --- CÓDIGO CORREGIDO ---
        # Ahora pasamos el título directamente como la etiqueta de st.metric
        # y eliminamos el st.write() redundante.
        col1, col2, col3, col4 = st.columns(4)
        
        col1.metric(
            label="Calories", 
            value=f"{totals.calories:.0f} / {goals['calories']:.0f} kcal"
        )
        col1.progress(min(totals.calories / goals['calories'], 1.0))

        col2.metric(
            label="Fat", 
            value=f"{totals.fat_grams:.1f} / {goals['fat_grams']:.1f} g"
        )
        col2.progress(min(totals.fat_grams / goals['fat_grams'], 1.0))

        col3.metric(
            label="Carbs", 
            value=f"{totals.carb_grams:.1f} / {goals['carb_grams']:.1f} g"
        )
        col3.progress(min(totals.carb_grams / goals['carb_grams'], 1.0))
        
        col4.metric(
            label="Protein", 
            value=f"{totals.protein_grams:.1f} / {goals['protein_grams']:.1f} g"
        )
        col4.progress(min(totals.protein_grams / goals['protein_grams'], 1.0))
    
    with tab2:
        # ... (esta sección ya era correcta, no necesita cambios)
        st.info("This is how much you have left to meet your daily goals.")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Calories Left", f"{remaining['calories']:.0f} kcal")
        col2.metric("Carbs Left", f"{remaining['carbs']:.1f} g")
        col3.metric("Fat Left", f"{remaining['fat']:.1f} g")
        col4.metric("Protein Left", f"{remaining['protein']:.1f} g")

    st.markdown("---") # Separador

    # --- 5. Visualizaciones y Tabla de Comidas ---
    col1, col2 = st.columns([1, 1]) # Dividir el espacio en dos

    with col1:
        st.subheader("Caloric Breakdown")
        # Gráfico de Donut con Tooltips
        if not totals.empty and totals.sum() > 0:
            macros_df = pd.DataFrame({
                'Macro': ['Fat (kcal)', 'Carbs (kcal)', 'Protein (kcal)'],
                'Calories': [totals.fat_grams * 9, totals.carb_grams * 4, totals.protein_grams * 4]
            })
            donut_chart = alt.Chart(macros_df).mark_arc(innerRadius=70, outerRadius=110).encode(
                theta=alt.Theta(field="Calories", type="quantitative"),
                color=alt.Color(field="Macro", type="nominal", 
                                scale=alt.Scale(scheme='viridis'),
                                legend=alt.Legend(title="Macronutrient")),
                tooltip=['Macro', 'Calories'] # <-- Tooltip interactivo
            ).properties(title="Caloric Source for Today")
            st.altair_chart(donut_chart, use_container_width=True)

        # Tracker de Hidratación
        st.subheader("💧 Hydration Tracker")
        if 'water_intake' not in st.session_state:
            st.session_state.water_intake = 0
        
        water_goal = 8 # Vasos
        water_col1, water_col2, water_col3 = st.columns([2,1,1])
        with water_col1:
            st.progress(min(st.session_state.water_intake / water_goal, 1.0))
        with water_col2:
            if st.button("Add Glass (+1)"):
                st.session_state.water_intake += 1
                st.rerun()
        with water_col3:
            if st.button("Remove (-1)"):
                st.session_state.water_intake = max(0, st.session_state.water_intake - 1)
                st.rerun()
        st.metric("Water Intake", f"{st.session_state.water_intake} / {water_goal} glasses")


    with col2:
        st.subheader("Today's Meals")
        # Formatear el DataFrame para mostrarlo
        display_df = meals_today[['timestamp', 'description', 'calories', 'protein_grams']].copy()
        display_df.rename(columns={
            'timestamp': 'Time',
            'description': 'Description',
            'calories': 'Calories (kcal)',
            'protein_grams': 'Protein (g)'
        }, inplace=True)
        
        display_df['Time'] = display_df['Time'].dt.strftime('%I:%M %p') # Formato de 12 horas
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn(width="small"),
                "Description": st.column_config.TextColumn(width="medium"),
                "Calories (kcal)": st.column_config.NumberColumn(format="%.0f"),
                "Protein (g)": st.column_config.NumberColumn(format="%.1f")
            }
        )

def render_add_meal_page(predictor):
    """
    Displays a beautiful and user-friendly page for adding a new meal.
    """
    st.header("Log a New Meal 📸")
    st.markdown("Use our AI to analyze an image of your food, or add a meal manually.")

    # --- Usar st.session_state para que los resultados no desaparezcan ---
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
        st.session_state.last_image = None

    # --- 1. Diseño de Dos Columnas ---
    col1, col2 = st.columns([1, 1]) # Columnas de igual tamaño

    # --- Columna Izquierda: Entrada del Usuario ---
    with col1:
        st.subheader("1. Describe & Upload Your Meal")
        with st.form("add_meal_form"):
            description = st.text_input(
                "Meal Description", 
                placeholder="e.g., 'Lunch - Chicken Salad'"
            )
            uploaded_file = st.file_uploader(
                "Upload an image of your meal", 
                type=["jpg", "jpeg", "png"]
            )
            
            submitted = st.form_submit_button(
                "✨ Analyze & Log Meal", 
                type="primary", 
                use_container_width=True
            )

            if submitted:
                if not description or not uploaded_file:
                    st.warning("Please provide a description and upload an image.")
                else:
                    with st.spinner('AI is analyzing the image...'):
                        image = Image.open(uploaded_file)
                        image_bytes = uploaded_file.getvalue()
                        
                        # --- Lógica de Predicción y Guardado ---
                        prediction = predictor.predict_from_bytes(image_bytes)
                        
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_filename = f"{st.session_state.username}_{timestamp_str}_{uploaded_file.name}"
                        image_path = IMAGE_STORAGE_PATH / image_filename
                        with open(image_path, "wb") as f:
                            f.write(image_bytes)
                        
                        db.add_meal(
                            st.session_state.user_id, 
                            datetime.now(), 
                            str(image_path), 
                            description, 
                            prediction, 
                            'AI Scan'
                        )
                        
                        # Guardar el resultado en el estado de la sesión para mostrarlo en la otra columna
                        st.session_state.last_prediction = prediction
                        st.session_state.last_image = image
                        
                        st.success("Meal logged successfully!")
                        # No es necesario st.rerun() aquí, Streamlit actualizará la otra columna automáticamente

    # --- Columna Derecha: Salida y Resultados ---
    with col2:
        st.subheader("2. AI Analysis Results")
        
        if st.session_state.last_prediction:
            prediction = st.session_state.last_prediction
            image = st.session_state.last_image

            st.image(image, caption="Analyzed Image", use_container_width=True)
            
            # Mostrar métricas
            metric_cols = st.columns(4)
            metric_cols[0].metric("Calories", f"{prediction['calories']:.0f} kcal")
            metric_cols[1].metric("Carbs", f"{prediction['carb_grams']:.1f} g")
            metric_cols[2].metric("Fat", f"{prediction['fat_grams']:.1f} g")
            metric_cols[3].metric("Protein", f"{prediction['protein_grams']:.1f} g")

            # Gráfico de Donut
            macros_df = pd.DataFrame({
                'Macro': ['Fat (g)', 'Carbs (g)', 'Protein (g)'],
                'Grams': [prediction['fat_grams'], prediction['carb_grams'], prediction['protein_grams']]
            })
            donut_chart = alt.Chart(macros_df).mark_arc(innerRadius=50).encode(
                theta=alt.Theta(field="Grams", type="quantitative"),
                color=alt.Color(field="Macro", type="nominal", scale=alt.Scale(scheme='viridis')),
                tooltip=['Macro', 'Grams']
            ).properties(title="Macronutrient Breakdown")
            st.altair_chart(donut_chart, use_container_width=True)

            # Botón para limpiar los resultados y registrar otra comida
            if st.button("Log Another Meal", use_container_width=True):
                st.session_state.last_prediction = None
                st.session_state.last_image = None
                st.rerun()
        else:
            st.info("Your meal analysis will appear here once you upload an image.")

def render_history_page():
    """
    Displays a comprehensive and interactive history of logged meals with charts and stats.
    """
    st.header("Meal History & Trends 📚")
    
    # --- 1. Cargar el Historial Completo del Usuario ---
    user_history = db.get_user_meals_df(st.session_state.user_id)

    if user_history.empty:
        st.info("👋 You haven't logged any meals yet. Go to the 'Log a Meal' page to get started!")
        st.stop() # Detener la ejecución si no hay historial

    # --- 2. Filtro de Rango de Fechas ---
    st.subheader("Filter Your History")
    
    date_option = st.selectbox(
        "Select a time range:",
        ("Last 7 Days", "Last 30 Days", "This Month", "All Time"),
        key="history_date_range"
    )

    # Filtrar el DataFrame según la opción seleccionada
    today = datetime.now().date()
    if date_option == "Last 7 Days":
        start_date = today - timedelta(days=7)
        filtered_history = user_history[user_history['timestamp'].dt.date >= start_date]
    elif date_option == "Last 30 Days":
        start_date = today - timedelta(days=30)
        filtered_history = user_history[user_history['timestamp'].dt.date >= start_date]
    elif date_option == "This Month":
        start_date = today.replace(day=1)
        filtered_history = user_history[user_history['timestamp'].dt.date >= start_date]
    else: # All Time
        filtered_history = user_history
    
    if filtered_history.empty:
        st.warning(f"No meals found for the selected period: '{date_option}'.")
        st.stop()

    # --- 3. Mostrar KPIs y Estadísticas del Período ---
    st.subheader(f"Summary for '{date_option}'")
    
    # Calcular promedios diarios
    num_days = (filtered_history['timestamp'].dt.date.max() - filtered_history['timestamp'].dt.date.min()).days + 1
    avg_calories = filtered_history['calories'].sum() / num_days
    avg_protein = filtered_history['protein_grams'].sum() / num_days
    
    kpi_cols = st.columns(3)
    kpi_cols[0].metric("Total Meals Logged", f"{len(filtered_history)}")
    kpi_cols[1].metric("Avg. Daily Calories", f"{avg_calories:.0f} kcal")
    kpi_cols[2].metric("Avg. Daily Protein", f"{avg_protein:.1f} g")

    st.markdown("---")

    # --- 4. Gráfico de Tendencias Mejorado ---
    st.subheader("Daily Intake Trends")
    
    # Agrupar por día y sumar los macros
    daily_summary = filtered_history.set_index('timestamp').resample('D').sum(numeric_only=True).reset_index()
    
    # Transformar los datos a formato "largo" para un gráfico más fácil con Altair
    daily_summary_long = daily_summary.melt(
        id_vars=['timestamp'], 
        value_vars=['calories', 'fat_grams', 'carb_grams', 'protein_grams'],
        var_name='Nutrient',
        value_name='Amount'
    )
    
    # Crear el gráfico de líneas interactivo
    trend_chart = alt.Chart(daily_summary_long).mark_line(point=True).encode(
        x=alt.X('timestamp:T', title='Date'),
        y=alt.Y('Amount:Q', title='Amount'),
        color=alt.Color('Nutrient:N', legend=alt.Legend(title='Nutrient')),
        tooltip=['timestamp:T', 'Nutrient:N', 'Amount:Q']
    ).properties(
        title="Your Daily Nutrient Intake Over Time"
    ).interactive() # <-- Hace el gráfico interactivo (zoom, pan)
    
    st.altair_chart(trend_chart, use_container_width=True)
    
    st.markdown("---")

    # --- 5. Historial Detallado de Comidas ---
    st.subheader("Detailed Meal Log")
    
    for _, entry in filtered_history.iterrows():
        # Crear un expander para cada comida
        with st.expander(f"**{entry.get('description', 'Meal')}** - {entry['timestamp'].strftime('%Y-%m-%d %I:%M %p')}"):
            col1, col2 = st.columns([1, 2])
            
            # Columna de la imagen
            with col1:
                if entry.get('image_path') and Path(entry['image_path']).exists():
                    st.image(str(entry['image_path']), use_container_width=True, caption=f"Source: {entry.get('source', 'N/A')}")
                else:
                    st.info("No image available for this entry.")
            
            # Columna de los detalles
            with col2:
                st.metric("Calories", f"{entry['calories']:.0f} kcal")
                st.text(f"Fat: {entry['fat_grams']:.1f} g")
                st.text(f"Carbs: {entry['carb_grams']:.1f} g")
                st.text(f"Protein: {entry['protein_grams']:.1f} g")

def render_settings_page():
    """
    Displays a beautiful and user-friendly page for setting nutritional goals.
    """
    # --- 1. Page Header ---
    st.header("⚙️ Goal Settings")
    st.markdown("Customize your daily nutritional targets to align with your personal health and fitness goals.")

    # --- 2. Cargar las Metas Actuales del Usuario ---
    goals = db.get_user_goals(st.session_state.user_id)
    
    # --- 3. Diseño de Dos Columnas ---
    col1, col2 = st.columns([1, 1.5]) # La columna del formulario es un poco más ancha

    # --- Columna Izquierda: Mostrar Metas Actuales ---
    with col1:
        st.subheader("Your Current Goals")
        
        # Usar un contenedor para agrupar visualmente las métricas
        with st.container(border=True):
            st.metric("🎯 Calories", f"{goals.get('calories', 0):.0f} kcal")
            st.metric("🥑 Fat", f"{goals.get('fat_grams', 0):.1f} g")
            st.metric("🍞 Carbs", f"{goals.get('carb_grams', 0):.1f} g")
            st.metric("🍗 Protein", f"{goals.get('protein_grams', 0):.1f} g")
        
        st.info("Update your goals in the form on the right. Your dashboard will reflect these new targets.")

    # --- Columna Derecha: Formulario para Actualizar Metas ---
    with col2:
        st.subheader("Update Your Goals")
        
        with st.form("goals_form"):
            st.write("Enter your new daily targets below:")
            
            # Usar columnas dentro del formulario para una mejor alineación
            form_col1, form_col2 = st.columns(2)
            
            with form_col1:
                calories = st.number_input(
                    "Calories (kcal)", 
                    min_value=0, 
                    value=int(goals.get('calories', 2000)), 
                    step=50
                )
                fat = st.number_input(
                    "Fat (g)", 
                    min_value=0.0, 
                    value=float(goals.get('fat_grams', 70)), 
                    step=1.0, 
                    format="%.1f"
                )

            with form_col2:
                carbs = st.number_input(
                    "Carbohydrates (g)", 
                    min_value=0.0, 
                    value=float(goals.get('carb_grams', 250)), 
                    step=5.0, 
                    format="%.1f"
                )
                protein = st.number_input(
                    "Protein (g)", 
                    min_value=0.0, 
                    value=float(goals.get('protein_grams', 150)), 
                    step=5.0, 
                    format="%.1f"
                )
            
            submitted = st.form_submit_button("💾 Save Changes", type="primary", use_container_width=True)
            
            if submitted:
                new_goals = {
                    'calories': calories, 
                    'fat_grams': fat, 
                    'carb_grams': carbs, 
                    'protein_grams': protein
                }
                db.update_user_goals(st.session_state.user_id, new_goals)
                st.success("Your goals have been updated successfully!")                
                # Opcional: st.rerun() para ver el cambio reflejado inmediatamente en la columna izquierda
                st.rerun()

def render_sidebar():
    """
    Renders the sidebar navigation and user information.
    Returns the selected page.
    """
    with st.sidebar:
        # --- 1. User Information and Logout Button ---
        st.subheader(f"Welcome, {st.session_state.username}! 👋")
        
        # Usamos st.container para agrupar y aplicar estilos si quisiéramos
        with st.container(border=True):
            st.info("Your AI assistant for effortless nutritional tracking.")
        
        st.title("Navigation")

        # --- 2. Navigation Menu with Icons ---
        # st.radio es excelente para esto. 'options' son los nombres que ve el usuario.
        # 'format_func' es una forma elegante de añadir iconos.
        page = st.radio(
            "Go to", 
            options=["Dashboard", "Log a Meal", "History", "Settings"],
            label_visibility="collapsed", # Oculta la etiqueta "Go to" para un look más limpio
            format_func=lambda x: {
                "Dashboard": "📊 Dashboard",
                "Log a Meal": "📸 Log a Meal",
                "History": "📚 History",
                "Settings": "⚙️ Settings"
            }.get(x)
        )
        
        st.markdown("---") # Un separador visual

        # --- 3. Logout Button ---
        # El botón de logout al final, centrado y con color.
        if st.button("Log Out", use_container_width=True):
            # Limpiar todo el estado de la sesión
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
            
        return page

def render_main_content(page: str, predictor):
    """
    Renders the main content area based on the page selected in the sidebar.
    """

    # Enrutamiento de la página
    if page == "Dashboard":
        render_dashboard()
    elif page == "Log a Meal":
        render_add_meal_page(predictor)
    elif page == "History":
        render_history_page()
    elif page == "Settings":
        render_settings_page()

def main():
    """
    Main function to orchestrate the Streamlit application flow.
    """
    # --- 1. Page Configuration ---
    st.set_page_config(
        page_title="Macro Estimator AI", 
        page_icon="🍽️", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # --- 2. Session State Initialization ---
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # --- 3. Main Application Logic ---
    if st.session_state.logged_in:
        # Cargar el modelo solo si el usuario ha iniciado sesión
        predictor = load_predictor()
        
        # Renderizar la barra lateral y obtener la página seleccionada
        selected_page = render_sidebar()
        
        # Renderizar el contenido principal
        render_main_content(selected_page, predictor)
    else:
        # Mostrar la página de login si no ha iniciado sesión
        render_login_page()

if __name__ == '__main__':
    main()