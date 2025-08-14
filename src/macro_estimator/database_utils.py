# src/macro_estimator/database_utils.py
import sqlite3
import hashlib
from pathlib import Path
import pandas as pd

class Database:
    """
    Gestiona todas las operaciones de la base de datos para la aplicación Macro Estimator.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        # check_same_thread=False es necesario para que Streamlit pueda acceder a la DB
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _hash_password(self, password: str, salt: str) -> str:
        """Hashea una contraseña con una sal (usamos el nombre de usuario como sal)."""
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def _create_tables(self):
        """Crea las tablas de la base de datos si no existen."""
        # Tabla de usuarios
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
        """)
        # Tabla para las metas del usuario
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_goals (
                user_id INTEGER PRIMARY KEY,
                calories REAL DEFAULT 2000,
                fat_grams REAL DEFAULT 70,
                carb_grams REAL DEFAULT 250,
                protein_grams REAL DEFAULT 150,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        # Tabla para el historial de comidas
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT,
                description TEXT,
                calories REAL NOT NULL,
                fat_grams REAL NOT NULL,
                carb_grams REAL NOT NULL,
                protein_grams REAL NOT NULL,
                source TEXT, -- 'AI Scan', 'Manual', 'Favorite'
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        self.conn.commit()

    def add_user(self, username: str, password: str) -> bool:
        """Añade un nuevo usuario y sus metas por defecto. Devuelve True si tiene éxito."""
        if not username or not password: return False
        password_hash = self._hash_password(password, username)
        try:
            self.cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            user_id = self.cursor.lastrowid
            self.cursor.execute("INSERT INTO user_goals (user_id) VALUES (?)", (user_id,))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: # El usuario ya existe
            return False

    def check_user(self, username: str, password: str) -> int | None:
        """Verifica las credenciales. Devuelve el user_id si son correctas."""
        self.cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        if result and self._hash_password(password, username) == result[1]:
            return result[0]
        return None

    def get_user_goals(self, user_id: int) -> dict:
        """Obtiene las metas nutricionales de un usuario."""
        self.cursor.execute("SELECT calories, fat_grams, carb_grams, protein_grams FROM user_goals WHERE user_id = ?", (user_id,))
        goals = self.cursor.fetchone()
        if goals:
            return {"calories": goals[0], "fat_grams": goals[1], "carb_grams": goals[2], "protein_grams": goals[3]}
        return {}

    def update_user_goals(self, user_id: int, goals: dict):
        """Actualiza las metas de un usuario."""
        self.cursor.execute("""
            UPDATE user_goals 
            SET calories = ?, fat_grams = ?, carb_grams = ?, protein_grams = ? 
            WHERE user_id = ?
        """, (goals['calories'], goals['fat_grams'], goals['carb_grams'], goals['protein_grams'], user_id))
        self.conn.commit()

    def add_meal(self, user_id: int, timestamp: str, image_path: str, description: str, prediction: dict, source: str):
        """Añade una comida al historial."""
        self.cursor.execute("""
            INSERT INTO meal_history (user_id, timestamp, image_path, description, calories, fat_grams, carb_grams, protein_grams, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, timestamp, image_path, description, prediction['calories'],
            prediction['fat_grams'], prediction['carb_grams'], prediction['protein_grams'], source
        ))
        self.conn.commit()
        
    def get_user_meals_df(self, user_id: int) -> pd.DataFrame:
        """Obtiene el historial de comidas de un usuario como un DataFrame."""
        df = pd.read_sql_query(f"SELECT * FROM meal_history WHERE user_id = {user_id} ORDER BY timestamp DESC", self.conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df