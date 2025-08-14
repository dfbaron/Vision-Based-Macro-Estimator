# src/macro_estimator/database_utils.py
import sqlite3
import hashlib
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

class Database:
    """
    Manages all database operations for the Streamlit app.
    """
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def _hash_password(self, password: str, salt: str) -> str:
        """Hashes a password with a given salt (username)."""
        return hashlib.sha256((password + salt).encode()).hexdigest()

    def create_tables(self):
        """Creates the necessary tables if they don't exist."""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL
            );
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                image_path TEXT NOT NULL,
                calories REAL NOT NULL,
                fat_grams REAL NOT NULL,
                carb_grams REAL NOT NULL,
                protein_grams REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            );
        """)
        self.conn.commit()

    def add_user(self, username: str, password: str) -> bool:
        """Adds a new user to the database. Returns True on success."""
        if not username or not password:
            return False
        password_hash = self._hash_password(password, username)
        try:
            self.cursor.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, password_hash))
            self.conn.commit()
            return True
        except sqlite3.IntegrityError: # Username already exists
            return False

    def check_user(self, username: str, password: str) -> Optional[int]:
        """Checks if a user exists and the password is correct. Returns user_id if successful."""
        self.cursor.execute("SELECT id, password_hash FROM users WHERE username = ?", (username,))
        result = self.cursor.fetchone()
        if result:
            user_id, stored_hash = result
            if self._hash_password(password, username) == stored_hash:
                return user_id
        return None

    def add_meal(self, user_id: int, timestamp: str, image_path: str, prediction: dict):
        """Adds a meal record to the database."""
        self.cursor.execute("""
            INSERT INTO meals (user_id, timestamp, image_path, calories, fat_grams, carb_grams, protein_grams)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id, timestamp, image_path,
            prediction['calories'], prediction['fat_grams'],
            prediction['carb_grams'], prediction['protein_grams']
        ))
        self.conn.commit()

    def get_user_meals(self, user_id: int) -> pd.DataFrame:
        """Retrieves all meals for a user as a pandas DataFrame."""
        df = pd.read_sql_query(f"SELECT * FROM meals WHERE user_id = {user_id} ORDER BY timestamp DESC", self.conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df