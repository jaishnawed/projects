import sqlite3
from flask import g

DATABASE = 'users.db'

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
    return db

def init_db():
    with app.app_context():
        db = get_db()
        cursor = db.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL
            )
        ''')
        # Add default user
        try:
            cursor.execute("INSERT INTO users (username, password) VALUES (?, ?)", 
                          ('wizards', 'ishukant'))
            db.commit()
        except sqlite3.IntegrityError:
            pass  # User already exists