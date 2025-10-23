# ==========================================================
# 🧠 user_db.py — User Table Setup & Management (with Department)
# ==========================================================
import sqlite3

DB_NAME = "SCAPS.db"

def init_user_table():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            role TEXT NOT NULL,
            department TEXT
        )
    """)
    # Insert default admin if not exists
    cursor.execute("SELECT * FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute("INSERT INTO users (username, password, role, department) VALUES (?, ?, ?, ?)",
                       ('admin', 'admin123', 'admin', 'All Departments'))
        print("✅ Default admin user created (username: admin, password: admin123)")
    conn.commit()
    conn.close()


def add_user(username, password, role, department=None):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO users (username, password, role, department) VALUES (?, ?, ?, ?)",
                       (username, password, role, department))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def verify_user(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT role, department FROM users WHERE username=? AND password=?", (username, password))
    result = cursor.fetchone()
    conn.close()
    if result:
        role, department = result
        return {"role": role, "department": department}
    return None
