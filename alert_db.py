import sqlite3

def init_db():
    conn = sqlite3.connect("alerts.db")
    cursor = conn.cursor()
    # Added clip_url TEXT to the schema to match main_app.py
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            item_name TEXT,
            person_id INTEGER,
            camera_id TEXT,
            location TEXT,
            clip_url TEXT,
            is_checked BOOLEAN DEFAULT 0
        )
    ''')
    conn.commit()
    conn.close()
    print("Database initialized with all required columns.")

if __name__ == "__main__":
    init_db()