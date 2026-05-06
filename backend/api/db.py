import sqlite3
import time

conn = sqlite3.connect("detections.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS detections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    confidence REAL,
    source TEXT,
    time TEXT
)
""")

def save_detection(name, conf, source):
    cursor.execute(
        "INSERT INTO detections (name, confidence, source, time) VALUES (?, ?, ?, ?)",
        (name, conf, source, time.ctime())
    )
    conn.commit()