"""
Gestion SQLite pour stocker l'historique des estimations de prix
(dataset Craigslist — nouvelles colonnes : condition, drive, type, state).
"""

import sqlite3
from contextlib import contextmanager

DB_PATH = "data/cars.db"


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cars (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                manufacturer TEXT NOT NULL,
                year INTEGER NOT NULL,
                odometer INTEGER NOT NULL,
                condition TEXT,
                fuel TEXT NOT NULL,
                transmission TEXT NOT NULL,
                drive TEXT,
                type TEXT,
                state TEXT,
                model_used TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        conn.commit()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def insert_car(car: dict) -> int:
    with get_connection() as conn:
        cursor = conn.execute("""
            INSERT INTO cars
                (manufacturer, year, odometer, condition, fuel, transmission,
                 drive, type, state, model_used, predicted_price)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            car["manufacturer"], car["year"], car["odometer"],
            car["condition"], car["fuel"], car["transmission"],
            car["drive"], car["type"], car["state"],
            car["model_used"], car["predicted_price"],
        ))
        conn.commit()
        return cursor.lastrowid


def get_all_cars():
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM cars ORDER BY created_at DESC").fetchall()
        return [dict(row) for row in rows]


def get_dashboard_stats():
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM cars").fetchone()["c"]
        avg_price = conn.execute("SELECT AVG(predicted_price) AS a FROM cars").fetchone()["a"]
        by_manufacturer = conn.execute("""
            SELECT manufacturer, COUNT(*) AS count, AVG(predicted_price) AS avg_price
            FROM cars GROUP BY manufacturer ORDER BY count DESC
        """).fetchall()
        by_fuel = conn.execute("""
            SELECT fuel, COUNT(*) AS count FROM cars GROUP BY fuel
        """).fetchall()
        by_type = conn.execute("""
            SELECT type, COUNT(*) AS count FROM cars GROUP BY type ORDER BY count DESC
        """).fetchall()
        return {
            "total_predictions": total,
            "average_price": round(avg_price, 2) if avg_price else 0,
            "by_manufacturer": [dict(r) for r in by_manufacturer],
            "by_fuel": [dict(r) for r in by_fuel],
            "by_type": [dict(r) for r in by_type],
        }


def delete_car(car_id: int) -> bool:
    with get_connection() as conn:
        cursor = conn.execute("DELETE FROM cars WHERE id = ?", (car_id,))
        conn.commit()
        return cursor.rowcount > 0
