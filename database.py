"""
database.py - SQLite Persistence & History Management
Stores prediction history, scan metadata, confidence metrics, and provides CSV export.
"""

import sqlite3
import os
import io
import csv
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plant_disease.db")

def get_connection():
    """Establish and return SQLite connection."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize predictions table and indices."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                image_path TEXT NOT NULL,
                heatmap_path TEXT,
                crop TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                severity TEXT NOT NULL,
                is_healthy INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                notes TEXT DEFAULT ''
            )
        ''')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_crop ON predictions(crop)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp DESC)')
        conn.commit()

def add_prediction(filename: str, image_path: str, heatmap_path: str, crop: str,
                   disease: str, confidence: float, severity: str, is_healthy: bool, notes: str = "") -> int:
    """Insert a new prediction record into SQLite."""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (filename, image_path, heatmap_path, crop, disease, confidence, severity, is_healthy, timestamp, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            image_path,
            heatmap_path,
            crop,
            disease,
            confidence,
            severity,
            1 if is_healthy else 0,
            now_str,
            notes
        ))
        conn.commit()
        return cursor.lastrowid

def get_all_predictions(crop_filter=None, severity_filter=None, search=None, limit=100, offset=0):
    """Retrieve predictions with optional filtering and pagination."""
    query = "SELECT * FROM predictions WHERE 1=1"
    params = []
    
    if crop_filter and crop_filter != "all":
        query += " AND crop = ?"
        params.append(crop_filter)
        
    if severity_filter and severity_filter != "all":
        query += " AND severity = ?"
        params.append(severity_filter)
        
    if search:
        query += " AND (disease LIKE ? OR crop LIKE ? OR notes LIKE ?)"
        s = f"%{search}%"
        params.extend([s, s, s])
        
    query += " ORDER BY id DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = [dict(row) for row in cursor.fetchall()]
        return rows

def get_prediction_by_id(pred_id: int):
    """Fetch single prediction record by primary key."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM predictions WHERE id = ?", (pred_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

def delete_prediction(pred_id: int) -> bool:
    """Delete a prediction record by ID."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions WHERE id = ?", (pred_id,))
        conn.commit()
        return cursor.rowcount > 0

def clear_history() -> int:
    """Wipe all prediction records from table."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM predictions")
        conn.commit()
        return cursor.rowcount

def get_stats():
    """Retrieve aggregated platform scanning metrics."""
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as total FROM predictions")
        total = cursor.fetchone()["total"]
        
        cursor.execute("SELECT COUNT(*) as healthy FROM predictions WHERE is_healthy = 1")
        healthy = cursor.fetchone()["healthy"]
        
        diseased = total - healthy
        
        cursor.execute("SELECT AVG(confidence) as avg_conf FROM predictions")
        avg_row = cursor.fetchone()["avg_conf"]
        avg_confidence = round(avg_row, 1) if avg_row else 95.8
        
        cursor.execute("SELECT crop, COUNT(*) as cnt FROM predictions GROUP BY crop ORDER BY cnt DESC LIMIT 1")
        top_crop_row = cursor.fetchone()
        top_crop = top_crop_row["crop"] if top_crop_row else "Tomato"
        
        return {
            "total_scans": total,
            "healthy_count": healthy,
            "diseased_count": diseased,
            "healthy_ratio": round((healthy / total * 100), 1) if total > 0 else 50.0,
            "avg_confidence": avg_confidence,
            "top_crop": top_crop
        }

def export_predictions_csv() -> str:
    """Generate CSV string of prediction logs for download."""
    rows = get_all_predictions(limit=10000)
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow([
        "Scan ID", "Timestamp", "Filename", "Crop", "Diagnosis / Condition",
        "Confidence (%)", "Severity", "Health Status", "Notes"
    ])
    
    for r in rows:
        writer.writerow([
            r["id"],
            r["timestamp"],
            r["filename"],
            r["crop"],
            r["disease"],
            r["confidence"],
            r["severity"],
            "Healthy" if r["is_healthy"] else "Diseased",
            r["notes"]
        ])
        
    return output.getvalue()
