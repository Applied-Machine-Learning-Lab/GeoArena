import json
from datetime import datetime
import config
import os

RELEASE_NOTES_FILE = "release_notes.json"

def load_release_notes():
    try:
        if os.path.exists(RELEASE_NOTES_FILE):
            with open(RELEASE_NOTES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        # 如果文件不存在，返回默认值
        return {
            "last_updated": datetime.now().isoformat(),
            "notes": []
        }
    except Exception as e:
        print(f"Error loading release notes: {str(e)}")
        return {
            "last_updated": datetime.now().isoformat(),
            "notes": []
        }

def get_release_notes_html():
    """Generate HTML display of release notes."""
    notes_data = load_release_notes()
    
    html = f"""
    <style>
        .release-notes {{
            font-family: Arial, sans-serif;
            max-width: 800px;
            margin: 0 auto;
        }}
        .release-note {{
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        .note-date {{
            font-size: 0.9em;
            color: #888;
            margin-bottom: 8px;
        }}
        .note-content {{
            white-space: pre-wrap;
            line-height: 1.6;
        }}
        .last-updated {{
            font-size: 0.8em;
            color: #666;
            text-align: right;
            margin-top: 20px;
            font-style: italic;
        }}
    </style>
    <div class="release-notes">
    """
    
    # Add notes in reverse chronological order
    for note in sorted(notes_data["notes"], key=lambda x: x["date"], reverse=True):
        html += f"""
        <div class="release-note">
            <div class="note-date">📅 {note["date"]}</div>
            <div class="note-content">{note["content"]}</div>
        </div>
        """
    
    html += f"""
        <div class="last-updated">
            Last updated: {notes_data["last_updated"].split("T")[0]}
        </div>
    </div>
    """
    
    return html 