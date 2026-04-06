import json
from flask import Flask, render_template, jsonify, send_from_directory
from datetime import datetime
import platform
import subprocess
import os 

app = Flask(__name__)

def get_sorted_data():
    try:
        with open('data.json', 'r') as f:
            detections = json.load(f)
        # Sort by timestamp (newest first)
        detections.sort(key=lambda x: datetime.strptime(x['timestamp'], "%Y-%m-%d %H:%M:%S"), reverse=True)
        return detections
    except Exception:
        return []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/data')
def get_data():
    return jsonify(get_sorted_data())

# In app.py
VIDEO_FOLDER = 'abondant_objects_clips'

@app.route('/abondant_objects_clips/<path:filename>')
def serve_video(filename):
    # This must match the folder name on your hard drive
    return send_from_directory(VIDEO_FOLDER, filename)


@app.route('/open_video/<path:filename>')
def open_video(filename):
    # Construct the full path to the video file
    video_path = os.path.join('abondant_objects_clips', filename)
    
    # Check if the file exists
    if os.path.exists(video_path):
        # Open with default Windows application
        if platform.system() == "Windows":
            os.startfile(video_path)
        else:
            # Fallback for macOS/Linux if needed
            opener = "open" if platform.system() == "Darwin" else "xdg-open"
            subprocess.call([opener, video_path])
            
        return jsonify({"status": "success", "message": f"Opening {filename}"})
    else:
        return jsonify({"status": "error", "message": "File not found"}), 404
    
@app.route('/api/delete/<path:filename>', methods=['POST'])
def delete_footage(filename):
    try:
        # 1. Physical File Deletion
        file_path = os.path.join('abondant_objects_clips', filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"File {filename} deleted from disk.")

        # 2. JSON Entry Deletion
        if os.path.exists('data.json'):
            with open('data.json', 'r') as f:
                data = json.load(f)
            
            # Filter out the deleted entry
            updated_data = [item for item in data if item['video_link'] != filename]
            
            with open('data.json', 'w') as f:
                json.dump(updated_data, f, indent=2)
            
            return jsonify({"status": "success", "message": f"Deleted {filename}"})
        
        return jsonify({"status": "error", "message": "JSON file not found"}), 404

    except Exception as e:
        print(f"Error during deletion: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)