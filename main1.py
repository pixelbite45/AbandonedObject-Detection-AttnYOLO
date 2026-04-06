import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import os
from datetime import datetime
import json



PERSON_CLASS_ID = 0
ITEM_CLASS_NAMES = ["backpack", "handbag", "suitcase", "laptop", "cell phone"]

FULL_FRAME_ALERT_DIR = "alert_full_frames" 
OBJECT_IMAGE_DIR = "object_images"
import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import defaultdict
import time
import os
from datetime import datetime
import json

MODEL_PATH = "yolov8n.pt"
VIDEO_STREAM_URL = r"C:\Users\rohan\Desktop\college\Innovative vii\test3.mp4"

PERSON_CLASS_ID = 0
ITEM_CLASS_NAMES = ["backpack", "handbag", "suitcase", "laptop", "cell phone"]

FULL_FRAME_ALERT_DIR = "alert_full_frames" 
OBJECT_IMAGE_DIR = "object_images"

# Parameters
ASSOCIATION_THRESHOLD = 850
ALERT_PATIENCE_SECONDS = 5
RESIZE_FACTOR = 0.6


def calculate_distance(p1: sv.Point, p2: sv.Point) -> float:
    """Calculates the Euclidean distance between two sv.Point objects."""
    return np.sqrt((p1.x - p2.x)*2 + (p1.y - p2.y)*2)

def main(VIDEO_STREAM=None): # Accept an optional override
    
    os.makedirs(FULL_FRAME_ALERT_DIR, exist_ok=True)
    os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    item_class_ids = [k for k, v in model.model.names.items() if v in ITEM_CLASS_NAMES]

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    person_states = defaultdict(lambda: {"items": set(), "last_box": None})
    tracked_items_info = {}

    
    orphan_candidates = {} 
    # Format: {item_id: {"timestamp": float, "person_id": int, "person_last_box_original_scale": np.ndarray}}

    # Use override stream if provided, otherwise use the default URL
    stream_to_open = VIDEO_STREAM if VIDEO_STREAM is not None else VIDEO_STREAM_URL
    cap = cv2.VideoCapture(stream_to_open)
    if not cap.isOpened():
        print(f"[ERROR main.py] Could not open video stream at {stream_to_open}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a clean copy of the frame for cropping and annotation later
        original_frame = frame.copy()

        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = np.array([cls in item_class_ids + [PERSON_CLASS_ID] for cls in detections.class_id])
        detections = detections[mask]
        tracked_detections = tracker.update_with_detections(detections)

        person_mask = tracked_detections.class_id == PERSON_CLASS_ID
        persons_detections = tracked_detections[person_mask]
        item_mask = np.isin(tracked_detections.class_id, item_class_ids)
        items_detections = tracked_detections[item_mask]

        for tracker_id, class_id in zip(items_detections.tracker_id, items_detections.class_id):
            tracked_items_info[tracker_id] = model.model.names[class_id]

        person_box_map = {tid: box for tid, box in zip(persons_detections.tracker_id, persons_detections.xyxy)}
        item_box_map = {tid: box for tid, box in zip(items_detections.tracker_id, items_detections.xyxy)}

        # --- LOGIC ---
        current_tracked_person_ids = set(persons_detections.tracker_id)
        current_tracked_item_ids = set(items_detections.tracker_id)

        # 1. Update associations and last known person box
        for person_id, person_box in person_box_map.items():
            # Scale the person_box back to original frame size for storage
            x1, y1, x2, y2 = map(int, person_box)
            person_last_box_original_scale = np.array([
                int(x1 / RESIZE_FACTOR), int(y1 / RESIZE_FACTOR), 
                int(x2 / RESIZE_FACTOR), int(y2 / RESIZE_FACTOR)
            ])
            person_states[person_id]["last_box"] = person_last_box_original_scale

            person_center = sv.Point(x=(person_box[0] + person_box[2]) / 2, y=(person_box[1] + person_box[3]) / 2)
            
            for item_id, item_box in item_box_map.items():
                item_center = sv.Point(x=(item_box[0] + item_box[2]) / 2, y=(item_box[1] + item_box[3]) / 2)
                
                if calculate_distance(person_center, item_center) < ASSOCIATION_THRESHOLD:
                    person_states[person_id]["items"].add(item_id)
                    if item_id in orphan_candidates:
                        # Person re-claimed the item, cancel the orphan timer
                        del orphan_candidates[item_id]

        # 2. Handle disappearing people to create orphan candidates
        for person_id in list(person_states.keys()):
            if person_id not in current_tracked_person_ids:
                # This person just disappeared.
                
                person_last_box_original_scale = person_states[person_id].get("last_box")
                items_left = person_states[person_id]["items"]
                
                if items_left and person_last_box_original_scale is not None:
                    for item_id in items_left:
                        if item_id in current_tracked_item_ids and item_id not in orphan_candidates:
                            # This item is now an orphan candidate.
                            # Store the data we need to save later.
                            orphan_candidates[item_id] = {
                                "timestamp": time.time(),
                                "person_id": person_id,
                                "person_last_box_original_scale": person_last_box_original_scale # Store scaled box
                            }
                del person_states[person_id]

        # 3. Process orphan candidates and trigger alerts
        for item_id in list(orphan_candidates.keys()):
            if item_id in current_tracked_item_ids:
                
                time_since_orphaned = time.time() - orphan_candidates[item_id]["timestamp"]
                
                # **** START OF CORRECTED BLOCK ****
                if time_since_orphaned > ALERT_PATIENCE_SECONDS:
                    item_name = tracked_items_info.get(item_id, "an item")
                    print(f"🚨 ALERT: An item '{item_name}' (ID: {item_id}) has been left unattended!")
                    
                    # Get the actual alert timestamp (this is what server.py needs)
                    alert_time_epoch = time.time()

                    # --- Write alert timestamp to file for server.py ---
                    alert_file = "alerts.json" # CORRECT FILENAME (plural)
                    try:
                        with open(alert_file, 'r') as f:
                            data = json.load(f)
                            # Load existing timestamps (supports list or dict)
                            if isinstance(data, dict) and "alerts" in data:
                                alerts = data["alerts"]
                            elif isinstance(data, list):
                                alerts = data
                            else:
                                alerts = []
                    except (FileNotFoundError, json.JSONDecodeError):
                        alerts = []
                    
                    alerts.append(alert_time_epoch) # Just append the new timestamp

                    with open(alert_file, 'w') as f:
                        # Save it in the format server.py can read (a list inside a dict)
                        # Using set() removes duplicates if this triggers multiple times
                        json.dump({"alerts": list(set(alerts))}, f, indent=4) 

                    # --- NOW, define data for image saving (THIS PART IS MOVED UP) ---
                    orphan_data = orphan_candidates[item_id]
                    # **** END OF CORRECTED BLOCK ****
                    
                    person_id = orphan_data["person_id"]
                    person_last_box_original_scale = orphan_data["person_last_box_original_scale"]
                    item_box_current_resized = item_box_map.get(item_id) # Get item's current box (resized scale)

                    now = datetime.now()
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

                    # --- Save Object Image (cropped from original frame) ---
                    if item_box_current_resized is not None:
                        try:
                            # Scale item's current box up to original frame size for cropping
                            x1_item, y1_item, x2_item, y2_item = map(int, item_box_current_resized)
                            x1_item, y1_item = int(x1_item / RESIZE_FACTOR), int(y1_item / RESIZE_FACTOR)
                            x2_item, y2_item = int(x2_item / RESIZE_FACTOR), int(y2_item / RESIZE_FACTOR)
                            
                            # Ensure coordinates are valid
                            x1_item, y1_item = max(0, x1_item), max(0, y1_item)
                            x2_item, y2_item = min(original_frame.shape[1], x2_item), min(original_frame.shape[0], y2_item)

                            cropped_item = original_frame[y1_item:y2_item, x1_item:x2_item]
                            
                            object_filename = f"{person_id}{timestamp_str}_object_id{item_id}.jpg"
                            object_save_path = os.path.join(OBJECT_IMAGE_DIR, object_filename)
                            cv2.imwrite(object_save_path, cropped_item)
                            print(f"Saved orphan item image: {object_save_path}")
                        except Exception as e:
                            print(f"Error saving item {item_id} image: {e}")

                    # --- Save Full Frame with Annotations ---
                    annotated_alert_frame = original_frame.copy()
                    
                    # Draw box for person (if available)
                    if person_last_box_original_scale is not None:
                        try:
                            x1_p, y1_p, x2_p, y2_p = map(int, person_last_box_original_scale)
                            cv2.rectangle(annotated_alert_frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 2) # Green for person
                            cv2.putText(annotated_alert_frame, f"Person {person_id} (Last Seen)", (x1_p, y1_p - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error drawing person box on alert frame: {e}")

                    # Draw box for abandoned item (if available)
                    if item_box_current_resized is not None:
                        try:
                            x1_item, y1_item, x2_item, y2_item = map(int, item_box_current_resized)
                            # Scale to original frame size for drawing
                            x1_item_orig = int(x1_item / RESIZE_FACTOR)
                            y1_item_orig = int(y1_item / RESIZE_FACTOR)
                            x2_item_orig = int(x2_item / RESIZE_FACTOR)
                            y2_item_orig = int(y2_item / RESIZE_FACTOR)

                            cv2.rectangle(annotated_alert_frame, (x1_item_orig, y1_item_orig), (x2_item_orig, y2_item_orig), (0, 0, 255), 2) # Red for item
                            cv2.putText(annotated_alert_frame, f"Abandoned {item_name} (ID: {item_id})", (x1_item_orig, y1_item_orig - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error drawing item box on alert frame: {e}")
                    
                    full_frame_filename = f"alert_P{person_id}I{item_id}{timestamp_str}.jpg"
                    full_frame_save_path = os.path.join(FULL_FRAME_ALERT_DIR, full_frame_filename)
                    cv2.imwrite(full_frame_save_path, annotated_alert_frame)
                    print(f"Saved full alert frame: {full_frame_save_path}")
                    # --- END NEW ---

                    del orphan_candidates[item_id]  # Reset to avoid repeated alerts
            else:
                # The item also disappeared, so remove it
                del orphan_candidates[item_id]

        # --- VISUALIZATION ---
        labels = [f"#{tracker_id} {model.model.names[class_id]}" for tracker_id, class_id in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        cv2.imshow("Abandoned Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # This block will not be run when server.py imports it,
    # but it allows you to test main.py on its own if you want.
    main()

# Parameters
ASSOCIATION_THRESHOLD = 850
ALERT_PATIENCE_SECONDS = 5
RESIZE_FACTOR = 0.6


def calculate_distance(p1: sv.Point, p2: sv.Point) -> float:
    """Calculates the Euclidean distance between two sv.Point objects."""
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def main(VIDEO_STREAM=None):  # Accept an optional override
    
    os.makedirs(FULL_FRAME_ALERT_DIR, exist_ok=True)
    os.makedirs(OBJECT_IMAGE_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    item_class_ids = [k for k, v in model.model.names.items() if v in ITEM_CLASS_NAMES]

    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator(text_thickness=1, text_scale=0.5)

    person_states = defaultdict(lambda: {"items": set(), "last_box": None})
    tracked_items_info = {}

    
    orphan_candidates = {} 
    # Format: {item_id: {"timestamp": float, "person_id": int, "person_last_box_original_scale": np.ndarray}}

    cap = cv2.VideoCapture(VIDEO_STREAM_URL)
    if not cap.isOpened():
        print(f"Error: Could not open video stream at {VIDEO_STREAM_URL}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Create a clean copy of the frame for cropping and annotation later
        original_frame = frame.copy()

        frame = cv2.resize(frame, (0, 0), fx=RESIZE_FACTOR, fy=RESIZE_FACTOR, interpolation=cv2.INTER_AREA)
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)

        mask = np.array([cls in item_class_ids + [PERSON_CLASS_ID] for cls in detections.class_id])
        detections = detections[mask]
        tracked_detections = tracker.update_with_detections(detections)

        person_mask = tracked_detections.class_id == PERSON_CLASS_ID
        persons_detections = tracked_detections[person_mask]
        item_mask = np.isin(tracked_detections.class_id, item_class_ids)
        items_detections = tracked_detections[item_mask]

        for tracker_id, class_id in zip(items_detections.tracker_id, items_detections.class_id):
            tracked_items_info[tracker_id] = model.model.names[class_id]

        person_box_map = {tid: box for tid, box in zip(persons_detections.tracker_id, persons_detections.xyxy)}
        item_box_map = {tid: box for tid, box in zip(items_detections.tracker_id, items_detections.xyxy)}

        # --- LOGIC ---
        current_tracked_person_ids = set(persons_detections.tracker_id)
        current_tracked_item_ids = set(items_detections.tracker_id)

        # 1. Update associations and last known person box
        for person_id, person_box in person_box_map.items():
            # Scale the person_box back to original frame size for storage
            x1, y1, x2, y2 = map(int, person_box)
            person_last_box_original_scale = np.array([
                int(x1 / RESIZE_FACTOR), int(y1 / RESIZE_FACTOR), 
                int(x2 / RESIZE_FACTOR), int(y2 / RESIZE_FACTOR)
            ])
            person_states[person_id]["last_box"] = person_last_box_original_scale

            person_center = sv.Point(x=(person_box[0] + person_box[2]) / 2, y=(person_box[1] + person_box[3]) / 2)
            
            for item_id, item_box in item_box_map.items():
                item_center = sv.Point(x=(item_box[0] + item_box[2]) / 2, y=(item_box[1] + item_box[3]) / 2)
                
                if calculate_distance(person_center, item_center) < ASSOCIATION_THRESHOLD:
                    person_states[person_id]["items"].add(item_id)
                    if item_id in orphan_candidates:
                        # Person re-claimed the item, cancel the orphan timer
                        del orphan_candidates[item_id]

        # 2. Handle disappearing people to create orphan candidates
        for person_id in list(person_states.keys()):
            if person_id not in current_tracked_person_ids:
                # This person just disappeared.
                
                person_last_box_original_scale = person_states[person_id].get("last_box")
                items_left = person_states[person_id]["items"]
                
                if items_left and person_last_box_original_scale is not None:
                    for item_id in items_left:
                        if item_id in current_tracked_item_ids and item_id not in orphan_candidates:
                            # This item is now an orphan candidate.
                            # Store the data we need to save *later*.
                            orphan_candidates[item_id] = {
                                "timestamp": time.time(),
                                "person_id": person_id,
                                "person_last_box_original_scale": person_last_box_original_scale # Store scaled box
                            }
                del person_states[person_id]

        # 3. Process orphan candidates and trigger alerts
        for item_id in list(orphan_candidates.keys()):
            if item_id in current_tracked_item_ids:
                
                time_since_orphaned = time.time() - orphan_candidates[item_id]["timestamp"]
                
                if time_since_orphaned > ALERT_PATIENCE_SECONDS:
                    item_name = tracked_items_info.get(item_id, "an item")
                    print(f"🚨 ALERT: An item '{item_name}' (ID: {item_id}) has been left unattended!")
                    
                    # --- NEW: Save images ONLY when alert is confirmed ---
                    orphan_data = orphan_candidates[item_id]
                    person_id = orphan_data["person_id"]
                    person_last_box_original_scale = orphan_data["person_last_box_original_scale"]
                    item_box_current_resized = item_box_map.get(item_id) # Get item's *current* box (resized scale)

                    now = datetime.now()
                    timestamp_str = now.strftime("%Y%m%d_%H%M%S")

                    # --- Save Object Image (cropped from original frame) ---
                    if item_box_current_resized is not None:
                        try:
                            # Scale item's current box up to original frame size for cropping
                            x1_item, y1_item, x2_item, y2_item = map(int, item_box_current_resized)
                            x1_item, y1_item = int(x1_item / RESIZE_FACTOR), int(y1_item / RESIZE_FACTOR)
                            x2_item, y2_item = int(x2_item / RESIZE_FACTOR), int(y2_item / RESIZE_FACTOR)
                            
                            # Ensure coordinates are valid
                            x1_item, y1_item = max(0, x1_item), max(0, y1_item)
                            x2_item, y2_item = min(original_frame.shape[1], x2_item), min(original_frame.shape[0], y2_item)

                            cropped_item = original_frame[y1_item:y2_item, x1_item:x2_item]
                            
                            object_filename = f"{person_id}_{timestamp_str}_object_id_{item_id}.jpg"
                            object_save_path = os.path.join(OBJECT_IMAGE_DIR, object_filename)
                            cv2.imwrite(object_save_path, cropped_item)
                            print(f"Saved orphan item image: {object_save_path}")
                        except Exception as e:
                            print(f"Error saving item {item_id} image: {e}")

                    # --- Save Full Frame with Annotations ---
                    annotated_alert_frame = original_frame.copy()
                    
                    # Draw box for person (if available)
                    if person_last_box_original_scale is not None:
                        try:
                            x1_p, y1_p, x2_p, y2_p = map(int, person_last_box_original_scale)
                            cv2.rectangle(annotated_alert_frame, (x1_p, y1_p), (x2_p, y2_p), (0, 255, 0), 2) # Green for person
                            cv2.putText(annotated_alert_frame, f"Person {person_id} (Last Seen)", (x1_p, y1_p - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        except Exception as e:
                            print(f"Error drawing person box on alert frame: {e}")

                    # Draw box for abandoned item (if available)
                    if item_box_current_resized is not None:
                        try:
                            x1_item, y1_item, x2_item, y2_item = map(int, item_box_current_resized)
                            # Scale to original frame size for drawing
                            x1_item_orig = int(x1_item / RESIZE_FACTOR)
                            y1_item_orig = int(y1_item / RESIZE_FACTOR)
                            x2_item_orig = int(x2_item / RESIZE_FACTOR)
                            y2_item_orig = int(y2_item / RESIZE_FACTOR)

                            cv2.rectangle(annotated_alert_frame, (x1_item_orig, y1_item_orig), (x2_item_orig, y2_item_orig), (0, 0, 255), 2) # Red for item
                            cv2.putText(annotated_alert_frame, f"Abandoned {item_name} (ID: {item_id})", (x1_item_orig, y1_item_orig - 10), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        except Exception as e:
                            print(f"Error drawing item box on alert frame: {e}")
                    
                    full_frame_filename = f"alert_P{person_id}_I{item_id}_{timestamp_str}.jpg"
                    full_frame_save_path = os.path.join(FULL_FRAME_ALERT_DIR, full_frame_filename)
                    cv2.imwrite(full_frame_save_path, annotated_alert_frame)
                    print(f"Saved full alert frame: {full_frame_save_path}")
                    # --- END NEW ---

                    del orphan_candidates[item_id]  # Reset to avoid repeated alerts
            else:
                # The item also disappeared, so remove it
                del orphan_candidates[item_id]

        # --- VISUALIZATION ---
        labels = [f"#{tracker_id} {model.model.names[class_id]}" for tracker_id, class_id in zip(tracked_detections.tracker_id, tracked_detections.class_id)]
        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=tracked_detections)
        annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=tracked_detections, labels=labels)
        cv2.imshow("Abandoned Object Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()