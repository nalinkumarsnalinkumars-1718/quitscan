# sound_detector.py
import sounddevice as sd
import threading
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd
import datetime
import os

# Load YAMNet once
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = yamnet_model.class_map_path().numpy().decode('utf-8')
class_names = list(pd.read_csv(class_map_path)['display_name'])

# Global flag for controlling detection
stop_flag = threading.Event()
trigger_event = threading.Event()
latest_detection = {"sound": "", "time": "", "image_path": ""}

def record_and_classify_sound(target_classes=('Bark', 'Dog', 'Glass')):
    duration = 2
    sample_rate = 16000

    print("Recording...")
    try:
        rec = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()
        audio = rec.flatten()
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)

        scores, embeddings, spectrogram = yamnet_model(audio)
        mean_scores = tf.reduce_mean(scores, axis=0)
        top_scores = tf.argsort(mean_scores, direction='DESCENDING')[:3]
        class_str = class_names[top_scores[0]]

        print(f"Detected: {class_str}")
        latest_detection['sound'] = class_str
        latest_detection['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if any(key.lower() in class_str.lower() for key in target_classes):
            print("Triggering camera!")
            trigger_event.set()
        else:
            print("Unwanted sound detected.")
    except Exception as e:
        print("Error during sound classification:", e)

def take_photo_on_event():
    cam = cv2.VideoCapture(0)
    try:
        while True:
            trigger_event.wait()
            ret, frame = cam.read()
            if ret:
                date_col = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
                folder = "static/images"
                os.makedirs(folder, exist_ok=True)
                filename = f"{folder}/{date_col}.jpg"
                cv2.imwrite(filename, frame)
                latest_detection["image_path"] = filename
                print(f"Saved photo: {filename}")
            trigger_event.clear()
    finally:
        cam.release()

def sound_monitor_loop(target_classes=('Speech', 'Dog', 'Glass')):
    while not stop_flag.is_set():
        record_and_classify_sound(target_classes)

def start_detection():
    stop_flag.clear()
    threading.Thread(target=sound_monitor_loop, daemon=True).start()
    threading.Thread(target=take_photo_on_event, daemon=True).start()

def stop_detection():
    stop_flag.set()

def get_latest_detection():
    return latest_detection
