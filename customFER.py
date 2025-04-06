import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import os
import json
import time
import tkinter as tk
import matplotlib.pyplot as plt
from collections import Counter

# Disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Suppress TensorFlow logs (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths to dataset
DATASET_PATH = '\\data_path\\'

# Parameters
IMG_SIZE = 48  # Image size (48x48)
BATCH_SIZE = 20
EPOCHS = 60
NUM_CLASSES = 7  # Categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

# Model ===================================================================================================
# 1. Load the Dataset and Preprocess
def load_dataset():
    X, y = [], []
    for emotion in os.listdir(DATASET_PATH):
        label = int(emotion) 
        for img_file in os.listdir(os.path.join(DATASET_PATH, emotion)):
            img_path = os.path.join(DATASET_PATH, emotion, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0  # Normalize
    y = tf.keras.utils.to_categorical(y, NUM_CLASSES)  # One-hot encode labels
    return X, y

# Load and split dataset
X, y = load_dataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Define the CNN Model
model = Sequential([
    Conv2D(64, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    BatchNormalization(),               # Keep BatchNormalization only here
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),  # Skip BatchNormalization
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(256, (3, 3), activation='relu'),  # Skip BatchNormalization
    MaxPooling2D((2, 2)),
    Flatten(),

    Dense(256, activation='relu'),
    BatchNormalization(),               # Keep BatchNormalization before dropout
    Dropout(0.5),
    Dense(NUM_CLASSES, activation='softmax')
])


# 3. Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Data Augmentation (to reduce overfitting)
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)
datagen.fit(X_train)

# 5. Train the Model
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          validation_data=(X_test, y_test),
          epochs=EPOCHS)

# 6. Save the Model
model.save('custom_fer_model.h5')
print("Model saved to custom_fer_model.h5")

# Global variables
log_data = []  # List to store logs
stop_camera = False  # Flag to stop the camera

# GUI Functions and Graphs ==================================================================================
# Function to stop the camera and save the JSON
def stop_and_save():
    global stop_camera
    stop_camera = True  # Set the flag to stop the camera

# Function to plot the emotion distribution
def plot_emotion_distribution():
    # Count occurrences of each emotion
    emotion_counts = Counter(entry['emotion'] for entry in log_data)
    
    # Plot using matplotlib
    plt.figure(figsize=(10, 6))
    plt.bar(emotion_counts.keys(), emotion_counts.values(), color='skyblue')
    plt.xlabel('Emotions')
    plt.ylabel('Frequency')
    plt.title('Emotion Distribution')
    plt.show()

# Main =========================================================================================================
# 7. Load the Model and Use It for Real-Time Emotion Detection with JSON logging
def real_time_emotion_detection():
    global stop_camera
    model = tf.keras.models.load_model('custom_fer_model.h5')  # Load the saved .h5 model
    cap = cv2.VideoCapture(0)

    log_interval = 5  # Log every 5 seconds
    last_log_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or stop_camera:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') \
                .detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE)) / 255.0
            roi = np.reshape(roi, (1, IMG_SIZE, IMG_SIZE, 1))

            # Predict emotion
            predictions = model.predict(roi)
            emotion_label = np.argmax(predictions)
            confidence = np.max(predictions)

            # Display emotion label and confidence   
            emotion_name = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][emotion_label]
            text = f"{emotion_name} ({confidence*100:.2f}%)"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Check if it's time to log data
            current_time = time.time()
            if current_time - last_log_time >= log_interval:
                # Log the data
                log_entry = {
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(current_time)),
                    "emotion": emotion_name,
                    "confidence": f"{confidence*100:.2f}%",
                    "position": {"x": int(x), "y": int(y), "width": int(w), "height": int(h)}
                }
                log_data.append(log_entry)
                last_log_time = current_time  # Update last log time

        # Display the frame
        cv2.imshow('Real-Time Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Save the log data to a JSON file after stopping
    with open("emotion_log.json", "w") as log_file:
        json.dump(log_data, log_file, indent=4)
    print("Emotion log saved to emotion_log.json")

# Tkinter GUI =================================================================================================
# Create a simple GUI with Tkinter for the stop button
root = tk.Tk()
root.title("Real-Time Emotion Detection")

# Start the real-time emotion detection in a separate thread to keep the GUI responsive
import threading
threading.Thread(target=real_time_emotion_detection).start()

# Add a Stop and Save button to end the detection
stop_button = tk.Button(root, text="Stop and Save", command=stop_and_save)
stop_button.pack(pady=20)

# Add a Plot button to display the emotion distribution
plot_button = tk.Button(root, text="Plot Emotion Distribution", command=plot_emotion_distribution)
plot_button.pack(pady=10)

# Run the Tkinter GUI loop
root.mainloop()
