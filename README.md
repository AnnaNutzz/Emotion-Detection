# Real-Time Emotion Detection with CNN, OpenCV, and Tkinter

This project performs **real-time facial emotion detection** using a custom-trained **Convolutional Neural Network (CNN)** on grayscale facial images. It uses **OpenCV** for webcam-based face detection, **TensorFlow/Keras** for deep learning, and **Tkinter** for a GUI interface to control and visualize results.

## 📌 Features

- Train a CNN model on your own facial emotion dataset
- Real-time emotion detection from webcam feed
- Logs detected emotions (with timestamp and position) into a JSON file
- Interactive GUI with buttons to stop detection and view emotion distribution
- Emotion categories: `Angry`, `Disgust`, `Fear`, `Happy`, `Sad`, `Surprise`, `Neutral`

---

## 🧠 Dataset Format

The dataset directory (`data/`) should be structured as follows:
Used data: [https://www.kaggle.com/datasets/ananthu017/emotion-detection-fer]

```
data/
├── 0  # Angry
├── 1  # Disgust
├── 2  # Fear
├── 3  # Happy
├── 4  # Sad
├── 5  # Surprise
└── 6  # Neutral
```

Each folder should contain grayscale `.jpg` or `.png` face images sized 48x48 or will be resized accordingly.

---

## 🧪 Model Architecture

- 3 convolutional blocks with `Conv2D`, `MaxPooling2D`, and `Dropout`
- Fully connected layer followed by `BatchNormalization` and `Dropout`
- Output layer with `softmax` activation for 7 emotion classes

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

Optional (for GUI):

```bash
pip install tk
```

---

## 🚀 How to Run

### 1. **Train the Model**

Make sure the dataset is in the correct format. The script automatically:

- Loads and preprocesses images
- Trains the model with data augmentation
- Saves the model to `custom_fer_model.h5`

```python
# Just run the script and model will be trained
```

### 2. **Start Real-Time Detection**

The GUI will launch with:

- A **Stop and Save** button to terminate camera and log data
- A **Plot Emotion Distribution** button to visualize the frequency of each emotion

Emotions detected will be saved in a file named: `emotion_log.json`.

---

## 📊 Sample JSON Log Format

```json
{
  "timestamp": "2025-04-06 13:45:22",
  "emotion": "Happy",
  "confidence": "96.23%",
  "position": {
    "x": 123,
    "y": 88,
    "width": 64,
    "height": 64
  }
}
```

---

## 🖼️ Emotion Distribution Graph

After running detection, you can click **"Plot Emotion Distribution"** to see a bar chart summarizing all detected emotions.

---

## 💡 Notes

- Ensure a good light environment for better face detection.
- Use a dataset with a balanced number of images per class for better accuracy.
- Press `q` during camera view to manually stop detection.

---

## 📌 Future Enhancements

- Integrate audio-based emotion detection
- Provide video input instead of just webcam
- Add option to export data to Excel or Google Sheets
- Create a mobile/desktop application interface

---

## 🙌 Credits

Developed by **Ahana**  
Project for **Emotion Detection** coursework at Bennett University  
Using: TensorFlow, OpenCV, Tkinter, Matplotlib

---
