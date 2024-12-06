import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models
import cv2
import os
import sys
sys.path.append("C:\\Users\\Mohammad Ishaan\\AppData\\Local\\Programs\\Python\\Python310\\Lib\\site-packages")
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('exercise_angles.csv')

# Separate features and labels
X = data.drop(columns=['Side', 'Label']).values  # Features (angles)
y = data['Label'].apply(lambda x: 1 if x == 'Jumping Jacks' else 0).values  # Binary labels

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')

# Build the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Save the model
model.save('exercise_angle_model.h5')

# Evaluate the model
y_val_pred = model.predict(X_val)
y_val_pred_classes = (y_val_pred > 0.5).astype(int)  # Convert probabilities to binary classes

# Calculate accuracy
accuracy = accuracy_score(y_val, y_val_pred_classes)
print(f'Model Accuracy: {accuracy * 100:.2f}%')

# Dummy function for pose estimation (replace with actual implementation)
def calculate_angles(frame):
    # Use MediaPipe or OpenPose here to extract joint angles from the frame
    # This is a placeholder returning random values
    return np.random.rand(10)  # Assuming 10 angles as input to the model

# Function to analyze angles from a video and provide feedback
def analyze_video(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Error opening video file: {video_path}")

    angles_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate angles for the frame
        angles = calculate_angles(frame)
        angles_list.append(angles)

    cap.release()

    # Convert angles to a format suitable for the model
    angles_array = np.array(angles_list)

    # Load the scaler
    scaler = joblib.load('scaler.pkl')
    angles_array = scaler.transform(angles_array)  # Normalize

    # Make predictions
    predictions = model.predict(angles_array)
    results = ["Correct" if pred > 0.5 else "Incorrect" for pred in predictions]

    # Calculate the accuracy score on a scale of 1 to 10
    correct_predictions = sum([1 for result in results if result == "Correct"])
    accuracy_percentage = (correct_predictions / len(results)) * 100
    accuracy_score = round(accuracy_percentage / 10)  # Scale to 1-10 range

    # Provide feedback based on accuracy
    feedback = ""
    if accuracy_score == 10:
        feedback = "Perfect form! Keep it up!"
    elif accuracy_score >= 8:
        feedback = "Great form! Just a few minor adjustments."
    elif accuracy_score >= 5:
        feedback = "Fair form. There are some areas to improve."
    else:
        feedback = "Form needs improvement. Try to focus on technique."

    return accuracy_score, feedback

# Example usage
video_name = "custom_video.mp4"  # Replace with your video file name
video_path = os.path.join(os.getcwd(), video_name)

try:
    accuracy, feedback = analyze_video(video_path)
    print(f"Form Accuracy: {accuracy}/10")
    print("Feedback:", feedback)
except Exception as e:
    print(f"Error: {e}")
