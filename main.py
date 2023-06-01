import cv2
import dlib
import numpy as np
from mtcnn import MTCNN
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import face_recognition

# Preprocessing: Face Alignment
def align_face(image):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("D:\FinalProject\shape_predictor_68_face_landmarks\shape_predictor_68_face_landmarks.dat")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    aligned_faces = []
    for face in faces:
        landmarks = predictor(gray, face)
        aligned_face = dlib.get_face_chip(image, landmarks)
        aligned_faces.append(aligned_face)
    return aligned_faces

# Preprocessing: Illumination Normalization
def normalize_illumination(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_l = clahe.apply(l)
    enhanced_lab = cv2.merge((clahe_l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Preprocessing: Face Detection and Cropping
def detect_and_crop_faces(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    cropped_faces = []
    for face in faces:
        x, y, w, h = face['box']
        face_image = image[y:y + h, x:x + w]
        cropped_faces.append(face_image)
    return cropped_faces

# Splitting the Dataset
def split_dataset(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test

# Performing Face Recognition
def perform_face_recognition(X_probe, y_probe, X_gallery, y_gallery):
    y_pred = []
    for i in range(len(X_probe)):
        probe_encoding = face_recognition.face_encodings(X_probe[i], num_jitters=10)[0]
        distances = face_recognition.face_distance(X_gallery, probe_encoding)
        min_distance_index = np.argmin(distances)
        if distances[min_distance_index] < matching_threshold:
            y_pred.append(y_gallery[min_distance_index])
        else:
            y_pred.append("Unknown")
    return y_pred

# Defining the Matching Threshold
matching_threshold = 0.6

# Evaluating Performance Metrics
def evaluate_performance(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    return accuracy, precision, recall, f1

def extract_label(image_path):
    file_name = os.path.basename(image_path)
    label = file_name.split(".")[0]
    return label

# Directory path where the dataset images are stored
dataset_dir = "D:\FinalProject\dataset\lfw"

# Initialize an empty list to store the images
X = []

# Initialize an empty list to store the labels
y = []

# Loop through the subdirectories in the dataset directory
for folder_name in os.listdir(dataset_dir):
    # Construct the path to the subdirectory
    folder_path = os.path.join(dataset_dir, folder_name)

    # Check if the item in the dataset directory is a subdirectory
    if os.path.isdir(folder_path):
        # Loop through the images in the subdirectory
        for image_file in os.listdir(folder_path):
            # Construct the image path
            image_path = os.path.join(folder_path, image_file)

            # Read the image using OpenCV
            image = cv2.imread(image_path)

            # Check if the image is successfully loaded
            if image is not None:
                # Append the image to the list
                X.append(image)

                # Extract the label from the folder name
                label = folder_name

                # Append the label to the list
                y.append(label)
            else:
                print(f"Failed to load image: {image_path}")

# Preprocessing Step: Detect, Align, and Crop Faces
X_processed = []
for image in X:
    aligned_images = align_face(image)
    for aligned_image in aligned_images:
        normalized_image = normalize_illumination(aligned_image)
        cropped_faces = detect_and_crop_faces(normalized_image)
        X_processed.extend(cropped_faces)

# Train and Test split
X_train, X_test, y_train, y_test = split_dataset(X_processed, y)

# Get a reference to the webcam
video_capture = cv2.VideoCapture(0)

# Loop to capture frames from the webcam
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    # Perform face recognition on the captured frame
    aligned_frame = align_face(frame)
    normalized_frame = normalize_illumination(aligned_frame)
    cropped_faces = detect_and_crop_faces(normalized_frame)

    # Perform face recognition on the cropped faces
    y_pred = perform_face_recognition(cropped_faces, y_train, X_train, y_train)

    # Display the recognized faces and labels on the frame
    for face, label in zip(cropped_faces, y_pred):
        x, y, w, h = face['box']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Webcam', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the windows
video_capture.release()
cv2.destroyAllWindows()
