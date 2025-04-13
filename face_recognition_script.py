import cv2
import face_recognition
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import threading

# Load known faces and names
known_face_encodings = []
known_face_names = []

# Load images from the "known_faces" folder
known_faces_dir = "known_faces"
image_extensions = {".jpg", ".jpeg", ".png"}  # Allowed image formats

for filename in os.listdir(known_faces_dir):
    if os.path.splitext(filename)[1].lower() in image_extensions:  # Ensure it's an image
        img_path = os.path.join(known_faces_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        
        if encodings:
            known_face_encodings.append(encodings[0])  # Get the first face encoding
            known_face_names.append(os.path.splitext(filename)[0])  # Extract name from filename

# Load Haarcascade model (for initial detection)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def recognize_faces(image_path):
    """ Detect and recognize faces in an image """
    img = cv2.imread(image_path)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Find the best match
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle & name
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(img, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return img

def open_image():
    """ Open an image file and recognize faces """
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        processed_img = recognize_faces(file_path)

        # Convert image for Tkinter display
        img_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Update label with new image
        image_label.config(image=img_tk)
        image_label.image = img_tk

def start_webcam():
    """ Start real-time face recognition """
    def webcam_feed():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Find the best match
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None

                if best_match_index is not None and matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw rectangle & name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Webcam Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    # Run webcam function in a separate thread to prevent GUI freezing
    threading.Thread(target=webcam_feed, daemon=True).start()

# Create main window
root = tk.Tk()
root.title("Face Recognition App")

# Create buttons
btn_image = tk.Button(root, text="Open Image", command=open_image, font=("Arial", 14))
btn_image.pack(pady=10)

btn_webcam = tk.Button(root, text="Start Webcam", command=start_webcam, font=("Arial", 14))
btn_webcam.pack(pady=10)

# Image display label
image_label = tk.Label(root)
image_label.pack()

# Run the Tkinter event loop
root.mainloop()
