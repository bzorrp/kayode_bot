import cv2
import face_recognition
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
import openai
from dotenv import load_dotenv
import os

#Load the environment variables from the .env file
load_dotenv()

# Hiding my API key because I paid for it
openai.api_key = os.getenv('OPENAI_API_KEY')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Load multiple images of authorized personnel for better encoding
authorized_images = ["your_face1.jpg", "your_face2.jpg", "your_face3.jpg"]
authorized_face_encodings = []

for image_path in authorized_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        authorized_face_encodings.append(encoding[0])

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=150,
        temperature=0.7,
    )
    return response['choices'][0]['message']['content']

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = data['message']
    
    # Facial recognition check
    if not is_recognized_face():
        return jsonify({'response': "Face not recognized. Access denied."}) #Could use a cheekier response

    reply = get_completion(message)
    return jsonify({'response': reply})

def is_recognized_face():
    try:
        # Capture a frame from the webcam
        video_capture = cv2.VideoCapture(0)
        ret, frame = video_capture.read()
        video_capture.release()

        if not ret:
            print("Failed to capture image from webcam")
            return False

        # Convert the frame from BGR to RGB
        rgb_frame = frame[:, :, ::-1]

        # Apply some preprocessing to the image to make this much easier
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
        equalized_frame = cv2.equalizeHist(gray_frame)
        rgb_frame = cv2.cvtColor(equalized_frame, cv2.COLOR_GRAY2RGB)

        # Find face locations and encodings in the captured frame
        face_locations = face_recognition.face_locations(rgb_frame)
        if not face_locations:
            print("No faces detected in the captured frame")
            return False

        print(f"Detected faces in captured frame: {face_locations}")

        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        if len(face_encodings) == 0:
            print("No face encodings found in the captured frame")
            return False

        # Compare the face encoding in the captured frame with the stored face encodings
        matches = face_recognition.compare_faces(authorized_face_encodings, face_encodings[0], tolerance=0.6)
        print(f"Face matches: {matches}")
        return any(matches)

    except Exception as e:
        print(f"Error in face recognition: {e}")
        return False

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
