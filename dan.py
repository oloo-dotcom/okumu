import cv2 as cv
import numpy as np
import tensorflow as tf

# Load the pre-trained Haar Cascade model for face detection
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the Emotion Recognition model
def load_emotion_model(model_path):
    return tf.keras.models.load_model(model_path, compile=False)

# Define emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect faces and classify emotions
def detect_and_classify_emotions(image_path, model):
    # Read the image
    image = cv.imread(image_path)
    
    if image is None:
        print(f'Failed to load image at {image_path}')
        return None
    
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        
        # Resize and normalize face for emotion prediction
        resized_face = cv.resize(face_roi, (64, 64))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 64, 64, 1))
        
        # Predict emotion
        emotion_scores = model.predict(reshaped_face)
        predicted_emotion = emotion_labels[np.argmax(emotion_scores)]
        
        # Draw rectangle around the face and label the emotion
        cv.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(image, predicted_emotion, (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv.LINE_AA)
    
    # Display the image with faces and emotions detected
    cv.imshow('Face Emotion Detection', image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Example usage:
if __name__ == "__main__":
    image_path = 'mza1.png'
    model_path = 'model.hdf5'  # Replace with your saved model path
    emotion_model = load_emotion_model(model_path)
    
    if emotion_model:
        detect_and_classify_emotions(image_path, emotion_model)
