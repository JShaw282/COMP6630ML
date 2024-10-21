import cv2
from videofunctions import FrameCount, GetFrame
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split

# test Frame count function
#print(FrameCount("testvideo.mp4"))
#frame = GetFrame("testvideo.mp4", 40)

# test get specific frame from video
#cv2.imshow('Frame', frame)
#cv2.waitKey(0)
def preprocess_image(image_path, target_size=(224, 224)):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image at {image_path}")
        return None
    
    # Convert BGR to RGB (OpenCV uses BGR by default)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize the image
    img = cv2.resize(img, target_size)
    
    # Normalize pixel values
    img = img.astype(np.float32) / 255.0
    
    return img

def predict_image(model, image):
    try:
        # Add batch dimension
        img = np.expand_dims(image, axis=0)
        
        # Make prediction
        prediction = model.predict(img, verbose=0)[0][0]
        return prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

def process_image_folder(model, folder_path):
    results = []
    
    # Check if the folder exists
    if not os.path.exists(folder_path):
        print(f"Error: Folder {folder_path} does not exist.")
        return results

    # Process each image in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            img = preprocess_image(image_path)
            
            if img is not None:
                prediction = predict_image(model, img)
                if prediction is not None:
                    results.append((filename, prediction))
                else:
                    print(f"Could not make a prediction for {filename}")
            else:
                print(f"Could not preprocess {filename}")
    
    return results

def extract_frames(video_path, n_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.array(frames)
    
    # Ensure we don't try to extract more frames than the video has
    n_frames = min(n_frames, total_frames)
    
    if n_frames > 1:
        step = max(1, total_frames // n_frames)
    else:
        step = 1

    for i in range(min(total_frames, n_frames)):
        frame_pos = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))
            frame = frame / 255.0  # Normalize pixel values
            frames.append(frame)

    cap.release()
    return np.array(frames)

def load_data(positive_dir, negative_dir):
    X = []
    y = []

    # Load positive examples
    for video_name in os.listdir(positive_dir):
        video_path = os.path.join(positive_dir, video_name)
        frames = extract_frames(video_path)
        X.extend(frames)
        y.extend([1] * len(frames))  # 1 for positive class

    # Load negative examples
    for video_name in os.listdir(negative_dir):
        video_path = os.path.join(negative_dir, video_name)
        frames = extract_frames(video_path)
        X.extend(frames)
        y.extend([0] * len(frames))  # 0 for negative class

    return np.array(X), np.array(y)

# Modify the create_model function
def create_model(input_shape):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Flatten(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def predict_video(model, video_path):
    frames = extract_frames(video_path)
    if len(frames) == 0:
        print(f"No frames could be extracted from {video_path}")
        return None
    
    try:
        # Ensure frames is a numpy array with the correct shape
        frames = np.array(frames)
        if len(frames.shape) == 3:
            # If we have a single frame, add a batch dimension
            frames = np.expand_dims(frames, axis=0)
        elif len(frames.shape) != 4:
            print(f"Unexpected frame shape: {frames.shape}")
            return None

        # Check if the input shape matches the model's expected input
        input_shape = model.input_shape[1:]  # Exclude batch dimension
        if frames.shape[1:] != input_shape:
            print(f"Frame shape {frames.shape[1:]} does not match model input shape {input_shape}")
            return None

        predictions = model.predict(frames, verbose=0)  # Set verbose to 0 to avoid progress bar issues
        avg_prediction = np.mean(predictions)
        return avg_prediction
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return None

positive_dir = 'pos'
negative_dir = 'neg'

X, y = load_data(positive_dir, negative_dir)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

input_shape = X_train.shape[1:]

model = create_model(input_shape)
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, validation_split=0.1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Save the model
model.save('binary_video_classification_model.h5')

# Path to your test images folder
test_folder = 'test-images'

# Process all images in the folder
results = process_image_folder(model, test_folder)

# Print results
print("\nResults:")
for filename, prediction in results:
    print(f"{filename}: {prediction:.4f} - {'Contains the item' if prediction > 0.5 else 'Does not contain the item'}")

# Print summary
print(f"\nProcessed {len(results)} images.")
positives = sum(1 for _, pred in results if pred > 0.5)
negatives = len(results) - positives
print(f"Positive predictions: {positives}")
print(f"Negative predictions: {negatives}")

# Additional debugging information
print("\nDebugging Information:")
print(f"Model input shape: {model.input_shape}")
