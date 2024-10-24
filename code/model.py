import cv2
from videofunctions import FrameCount, GetFrame
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
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

def extract_frames(video_path, n_frames=30):  # Increased number of frames
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        cap.release()
        return np.array(frames)
    
    n_frames = min(n_frames, total_frames)
    step = max(1, total_frames // n_frames)
    
    for i in range(min(total_frames, n_frames)):
        frame_pos = min(i * step, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        if ret:
            # Add basic preprocessing
            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
            # Add data augmentation
            if np.random.random() > 0.5:
                frame = cv2.flip(frame, 1)  # Random horizontal flip
            frame = frame / 255.0  # Normalize
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

def load_data(positive_dir, negative_dir, batch_size=1000):
    """
    Load video frames in batches to manage memory usage.

    Args:
        positive_dir: Directory containing positive example videos
        negative_dir: Directory containing negative example videos
        batch_size: Number of frames to load at once

    Returns:
        Generator yielding batches of (X, y) pairs
    """
    def process_directory(directory, label):
        X_batch = []
        y_batch = []

        for root, _, files in os.walk(directory):
            for video_name in tqdm(files, desc=f"Processing {'positive' if label==1 else 'negative'} videos"):
                if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):  # Add more extensions if needed
                    continue

                video_path = os.path.join(root, video_name)
                try:
                    frames = extract_frames(video_path)

                    for frame in frames:
                        X_batch.append(frame)
                        y_batch.append(label)

                        # Yield batch when it reaches the specified size
                        if len(X_batch) >= batch_size:
                            yield np.array(X_batch), np.array(y_batch)
                            X_batch = []
                            y_batch = []

                except Exception as e:
                    print(f"Error processing {video_path}: {str(e)}")
                    continue

        # Yield remaining frames
        if X_batch:
            yield np.array(X_batch), np.array(y_batch)

    # Process positive examples
    for X_pos, y_pos in process_directory(positive_dir, label=1):
        yield X_pos, y_pos

    # Process negative examples
    for X_neg, y_neg in process_directory(negative_dir, label=0):
        yield X_neg, y_neg

'''
def load_data(positive_dir, negative_dir):
    X = []
    y = []

    # Load positive examples from all subfolders
    for root, dirs, files in os.walk(positive_dir):
        for video_name in files:
            video_path = os.path.join(root, video_name)
            frames = extract_frames(video_path)
            X.extend(frames)
            y.extend([1] * len(frames))  # 1 for positive class
            
    # Load negative examples from all subfolders
    for root, dirs, files in os.walk(negative_dir):
        for video_name in files:
            video_path = os.path.join(root, video_name)
            frames = extract_frames(video_path)
            X.extend(frames)
            y.extend([0] * len(frames))  # 0 for negative class

    return np.array(X), np.array(y)
'''
# Modify the create_model function
def create_model(input_shape):
    model = models.Sequential([
        # First convolution block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.1),
        
        # Second convolution block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Third convolution block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Modified training setup
def train_model(X_train, y_train, X_test, y_test):
    input_shape = X_train.shape[1:]
    model = create_model(input_shape)
    
    # Use learning rate scheduling
    initial_learning_rate = 0.001
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Add callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5
        )
    ]
    
    # Train with a larger batch size
    history = model.fit(
        X_train, y_train,
        epochs=50,  # Increase epochs, EarlyStopping will prevent overfitting
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        shuffle=True
    )
    
    return model, history

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
#model.compile(optimizer='adam',
#              loss='binary_crossentropy',
#              metrics=['accuracy'])

for batch_idx, (X_batch, y_batch) in enumerate(load_date(positive_dir, negative_dir)):
    model.train_on_batch(X_batch, y_batch)

    if batch_idx % 10 == 0:
        print(f"Processed {batch_idx} batches")
#history = model.fit(X_train, y_train, epochs=25, validation_split=0.1)
#model, history = train_model(X_train, y_train, X_test, y_test)

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
