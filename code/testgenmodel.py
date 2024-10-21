import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import os

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

# Main execution
if __name__ == "__main__":
    # Load the saved model
    try:
        model = tf.keras.models.load_model('binary_video_classification_model.h5')
    except Exception as e:
        print(f"Error loading the model: {str(e)}")
        exit(1)

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
