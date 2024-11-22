import tensorflow as tf
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple
import logging
from tensorflow.keras.layers import Layer
from tensorflow.keras import layers


@tf.keras.utils.register_keras_serializable()
class SpatialAttention(Layer):
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=3, padding='same')

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = tf.sigmoid(self.conv(concat))
        return x * attention

    def get_config(self):
        config = super().get_config()
        return config

@tf.keras.utils.register_keras_serializable()
class ChannelAttention(Layer):
    def __init__(self, ratio=16, **kwargs):
        super(ChannelAttention, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        channel = input_shape[-1]
        reduced_dim = max(channel // self.ratio, 8)
        self.shared_dense_1 = layers.Dense(reduced_dim, activation='relu')
        self.shared_dense_2 = layers.Dense(channel)
        super(ChannelAttention, self).build(input_shape)

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        avg_out = self.shared_dense_2(self.shared_dense_1(avg_pool))
        max_out = self.shared_dense_2(self.shared_dense_1(max_pool))
        attention = tf.sigmoid(avg_out + max_out)
        return x * attention

    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config

# Add to evaluation file only:
def load_model_with_custom_objects(model_path: str):
    custom_objects = {
        'SpatialAttention': SpatialAttention,
        'ChannelAttention': ChannelAttention
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

def load_and_preprocess_image(
    image_path: str,
    target_size: Tuple[int, int] = (128, 128)
) -> np.ndarray:
    """Load and preprocess a single image for prediction."""
    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image at {image_path}")
            
        # Resize
        image = cv2.resize(image, target_size)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
        
    except Exception as e:
        logging.error(f"Error processing image {image_path}: {str(e)}")
        return None

def test_model(
    model_path: str,
    test_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    confidence_threshold: float = 0.5
) -> List[Dict]:
    """
    Test a trained model on images in a directory.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test images
        target_size: Size to resize images to
        confidence_threshold: Threshold for positive prediction
        
    Returns:
        List of dictionaries containing results for each image
    """
    try:
        # Load model
        model = load_model_with_custom_objects(model_path)
        logging.info(f"Model loaded successfully from {model_path}")
        
        # Get list of image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            f for f in os.listdir(test_dir)
            if f.lower().endswith(valid_extensions)
        ]
        
        if not image_files:
            logging.warning(f"No valid images found in {test_dir}")
            return []
        
        results = []
        
        # Process each image
        for image_file in image_files:
            image_path = os.path.join(test_dir, image_file)
            
            # Load and preprocess image
            processed_image = load_and_preprocess_image(image_path, target_size)
            if processed_image is None:
                continue
                
            try:
                # Make prediction
                prediction = model.predict(processed_image, verbose=0)[0][0]
                
                # Store results
                result = {
                    'image_name': image_file,
                    'prediction_score': float(prediction),
                    'predicted_class': 'positive' if prediction >= confidence_threshold else 'negative',
                    'confidence': float(prediction if prediction >= 0.5 else 1 - prediction)
                }
                
                results.append(result)

                print(
                    f"Image: {image_file}\n"
                    f"Prediction Score: {prediction:.4f}\n"
                    f"Class: {result['predicted_class']}\n"
                    f"Confidence: {result['confidence']:.2%}\n"
                    f"{'-' * 40}"
                )
                
                logging.info(
                    f"Processed {image_file}: "
                    f"class={result['predicted_class']}, "
                    f"confidence={result['confidence']:.2%}"
                )
                
            except Exception as e:
                logging.error(f"Error predicting {image_file}: {str(e)}")
                continue
        
        # Print summary
        pos_count = sum(1 for r in results if r['predicted_class'] == 'positive')
        total_count = len(results)
        
        print("\nTesting Summary:")
        print(f"Total images processed: {total_count}")
        print(f"Positive predictions: {pos_count}")
        print(f"Negative predictions: {total_count - pos_count}")
        
        # Print detailed results
        print("\nDetailed Results:")
        print("-" * 80)
        print(f"{'Image Name':<30} {'Predicted Class':<15} {'Confidence':<10}")
        print("-" * 80)
        
        for result in results:
            print(
                f"{result['image_name']:<30} "
                f"{result['predicted_class']:<15} "
                f"{result['confidence']:.2%}"
            )
        
        return results
        
    except Exception as e:
        logging.error(f"Error in test_model: {str(e)}")
        return []

def visualize_results(
    model_path: str,
    test_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (128, 128),
    confidence_threshold: float = 0.5
):
    """
    Test model and save visualizations of the results.
    
    Args:
        model_path: Path to the saved model
        test_dir: Directory containing test images
        output_dir: Directory to save visualized results
        target_size: Size to resize images to
        confidence_threshold: Threshold for positive prediction
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load model
        model = load_model_with_custom_objects(model_path)
        #model = tf.keras.models.load_model(model_path)
        
        # Get list of image files
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        image_files = [
            f for f in os.listdir(test_dir)
            if f.lower().endswith(valid_extensions)
        ]
        
        for image_file in image_files:
            image_path = os.path.join(test_dir, image_file)
            
            # Load image for visualization
            original_image = cv2.imread(image_path)
            if original_image is None:
                continue
                
            # Load and preprocess image for prediction
            processed_image = load_and_preprocess_image(image_path, target_size)
            if processed_image is None:
                continue
                
            # Make prediction
            prediction = model.predict(processed_image, verbose=0)[0][0]
            predicted_class = 'positive' if prediction >= confidence_threshold else 'negative'
            confidence = float(prediction if prediction >= 0.5 else 1 - prediction)
            
            # Draw results on image
            height, width = original_image.shape[:2]
            text_scale = min(width, height) / 1000.0
            thickness = int(min(width, height) / 500.0)
            
            # Create colored border based on prediction
            color = (0, 255, 0) if predicted_class == 'positive' else (0, 0, 255)
            bordered_image = cv2.copyMakeBorder(
                original_image,
                10, 10, 10, 10,
                cv2.BORDER_CONSTANT,
                value=color
            )
            
            # Add text with prediction
            cv2.putText(
                bordered_image,
                f"{predicted_class.upper()} ({confidence:.2%})",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                color,
                thickness
            )
            
            # Save result
            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(image_file)[0]}_result{os.path.splitext(image_file)[1]}"
            )
            cv2.imwrite(output_path, bordered_image)
            
            logging.info(f"Saved visualization for {image_file}")
            
    except Exception as e:
        logging.error(f"Error in visualize_results: {str(e)}")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Configuration
    config = {
        'model_path': 'binary_video_classification_model.keras',  # Update with your model path
        'test_dir': 'test-images',
        'output_dir': 'test-results',
        'target_size': (128, 128),
        'confidence_threshold': 0.5
    }
    
    # Run tests and get results
    results = test_model(
        config['model_path'],
        config['test_dir'],
        config['target_size'],
        config['confidence_threshold']
    )
    
    # Generate visualizations
    visualize_results(
        config['model_path'],
        config['test_dir'],
        config['output_dir'],
        config['target_size'],
        config['confidence_threshold']
    )
