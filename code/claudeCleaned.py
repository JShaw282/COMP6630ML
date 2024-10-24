import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
import logging
from typing import Generator, Tuple, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def attention_block(x, filters):
    """Attention mechanism to help model focus on important features."""
    attention = layers.GlobalAveragePooling2D()(x)
    attention = layers.Dense(filters, activation='relu')(attention)
    attention = layers.Dense(filters, activation='sigmoid')(attention)
    attention = layers.Reshape((1, 1, filters))(attention)
    return layers.multiply([x, attention])

class VideoProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (224, 224),
        n_frames: int = 50,
        batch_size: int = 32
    ):
        self.target_size = target_size
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.frame_shape = (*target_size, 3)
        self.is_training = True

    def augment_frame(self, frame: np.ndarray) -> np.ndarray:
        """Apply various augmentations to the frame."""
        if not self.is_training:
            return frame

        frame = frame.copy()
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            frame = cv2.flip(frame, 1)
        
        # Random brightness
        if np.random.random() > 0.5:
            beta = np.random.uniform(-0.2, 0.2)
            frame = np.clip(frame + beta, 0, 1)
            
        # Random contrast
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            frame = np.clip(alpha * frame, 0, 1)
            
        return frame

    def process_image(self, image_path: str) -> Optional[np.ndarray]:
        """Process a single image for testing."""
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not read image: {image_path}")
                return None
            
            # Preprocess
            image = cv2.resize(image, self.target_size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32) / 255.0
            
            return image
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
            return None

    def extract_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Extract frames with improved frame selection and augmentation."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"No frames in video: {video_path}")
                return
            
            # Random frame selection
            frame_indices = np.random.choice(
                total_frames, 
                min(self.n_frames, total_frames), 
                replace=False
            )
            frame_indices.sort()
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.resize(frame, self.target_size)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = frame.astype(np.float32) / 255.0
                    
                    if self.is_training:
                        frame = self.augment_frame(frame)
                        
                    yield frame
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()

    def load_videos(
        self,
        directory: str,
        label: int,
        valid_extensions: tuple = ('.mp4', '.avi', '.mov', '.mkv')
    ) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """Load videos with improved batch handling."""
        video_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(valid_extensions):
                    video_files.append(os.path.join(root, file))
        
        np.random.shuffle(video_files)
        
        X_batch = []
        y_batch = []
        
        for video_path in tqdm(video_files, desc=f"Processing {'positive' if label==1 else 'negative'} videos"):
            try:
                frames = list(self.extract_frames(video_path))
                if not frames:
                    continue
                
                for frame in frames:
                    X_batch.append(frame)
                    y_batch.append(label)
                    
                    if len(X_batch) >= self.batch_size:
                        yield (
                            tf.convert_to_tensor(np.array(X_batch[:self.batch_size]), dtype=tf.float32),
                            tf.convert_to_tensor(np.array(y_batch[:self.batch_size]), dtype=tf.float32)
                        )
                        X_batch = X_batch[self.batch_size:]
                        y_batch = y_batch[self.batch_size:]
                        
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                continue

        if X_batch:
            if len(X_batch) < self.batch_size:
                # Pad the last batch if needed
                pad_size = self.batch_size - len(X_batch)
                X_batch.extend([X_batch[0]] * pad_size)  # Duplicate first frame
                y_batch.extend([y_batch[0]] * pad_size)
            
            yield (
                tf.convert_to_tensor(np.array(X_batch[:self.batch_size]), dtype=tf.float32),
                tf.convert_to_tensor(np.array(y_batch[:self.batch_size]), dtype=tf.float32)
            )

class VideoClassifier:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.model = self._create_model()

    def _create_model(self):
        """Create model with attention mechanisms and improved architecture."""
        inputs = layers.Input(shape=self.input_shape)
        
        # First block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(inputs)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = attention_block(x, 32)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = attention_block(x, 64)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.3)(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(0.01))(x)
        x = attention_block(x, 128)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.4)(x)
        
        # Dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        """Compile model with improved learning rate schedule."""
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.learning_rate, decay_steps=1000, decay_rate=0.9
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()
        
        self.model.compile(
            optimizer=self.optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy']
        )

    @tf.function(reduce_retracing=True)
    def train_step(self, x, y):
        """Training step with fixed input shapes."""
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.round(predictions), y), tf.float32))
        return loss, accuracy

    def predict_image(self, image: np.ndarray) -> float:
        """Predict single image."""
        image_batch = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image_batch, verbose=0)[0][0]
        return float(prediction)

    def predict_images(self, image_paths: List[str], processor: VideoProcessor) -> List[Tuple[str, float]]:
        """Predict multiple images."""
        results = []
        for image_path in tqdm(image_paths, desc="Processing test images"):
            image = processor.process_image(image_path)
            if image is not None:
                prediction = self.predict_image(image)
                results.append((os.path.basename(image_path), prediction))
        return results

    def save_model(self, filepath: str):
        self.model.save(filepath)

def train_model(config: dict):
    """Training function."""
    processor = VideoProcessor(
        target_size=config['target_size'],
        n_frames=config['n_frames'],
        batch_size=config['batch_size']
    )
    processor.is_training = True

    classifier = VideoClassifier(
        input_shape=(*config['target_size'], 3),
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    classifier.compile_model()

    try:
        batch_count = 0
        total_loss = 0
        total_accuracy = 0
        
        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            
            for label, directory in [(1, config['positive_dir']), (0, config['negative_dir'])]:
                for X_batch, y_batch in processor.load_videos(directory, label):
                    loss, accuracy = classifier.train_step(X_batch, y_batch)
                    
                    batch_count += 1
                    total_loss += float(loss)
                    total_accuracy += float(accuracy)
                    
                    if batch_count % 10 == 0:
                        avg_loss = total_loss / 10
                        avg_accuracy = total_accuracy / 10
                        logger.info(f"Batch {batch_count}: avg_loss = {avg_loss:.4f}, avg_accuracy = {avg_accuracy:.4f}")
                        total_loss = 0
                        total_accuracy = 0

        classifier.save_model(config['model_save_path'])
        logger.info(f"Model saved to {config['model_save_path']}")
        return classifier, processor

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        classifier.save_model(config['model_save_path'])
        logger.info(f"Model saved to {config['model_save_path']}")
        return classifier, processor

def test_images(model_path: str, test_directory: str, config: dict):
    """Test function for images."""
    processor = VideoProcessor(
        target_size=config['target_size'],
        n_frames=config['n_frames'],
        batch_size=config['batch_size']
    )
    processor.is_training = False

    classifier = tf.keras.models.load_model(model_path, custom_objects={'attention_block': attention_block})
    
    # Get all images in test directory
    image_paths = []
    for root, _, files in os.walk(test_directory):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))

    results = classifier.predict_images(image_paths, processor)
    
    # Print results
    logger.info("\nTest Results:")
    for filename, prediction in results:
        logger.info(f"{filename}: {prediction:.4f} - {'Contains duck' if prediction > 0.5 else 'No duck'}")

    # Print summary
    positives = sum(1 for _, pred in results if pred > 0.5)
    logger.info(f"\nProcessed {len(results)} images")
    logger.info(f"Positive predictions: {positives}")
    logger.info(f"Negative predictions: {len(results) - positives}")

def main():
    config = {
        'positive_dir': 'pos',
        'negative_dir': 'neg',
        'test_dir': 'test-images',
        'target_size': (224, 224),
        'n_frames': 50,
        'batch_size': 32,
        'learning_rate': 0.001,
        'model_save_path': 'binary_video_classification_model.h5',
        'epochs': 6
    }

    # Training phase
    logger.info("Starting training phase...")
    classifier, processor = train_model(config)

    # Testing phase
    logger.info("\nStarting testing phase...")
    test_images(config['model_save_path'], config['test-images'], config)

if __name__ == "__main__":
    main()
