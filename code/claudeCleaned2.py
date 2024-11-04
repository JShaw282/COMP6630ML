import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tqdm import tqdm
import logging
import gc
from typing import Generator, Tuple, List, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SpatialAttention(layers.Layer):
    """Memory-optimized spatial attention mechanism."""
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)
        self.conv = layers.Conv2D(1, kernel_size=3, padding='same')

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = tf.sigmoid(self.conv(concat))
        return x * attention

class ChannelAttention(layers.Layer):
    """Memory-optimized channel attention mechanism."""
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

class VideoProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (128, 128),
        n_frames: int = 8,
        batch_size: int = 2
    ):
        self.target_size = target_size
        self.n_frames = n_frames
        self.batch_size = batch_size
        self.frame_shape = (*target_size, 3)
        self.is_training = True
        self.object_detector = cv2.createBackgroundSubtractorMOG2(
            history=100,
            varThreshold=16,
            detectShadows=False
        )

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Memory-efficient frame preprocessing."""
        if frame.dtype != np.uint8:
            frame = (frame * 255).astype(np.uint8)
            
        frame = cv2.resize(frame, self.target_size)
        
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        elif frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        fg_mask = self.object_detector.apply(frame)
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        fg_mask = (fg_mask > 128).astype(np.float32)
        
        frame = frame.astype(np.float32) / 255.0
        frame *= np.expand_dims(fg_mask * 0.7 + 0.3, -1)
        
        return frame.astype(np.float32)

    def extract_frames(self, video_path: str) -> Generator[np.ndarray, None, None]:
        """Memory-efficient frame extraction."""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                logger.warning(f"No frames in video: {video_path}")
                return

            n_frames = min(self.n_frames, total_frames)
            frame_indices = np.linspace(0, total_frames-1, n_frames, dtype=int)
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = self.preprocess_frame(frame)
                    yield frame
                
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

    def load_videos(
        self,
        directory: str,
        label: int,
        valid_extensions: tuple = ('.mp4', '.avi', '.mov')
    ) -> Generator[Tuple[tf.Tensor, tf.Tensor], None, None]:
        """Memory-efficient video loading with streaming."""
        video_paths = [
            os.path.join(root, f) 
            for root, _, files in os.walk(directory)
            for f in files if f.lower().endswith(valid_extensions)
        ]
        np.random.shuffle(video_paths)

        current_batch = {'X': [], 'y': []}
        
        for video_path in tqdm(video_paths, desc=f"Processing {'positive' if label==1 else 'negative'} videos"):
            try:
                for frame in self.extract_frames(video_path):
                    current_batch['X'].append(frame)
                    # Convert label to float32 and correct shape (1,)
                    current_batch['y'].append(np.array([label], dtype=np.float32))
                    
                    if len(current_batch['X']) >= self.batch_size:
                        X_tensor = tf.convert_to_tensor(current_batch['X'], dtype=tf.float32)
                        # Stack labels and reshape to (batch_size, 1)
                        y_tensor = tf.stack(current_batch['y'])
                        y_tensor = tf.reshape(y_tensor, (-1, 1))
                        
                        yield X_tensor, y_tensor
                        current_batch = {'X': [], 'y': []}
                        gc.collect()
                        
            except Exception as e:
                logger.error(f"Error processing {video_path}: {str(e)}")
                continue

        if current_batch['X']:
            while len(current_batch['X']) < self.batch_size:
                current_batch['X'].append(current_batch['X'][0])
                current_batch['y'].append(np.array([current_batch['y'][0][0]], dtype=np.float32))
            
            X_tensor = tf.convert_to_tensor(current_batch['X'][:self.batch_size], dtype=tf.float32)
            y_tensor = tf.stack(current_batch['y'][:self.batch_size])
            y_tensor = tf.reshape(y_tensor, (-1, 1))
            
            yield X_tensor, y_tensor

class VideoClassifier:
    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        batch_size: int = 2,
        learning_rate: float = 0.0005
    ):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        # Initialize metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.BinaryAccuracy(name='train_accuracy')
        
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                    tf.config.experimental.enable_tensor_float_32_execution(False)
                except RuntimeError as e:
                    logger.warning(f"Error setting memory growth: {e}")
        
        self.model = self._create_model()

    def _create_model(self):
        """Create memory-optimized model with reduced parameters."""
        inputs = layers.Input(shape=self.input_shape)
        
        # First block - reduced filters
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Second block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = SpatialAttention()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Third block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = ChannelAttention()(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Dense layers - reduced size
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        return models.Model(inputs=inputs, outputs=outputs)

    def compile_model(self):
        """Compile model with memory-efficient settings."""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            epsilon=1e-7
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy()],
            jit_compile=True
        )

    def train_step(self, x, y):
        """Memory-efficient training step with proper shape handling."""
        if not isinstance(x, tf.Tensor):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
        if not isinstance(y, tf.Tensor):
            y = tf.convert_to_tensor(y, dtype=tf.float32)
        
        # Ensure y has shape (batch_size, 1)
        if len(y.shape) == 1:
            y = tf.expand_dims(y, axis=-1)
        
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss = tf.keras.losses.binary_crossentropy(y, y_pred)
            loss = tf.reduce_mean(loss)
        
        # Calculate gradients and update weights
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(loss)
        self.train_accuracy.update_state(y, y_pred)
        
        return float(loss), float(self.train_accuracy.result())

    def save_model(self, filepath: str):
        """Save the model to disk."""
        try:
            self.model.save(filepath)
            logger.info(f"Model saved successfully to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model to {filepath}: {e}")

    @staticmethod
    def clear_memory():
        """Clear memory after batch processing."""
        tf.keras.backend.clear_session()
        gc.collect()
        
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

def train_model(config: dict):
    """Memory-efficient training function."""
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
                tf.config.experimental.enable_tensor_float_32_execution(False)
            except RuntimeError as e:
                logger.warning(f"Error setting memory growth: {e}")

    processor = VideoProcessor(
        target_size=config['target_size'],
        n_frames=config['n_frames'],
        batch_size=config['batch_size']
    )
    
    classifier = VideoClassifier(
        input_shape=(*config['target_size'], 3),
        batch_size=config['batch_size'],
        learning_rate=config['learning_rate']
    )
    classifier.compile_model()

    try:
        for epoch in range(config['epochs']):
            logger.info(f"Epoch {epoch + 1}/{config['epochs']}")
            
            # Reset metrics at the start of each epoch
            classifier.train_loss.reset_state()
            classifier.train_accuracy.reset_state()
            
            for label, directory in [(1, config['positive_dir']), (0, config['negative_dir'])]:
                batch_count = 0
                
                for X_batch, y_batch in processor.load_videos(directory, label):
                    try:
                        loss, accuracy = classifier.train_step(X_batch, y_batch)
                        
                        batch_count += 1
                        if batch_count % 10 == 0:
                            logger.info(
                                f"Batch {batch_count}: "
                                f"loss = {classifier.train_loss.result():.4f}, "
                                f"accuracy = {classifier.train_accuracy.result():.4f}"
                            )
                            classifier.clear_memory()
                            
                    except tf.errors.ResourceExhaustedError as e:
                        logger.warning(f"Memory error encountered: {e}")
                        classifier.clear_memory()
                        continue
                    
                    except tf.errors.InvalidArgumentError as e:
                        logger.warning(f"Invalid argument error: {e}")
                        continue
                
                # Log metrics for this class
                logger.info(
                    f"Class {label}: "
                    f"avg_loss = {classifier.train_loss.result():.4f}, "
                    f"avg_accuracy = {classifier.train_accuracy.result():.4f}"
                )
            
            # Save model after each epoch
            classifier.save_model(f"{epoch+1}{config['model_save_path']}")
            
        return classifier, processor

    except KeyboardInterrupt:
        logger.info("Training interrupted")
        classifier.save_model(config['model_save_path'])
        return classifier, processor

def main():
    # Set global TF configuration
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(2)

    config = {
        'positive_dir': 'pos',
        'negative_dir': 'neg',
        'test_dir': 'test-images',
        'target_size': (128, 128),
        'n_frames': 8,
        'batch_size': 2,
        'learning_rate': 0.0005,
        'model_save_path': 'binary_video_classification_model.keras',
        'epochs': 5
    }

    train_model(config)

if __name__ == "__main__":
    main()

