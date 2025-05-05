import tensorflow as tf
from typing import Optional, Dict, Any

class FraudDetectionModel:
    def __init__(self, num_features: Optional[int] = None):
        """
        Initialize fraud detection model.
        
        Args:
            num_features: Number of input features. Must be provided before building model.
        """
        self.num_features = num_features
        self.model = None
    
    def build_model(self, compile_kwargs: Optional[Dict[str, Any]] = None) -> tf.keras.Model:
        """
        Build and compile the fraud detection model for numerical features.
        
        Args:
            compile_kwargs: Optional dictionary of arguments for model compilation.
                           Defaults to Adam optimizer with binary_crossentropy loss.
        
        Returns:
            Compiled Keras model
        
        Raises:
            ValueError: If num_features was not provided during initialization
        """
        if self.num_features is None:
            raise ValueError("num_features must be specified to build model")
            
        inputs = tf.keras.Input(shape=(self.num_features,))
        
        # Neural network architecture
        x = tf.keras.layers.Dense(
            128, 
            activation='relu',
            kernel_initializer='he_normal'
        )(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(
            64, 
            activation='relu',
            kernel_initializer='he_normal'
        )(x)
        
        # Output layer (binary classification)
        outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Default compilation parameters
        default_compile_kwargs = {
            'optimizer': tf.keras.optimizers.Adam(learning_rate=0.001),
            'loss': 'binary_crossentropy',
            'metrics': ['accuracy']
        }
        
        # Update with any user-provided kwargs
        if compile_kwargs:
            default_compile_kwargs.update(compile_kwargs)
        
        self.model.compile(**default_compile_kwargs)
        return self.model
    
    def save(self, filepath: str, **kwargs) -> None:
        """
        Save the model to disk.
        
        Args:
            filepath: Path to save the model (should end with .keras)
            **kwargs: Additional arguments to pass to model.save()
        """
        if self.model is None:
            raise RuntimeError("Model must be built before saving")
        self.model.save(filepath, **kwargs)
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is None:
            raise RuntimeError("Model must be built before showing summary")
        self.model.summary()