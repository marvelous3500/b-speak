import tensorflow as tf
from tensorflow.keras.models import save_model

# 1. Load the SavedModel the correct way
model = tf.saved_model.load("models/fraud_model")

# 2. Get the concrete function for serving
serve = model.signatures['serving_default']

# 3. Convert to Keras model
inputs = [tf.keras.Input(shape=(), dtype=tf.string)]  # Adjust shape as needed
outputs = serve(tf.cast(inputs[0], tf.float32))  # May need adjustment
keras_model = tf.keras.Model(inputs, outputs)

# 4. Save in .keras format
save_model(keras_model, "models/fraud_model.keras")
print("✅ Successfully converted to fraud_model.keras")