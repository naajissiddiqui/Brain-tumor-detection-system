import tensorflow as tf

# Load using tf.keras instead of keras
old_model = tf.keras.models.load_model("models/model.h5", compile=False)

# Save in new format
old_model.save("models/model.keras", save_format="keras")

print("✅ Model successfully converted to model.keras")
