import os
import tensorflow as tf

# Load your trained model
model_filename = "dermbot2.h5"
model_path = os.path.join(os.getcwd(), model_filename)
trained_model = tf.keras.models.load_model(model_path)

# Export the model to SavedModel format
export_path = r"C:\Users\Bhushan\Desktop\bot dock"  # Specify the directory to save the exported model
tf.saved_model.save(trained_model, export_path)

