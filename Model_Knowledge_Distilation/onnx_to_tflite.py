# from onnx_tf.backend import prepare
# import onnx

# # Load ONNX model
# onnx_model = onnx.load("student_model_pruned.onnx")

# # Convert to TensorFlow
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("student_model_tf")



# import tensorflow as tf

# # Load the saved model
# converter = tf.lite.TFLiteConverter.from_saved_model("student_model_tf")
# converter.optimizations = [tf.lite.Optimize.DEFAULT]  # optional: for size/speed optimization

# # Convert to TFLite
# tflite_model = converter.convert()

# # Save the TFLite model
# with open("student_model_pruned.tflite", "wb") as f:
#     f.write(tflite_model)




# from onnx_tf.backend import prepare
# import onnx

# # Load ONNX model
# onnx_model = onnx.load("student_model_new.onnx")

# # Convert to TensorFlow
# tf_rep = prepare(onnx_model)
# tf_rep.export_graph("student_model_new_tf")

# import tensorflow as tf

# # Load SavedModel directory
# converter = tf.lite.TFLiteConverter.from_saved_model("student_model_new_tf")

# # Optional: optimize for size
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Convert
# tflite_model = converter.convert()

# # Save to file
# with open("student_model_new.tflite", "wb") as f:
#     f.write(tflite_model)







from onnx_tf.backend import prepare
import onnx

# Load ONNX model
onnx_model = onnx.load("student_model_tiny.onnx")

# Convert to TensorFlow
tf_rep = prepare(onnx_model)
tf_rep.export_graph("student_model_tiny_tf")

import tensorflow as tf

# Load SavedModel directory
converter = tf.lite.TFLiteConverter.from_saved_model("student_model_tiny_tf")

# Optional: optimize for size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert
tflite_model = converter.convert()

# Save to file
with open("student_model_tiny.tflite", "wb") as f:
    f.write(tflite_model)