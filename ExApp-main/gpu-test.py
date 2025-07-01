import tensorflow as tf 
print("TF 版本:", tf.__version__)
print("可用實體 GPU:", tf.config.list_physical_devices("GPU"))
