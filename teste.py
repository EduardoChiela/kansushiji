import tensorflow as tf
print("Placas de vídeo detectadas:", len(tf.config.list_physical_devices('GPU')))