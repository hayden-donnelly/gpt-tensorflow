import tensorflow as tf

available_gpus = tf.config.list_physical_devices('GPU')
print('\n')
print('Available GPUs:')
print(available_gpus)
print('\n')
