import tensorflow as tf
from tensorflow.python.client import device_lib
import Model_Original

device_list = device_lib.list_local_devices()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)

try:
    # Load model
    print("Load Model...\n\n")
    model = Model_Original.Categorical()
    model.summary()
    print("\nDone")

    # Freeze model
    print("\n\nFreeze Model...")
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[tf.keras.metrics.MeanSquaredError()])
    print("  Done\n\n")

    # Save model
    print("\n\nSave Model...")
    model.save('Model_Original.h5')
    print("  Done\n\n")

except:
    import traceback
    traceback.print_exc()

finally:
    print("Done")
