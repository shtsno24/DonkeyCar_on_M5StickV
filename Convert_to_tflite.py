import tensorflow as tf
import Model_Original
from tensorflow.keras.models import Model


def get_layer_index(model, layer_name):
    for i, l in enumerate(model.layers):
        if l.name == layer_name:
            return i
    return None


try:
    MODEL_FILE = "Model_Original.h5"
    MODEL_TFLITE = "Model_Original.tflite"

    with tf.device('/cpu:0'):
        # Load model
        print("\n\nLoad Model...\n")
        model = tf.keras.models.load_model(MODEL_FILE)
        output_index = []
        for n in ['throttle', 'steer']:
            i = get_layer_index(model, n)
            if i is not None:
                print(i)
                output_index.append(i)
        # model.summary()
        model_new = Model(model.input, [model.layers[x].output for x in output_index])
        model_new.summary()
        print("\nDone")

        converter = tf.lite.TFLiteConverter.from_keras_model(model_new)
        converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                tf.lite.OpsSet.SELECT_TF_OPS]
        tfmodel = converter.convert()
        with open(MODEL_TFLITE, "wb") as m:
            m.write(tfmodel)

except:
    import traceback
    traceback.print_exc()

else:
    print("Done")
