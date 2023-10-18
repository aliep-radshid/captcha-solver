import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string

from api import call_captcha
from glob import glob

img_width = 165
img_height = 45

characters = string.digits + string.ascii_uppercase

char_to_num = layers.StringLookup(
    vocabulary=list(characters), mask_token=None
)

# Mapping integers back to original characters
num_to_char = layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True
)


class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        # Compute the training-time loss value and add it
        # to the layer using `self.add_loss()`.
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * \
            tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions
        return y_pred


def build_model():
    # Inputs to the model
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    # RNNs
    x = layers.Bidirectional(layers.LSTM(
        128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(
        64, return_sequences=True, dropout=0.25))(x)

    # Output layer
    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define the model
    model = keras.models.Model(
        inputs=[input_img, labels], outputs=output, name="ocr_model_v1"
    )
    # Optimizer
    opt = keras.optimizers.Adam()
    # Compile the model and return
    model.compile(optimizer=opt, run_eagerly=True)
    return model


# Get the model
model = build_model()
model.summary()

model.load_weights('captcha_model.keras')

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model.get_layer(name="image").input, model.get_layer(name="dense2").output
)
prediction_model.summary()

# A utility function to decode the output of the network


def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :5
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text


# images = []
# for i in range(10):
#     captcha_code, img_cv = call_captcha(cv2.IMREAD_GRAYSCALE)
#     print(i)
#     cv2.imwrite(str(i) + '.jpg', img_cv)

#     print(img_cv.shape)
#     img_cv = img_cv.reshape((img_height, img_width, 1))
#     print(img_cv.shape)
#     x_img = tf.image.convert_image_dtype(img_cv, tf.float32)
#     x_img = tf.transpose(x_img, perm=[1, 0, 2])
#     print(x_img.shape)
#     images.append(x_img)

# images = np.array(images)
# preds = prediction_model.predict(images)
# pred_texts = decode_batch_predictions(preds)

# print(pred_texts)

# files = glob('./*.jpg')
# _, ax = plt.subplots(2, 4, figsize=(15, 5))
# images = []
# raw_images = []
# for f in files:
#     print(f)
#     # 1. Read image
#     x_img = tf.io.read_file(f)
#     # 2. Decode and convert to grayscale
#     x_img = tf.io.decode_jpeg(x_img, channels=1)
#     raw_images.append(x_img)
#     # 3. Convert to float32 in [0, 1] range
#     x_img = tf.image.convert_image_dtype(x_img, tf.float32)
#     x_img = tf.transpose(x_img, perm=[1, 0, 2])
#     images.append(x_img)

# images = np.array(images)
# preds = prediction_model.predict(images)
# pred_texts = decode_batch_predictions(preds)
# print(pred_texts)

tries = 0

while tries < 5:
    tries += 1
    print('try: ', tries)
    captcha_code, img_cv = call_captcha(cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('test.jpg', img_cv)

    img_cv = img_cv.reshape((img_height, img_width, 1))
    x_img = tf.image.convert_image_dtype(img_cv, tf.float32)
    x_img = tf.transpose(x_img, perm=[1, 0, 2])
    tf.expand_dims(x_img)

    preds = prediction_model.predict(x_img)
    pred_texts = decode_batch_predictions(preds)

    captcha_value = pred_texts[0] if pred_texts[0].find(
        '[UNK]') == -1 else None
    if captcha_value is None:
        continue
