# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 19:27:57 2026

@author: manot
"""

import tensorflow as tf
import numpy as np


def generate_gradcam(
        model,
        img_array,
        last_conv_layer,
        predicted_class
):

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            model.get_layer(last_conv_layer).output,
            model.output
        ]
    )


    with tf.GradientTape() as tape:

        conv_outputs, predictions = grad_model(img_array)

        loss = predictions[:, predicted_class]


    grads = tape.gradient(
        loss,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0,1,2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]


    heatmap = tf.squeeze(
        heatmap
    )


    heatmap = np.maximum(
        heatmap,
        0
    )

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]

    heatmap = tf.squeeze(
        heatmap
    )


    heatmap = np.maximum(
        heatmap,
        0
    )


    if np.max(heatmap) != 0:
        heatmap /= np.max(heatmap)


    return heatmap