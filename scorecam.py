# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 12:26:26 2026

@author: manot
"""

# -*- coding: utf-8 -*-
"""
Score-CAM implementation
Medical-XAI-Lab

"""

import numpy as np
import cv2
import tensorflow as tf


def generate_scorecam(
        model,
        img_array,
        last_conv_layer_name,
        class_index,
        max_N=32):

    """
    Generate Score-CAM heatmap

    Parameters:
    model:
        trained keras model

    img_array:
        preprocessed input image

    last_conv_layer_name:
        last convolutional layer

    class_index:
        predicted class index

    max_N:
        maximum number of feature maps used

    Returns:
        heatmap
    """


    # Get feature extraction model

    conv_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer(
            last_conv_layer_name
        ).output
    )


    # Extract activation maps

    conv_output = conv_model(
        img_array
    )


    conv_output = conv_output.numpy()


    feature_maps = conv_output[0]


    # Limit number of maps

    if feature_maps.shape[-1] > max_N:

        feature_maps = feature_maps[:, :, :max_N]


    heatmap = np.zeros(
        feature_maps.shape[:2]
    )


    scores = []


    # Normalize feature maps

    for i in range(
        feature_maps.shape[-1]
    ):

        fmap = feature_maps[:, :, i]


        # Resize feature map to input image size
        fmap = cv2.resize(
            fmap,
            (img_array.shape[2],
             img_array.shape[1])
        )

    # Remove negative values
    fmap = np.maximum(
        fmap,
        0
    )


    # Normalize feature map
    if np.max(fmap) != 0:

        fmap = fmap / np.max(fmap)


        # Convert 2D map to 3-channel mask
        mask = np.expand_dims(
            fmap,
            axis=-1
        )


        mask = np.repeat(
            mask,
            3,
            axis=-1
        )


        # Apply mask to image
        masked_img = img_array[0] * mask


        masked_img = np.expand_dims(
            masked_img,
            axis=0
        )


        prediction = model.predict(
            masked_img,
            verbose=0
        )


        score = prediction[0][class_index]


        scores.append(
            score
        )


        heatmap += score * feature_maps[:, :, i]


        scores.append(
            score
        )


        heatmap += score * feature_maps[:, :, i]

    # Normalize final heatmap

    heatmap = np.maximum(
        heatmap,
        0
    )


    heatmap = heatmap / np.max(
        heatmap
    )


    return heatmap