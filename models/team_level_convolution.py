import tensorflow as tf


def get_model():

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Reshape([598, 1, 1]),
            tf.keras.layers.Conv2D(
                filters=10,
                kernel_size=[23, 1],
                strides=23,
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            tf.keras.layers.Conv2D(
                filters=3,
                kernel_size=[13, 1],
                strides=13,
                padding="valid",
                activation="relu",
                use_bias=True,
            ),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    epochs = 40
    learning_rate = 0.001

    return model, epochs, learning_rate
