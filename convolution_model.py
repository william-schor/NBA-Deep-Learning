

def get_model():
    model = tf.keras.models.Sequential(
        [
            # tf.keras.layers.Flatten(input_shape=train_x[0].shape),
            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(2048, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(256, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    )
    return model



