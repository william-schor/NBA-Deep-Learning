import tensorflow as tf

def get_model(in_val):

    model = tf.keras.models.Sequential(
        [
            
            tf.keras.layers.Conv1D(filters=5,
                kernel_size=23,
                strides=23,
                padding='valid',
                activation='relu',
                use_bias=True,
                input_shape=in_val   
            ),

            tf.keras.layers.Dense(4096, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.1),
            
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    )
    epochs = 20
    learning_rate = 0.001

    return model, epochs, learning_rate 



