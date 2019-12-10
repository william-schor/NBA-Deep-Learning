import tensorflow as tf


def get_model(input_shape):
    
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.LSTM(
                500,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                return_state=False,
                input_shape=input_shape
                
            ),
            tf.keras.layers.LSTM(
                500,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                return_state=False,    
            ),
            tf.keras.layers.LSTM(
                500,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                return_sequences=True,
                return_state=False,    
            ),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ]
    )
    epochs_val = 50
    learning_rate = 0.001
    
    return model, epochs_val, learning_rate



