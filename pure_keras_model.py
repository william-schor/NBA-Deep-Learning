import tensorflow as tf
import numpy as np
import preprocess


print("reading file...")
pd, wlpr = preprocess.get_data(
    "final_data/wl_per_rosters_2.npy", "final_data/player_dict_2.json"
)

player_matrix2 = {}
for key in pd:
    player_matrix2[int(key)] = pd[key]

all_data = preprocess.get_2d_data(wlpr, player_matrix2)


indices = np.random.permutation(len(all_data))
all_data = all_data[indices]
wlpr = wlpr[indices]

# random sample of games, split into test and train set
cut = int(len(all_data) * 0.8)

train_x = all_data[:cut]
train_y = np.array([1 if game[3] else 0 for game in wlpr[:cut]])


test_x = all_data[cut:]
test_y = np.array([1 if game[3] else 0 for game in wlpr[cut:]])


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=train_x[0].shape),
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
        tf.keras.layers.Dense(1),
    ]
)

# optimizer = tf.keras.optimizers.RMSprop(0.0001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    loss="mean_squared_error",  # mean_squared_error or binary_crossentropy
    optimizer=optimizer,
    metrics=["binary_accuracy"],
)

model.summary()


history = model.fit(x=train_x, y=train_y, epochs=75, shuffle=True)

score = model.evaluate(test_x, test_y)

print(score)