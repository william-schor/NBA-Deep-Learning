import tensorflow as tf
import numpy as np
import preprocess
import lines
import model_profit
from models import convolution_model
from models import dense_model
from models import rnn_model
import argparse



print("reading file...")
pd, wlpr = preprocess.get_data(
    "final_data/wl_per_rosters_2.npy", "final_data/player_dict_2.json"
)

player_matrix2 = {}
for key in pd:
    player_matrix2[int(key)] = pd[key]

all_data, games = preprocess.get_2d_data(wlpr, player_matrix2)

all_data = all_data[105:]
games = np.array(games[105:])
wlpr = wlpr[105:]

indices = np.random.permutation(len(all_data))
all_data = all_data[indices]
wlpr = wlpr[indices]
games = games[indices]



# random sample of games, split into test and train set
cut = int(len(all_data) * 0.8)

train_x = all_data[:cut]
train_y = np.array([1 if game[3] else 0 for game in wlpr[:cut]])
train_game_ids = games[:cut]

test_x = all_data[cut:]
test_y = np.array([1 if game[3] else 0 for game in wlpr[cut:]])
test_game_ids = games[cut:]


# model = tf.keras.models.Sequential(
#     [
#         # tf.keras.layers.Flatten(input_shape=train_x[0].shape),
#         tf.keras.layers.Dense(4096, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Dense(2048, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Dense(512, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Dense(256, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Dense(128, activation=tf.nn.relu),
#         tf.keras.layers.Dropout(0.1),
#         tf.keras.layers.Dense(1, activation='sigmoid'),
#     ]
# )

print("before", train_x.shape)
tf.expand_dims(train_x, 1)
print("after", train_x.shape)

model, epochs_val, learning_rate = convolution_model.get_model(train_x.shape)

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    metrics=["binary_accuracy"],
)

model.build(train_x.shape)
model.summary()



history = model.fit(x=train_x, y=train_y, epochs=epochs_val, shuffle=True)

score = model.evaluate(test_x, test_y)
print("eval score:", score)

predictions = model.predict(test_x)

all_moneylines = lines.build_line_dict()
my_lines = lines.get_lines(all_moneylines, test_game_ids)

model_profit.evaluate_model(predictions, test_y, my_lines)





