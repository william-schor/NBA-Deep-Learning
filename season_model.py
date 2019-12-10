import tensorflow as tf
import numpy as np
import preprocess
import lines
import model_profit


print("loading data...")
pd2017, wlpr2017 = preprocess.get_data(
    "final_data/wl_per_rosters_2017.npy", "final_data/player_dict_2017.json"
)
pd2018, wlpr2018 = preprocess.get_data(
    "final_data/wl_per_rosters_2018.npy", "final_data/player_dict_2018.json"
)

conv_pd2017 = {}
for key in pd2017:
    conv_pd2017[int(key)] = pd2017[key]

conv_pd2018 = {}
for key in pd2018:
    conv_pd2018[int(key)] = pd2018[key]

print("building data...")
data2017, games2017 = preprocess.get_2d_data(wlpr2017, conv_pd2017)
data2018, games2018 = preprocess.get_2d_data(wlpr2018, conv_pd2018)


train_x = data2017
train_y = np.array([1 if game[3] else 0 for game in wlpr2017])

test_x = data2018
test_y = np.array([1 if game[3] else 0 for game in wlpr2018])
test_game_ids = games2018
###############


model = tf.keras.models.Sequential(
    [
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
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

# optimizer = tf.keras.optimizers.RMSprop(0.0001)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
# loss_1 = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    # loss="mean_squared_error",  # mean_squared_error or binary_crossentropy
    optimizer=optimizer,
    metrics=["binary_accuracy"],
)

model.build(train_x.shape)
model.summary()


history = model.fit(x=train_x, y=train_y, epochs=20, shuffle=True)


score = model.evaluate(test_x, test_y, verbose=0)
print("eval score:", score)


predictions = model.predict(test_x)

correct = 0
for pred, val in zip(predictions, test_y):
    if pred <= 0.5:
        if val == 0:
            correct += 1
    if pred > 0.5:
        if val == 1:
            correct += 1

print("My score:", correct / len(predictions))
print("---------------------------------------------------------")


all_moneylines = lines.get_line_dict("2018")
my_lines = lines.get_lines(all_moneylines, test_game_ids)

model_profit.evaluate_model(predictions, test_y, my_lines)
