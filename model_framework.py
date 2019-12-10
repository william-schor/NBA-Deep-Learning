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


train_x = data2017[:-1]
train_y = np.array([1 if game[3] else 0 for game in wlpr2017[1:]])

test_x = data2018[:-1]
test_y = np.array([1 if game[3] else 0 for game in wlpr2018[1:]])
test_game_ids = games2018


###############
from models import dense_model
from models import convolution_model

model, epochs_val, learning_rate = dense_model.get_model()
###############


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer=optimizer,
    metrics=["binary_accuracy"],
)

model.build(train_x.shape)
model.summary()

print("training...")
history = model.fit(x=train_x, y=train_y, epochs=epochs_val, shuffle=True)

print("testing...")
score = model.evaluate(test_x, test_y, verbose=0)
print("eval score:", score)

predictions = model.predict(test_x)


all_moneylines = lines.get_line_dict("2018")
my_lines = lines.get_lines(all_moneylines, test_game_ids)

model_profit.evaluate_model(predictions, test_y, my_lines)
