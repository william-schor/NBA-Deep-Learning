import os
import tensorflow as tf
import numpy as np
import random
import nba_loss
import preprocess
import lines

from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    BatchNormalization,
    ReLU,  # I added this
    LeakyReLU,
    Reshape,
    Conv2DTranspose,
)

ACTIVE_ROSTER_SIZE = 13
NUM_STATS = 23

from nba_api.stats.static import players


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        # self.model = tf.keras.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.batch_size = 10
        self.flat = Flatten()
        self.d1 = Dense(
            1024, input_shape=(ACTIVE_ROSTER_SIZE * 2, NUM_STATS), activation="relu"
        )
        self.d2 = Dense(1024, activation="relu")
        self.d3 = Dense(512, activation="relu")
        self.d4 = Dense(256, activation="relu")
        self.d5 = Dense(1, activation="softmax")
        # self.model.add(Flatten())
        # self.model.add(
        #     Dense(1024, input_shape=(ACTIVE_ROSTER_SIZE, NUM_STATS), activation="relu")
        # )
        # self.model.add(Dense(1024, activation="relu"))
        # self.model.add(Dense(1024, activation="relu"))
        # self.model.add(Dense(1024, activation="relu"))
        # self.model.add(Dense(1024, activation="relu"))

        # self.model.add(Dense(1024, activation="relu"))
        # self.model.add(Dense(512, activation="relu"))
        # self.model.add(Dense(256, activation="relu"))
        # self.model.add(Dense(1, activation="softmax"))

    def call(self, inputs):
        o1 = self.flat(inputs)
        o2 = self.d1(o1)
        o3 = self.d2(o2)
        o4 = self.d3(o3)
        o5 = self.d4(o4)
        o6 = self.d5(o5)
        print(o6)
        return o6
        # return self.model(inputs)


def train(model, wl_per_rosters, player_matrix, line_dict):
    num_iterations = len(wl_per_rosters) // model.batch_size

    for i in range(num_iterations):
        with tf.GradientTape() as tape:
            wl_per_rosters = wl_per_rosters[::-1]
            batch_games = wl_per_rosters[
                i * model.batch_size : (i + 1) * model.batch_size
            ]

            batch_stats = []
            for game in batch_games:
                batch_stats.append(get_stats(player_matrix, game[0], game[1], game[2]))

            print(batch_stats[0])
            logits = model(
                tf.convert_to_tensor(np.array(batch_stats), dtype=tf.float32)
            )

            labels = get_labels(wl_per_rosters, i, model.batch_size)

            line_set = lines.get_lines(
                line_dict, [int(game[0]) for game in batch_games]
            )

            # loss = nba_loss.eric_loss_function(line_set, logits, labels)
            loss = nba_loss.cross_entropy_loss(logits, labels)
            print(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def get_labels(wl_per_rosters, i, scale):
    return [game[3] for game in wl_per_rosters[i * scale : (i + 1) * scale]]


def get_stats(player_matrix, game_id, roster1, roster2):
    # roster is a list of player ids
    end_matrix = []
    print(game_id)
    for player in roster1:
        print(players.find_player_by_id(player))
        print(player_matrix[player]["00" + str(int(game_id) - 1)])
        end_matrix.append(np.array(player_matrix[player]["00" + str(int(game_id) - 1)]))
    while len(end_matrix) < 13:
        end_matrix.append(np.zeros(23))
    for player in roster2:
        end_matrix.append(np.array(player_matrix[player]["00" + str(int(game_id) - 1)]))
    while len(end_matrix) < 26:
        end_matrix.append(np.zeros(23))

    import sys

    sys.exit(0)

    # stack = np.dstack(end_matrix)

    # return stack[0]


def test(model, wl_per_rosters, player_matrix, line_dict):
    stats = []
    for game in wl_per_rosters:
        stats.append(get_stats(player_matrix, game[0], game[1], game[2]))

    logits = model(tf.convert_to_tensor(np.array(stats), dtype=tf.float32))

    labels = get_labels(wl_per_rosters, 0, len(wl_per_rosters))

    line_set = lines.get_lines(line_dict, [int(game[0]) for game in wl_per_rosters])
    # loss = nba_loss.eric_loss_function(line_set, logits, labels)
    loss = nba_loss.cross_entropy_loss(logits, labels)
    return loss


def main(roster_file, matrix_file):
    print("Reading data from file...")
    player_matrix, wl_per_rosters = preprocess.get_data(roster_file, matrix_file)

    player_matrix2 = {}
    for key in player_matrix:
        player_matrix2[int(key)] = player_matrix[key]

    print("data loaded...")
    model = Model()
    print("model defined...")

    # get rid of first 104 games bc no monyline data
    wl_per_rosters = wl_per_rosters[104:]

    ###########
    # random sample of games, split into test and train set
    cut = int(len(wl_per_rosters) * 0.8)
    # random.shuffle(wl_per_rosters)
    train_games = wl_per_rosters[:cut]
    test_games = wl_per_rosters[cut:]
    print("data sets defined...")
    #############

    line_dict = lines.build_line_dict()
    print("training...")
    train(model, train_games, player_matrix2, line_dict)
    print("testing...")
    loss_val = test(model, test_games, player_matrix2, line_dict)
    print(f"Loss: {loss_val}")


if __name__ == "__main__":
    main("final_data/wl_per_rosters_2.npy", "final_data/player_dict_2.json")
