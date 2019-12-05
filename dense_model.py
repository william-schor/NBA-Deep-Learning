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


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.model = tf.keras.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.batch_size = 10
        self.model.add(Flatten())
        self.model.add(
            Dense(1024, input_shape=(ACTIVE_ROSTER_SIZE, NUM_STATS), activation="relu")
        )
        self.model.add(Dense(512, activation="relu"))
        self.model.add(Dense(256, activation="relu"))
        self.model.add(Dense(1, activation="softmax"))

    def call(self, inputs):
        return self.model(inputs)


def train(model, wl_per_rosters, player_matrix, line_dict):
    num_iterations = len(wl_per_rosters) // model.batch_size

    for i in range(num_iterations):
        print(f"batch number: {i}")
        with tf.GradientTape() as tape:

            batch_games = wl_per_rosters[
                i * model.batch_size : (i + 1) * model.batch_size
            ]

            batch_stats = []
            for game in batch_games:
                batch_stats.append(get_stats(player_matrix, game[0], game[1], game[2]))

            batch_stats = np.array(batch_stats)
            print(batch_stats.shape)
            import sys

            sys.exit(0)

            logits = model(
                tf.convert_to_tensor(np.array(batch_stats), dtype=tf.float32)
            )

            labels = get_labels(wl_per_rosters, i, model.batch_size)

            ## Here are your lines!
            line_set = lines.get_lines(line_dict, batch_games)

            loss = nba_loss.loss_function(line_set, logits, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def get_labels(wl_per_rosters, i, scale):
    return [game[3] for game in wl_per_rosters[i * scale : (i + 1) * scale]]


def get_stats(player_matrix, game_id, roster1, roster2):
    # roster is a list of player ids
    end_matrix = []
    for player in roster1:
        # may contains None
        end_matrix.append(np.array(player_matrix[player][game_id]))
    for player in roster2:
        # may contains None
        end_matrix.append(np.array(player_matrix[player][game_id]))
    stack = np.dstack(end_matrix)
    print(f"EM shape: {stack.shape}")
    return stack


def test(model, test_games, test_labels, player_matrix):
    stats = [get_stats(player_matrix, game[0], game[1], game[2]) for game in test_games]
    labels = get_labels(test_games, 0, len(test_games))
    logits = model(stats)
    loss = nba_loss.loss_function(line, logits, labels)
    return loss


def main(roster_file, matrix_file):
    player_matrix, wl_per_rosters = preprocess.get_data(roster_file, matrix_file)

    player_matrix2 = {}
    for key in player_matrix:
        player_matrix2[int(key)] = player_matrix[key]

    print("data loaded...")
    model = Model()
    print("model defined...")
    print("training...")
    line_dict = lines.build_line_dict()
    train(model, wl_per_rosters, player_matrix2, line_dict)
    loss_val = test(model, test_games, test_labels, player_matrix)


if __name__ == "__main__":
    main("final_data/wl_per_rosters_2.npy", "final_data/player_dict_2.json")
