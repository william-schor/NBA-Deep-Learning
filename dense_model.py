import os
import tensorflow as tf
import numpy as np
import random
import nba_loss

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
NUM_STATS = 20


class Model(tf.keras.Model):
    def __init__(self):
        super(Model, self).__init__()

        self.model = tf.Sequential()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.batch_size = 10

        model.add(
            Dense(1024, input_shape=(ACTIVE_ROSTER_SIZE, NUM_STATS), activation="relu")
        )
        model.add(Dense(512, activation="relu"))
        model.add(Dense(256, activation="relu"))
        model.add(Dense(1, activation="softmax"))

    def call(self, inputs):
        return self.model(inputs)


def train(model, wl_per_rosters, player_matrix):
    num_iterations = len(wl_per_rosters) // model.batch_size

    for i in range(num_iterations):
        with tf.GradientTape() as tape:

            batch_games = wl_per_rosters[i * batch_size : (i + 1) * batch_size]
            batch_stats = [
                get_stats(player_matrix, game[0], game[1], game[2])
                for game in batch_games
            ]
            labels = get_labels(wl_per_rosters, i, model.batch_size)
            logits = model(batch_stats)
            loss = nba_loss.loss_function(line, logits, labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def get_labels(wl_per_rosters, i, scale):
    return [game[4] for games in wl_per_rosters[i * scale : (i + 1) * scale]]


def get_stats(player_matrix, game_id, roster1, roster2):
    # roster is a list of player ids
    end_matrix = []
    for player in roster1:
        # may contains None
        end_matrix.append(player_matrix[player][game_id])
    for player in roster2:
        # may contains None
        end_matrix.append(player_matrix[player][game_id])
    return end_matrix


def test(model, test_games, test_labels, player_matrix):
    stats = [get_stats(player_matrix, game[0], game[1], game[2]) for game in test_games]
    labels = get_labels(test_games, 0, len(test_games))
    logits = model(stats)
    loss = nba_loss.loss_function(line, logits, labels)


def main():
    pass


if __name__ == "__main__":
    main()
