import tensorflow as tf
import sys
import numpy as np

BET_SIZE = 100


def loss_function(lines, predictions, team1_wins):
    profits = []
    for line, prediction, team1_win in zip(lines, predictions, team1_wins):
        team1_odds = calc_team1_odds(line)
        if team1_odds > prediction:
            # bet on team 2
            if team1_win == 1:
                profit = -BET_SIZE
            else:
                profit = (us2dec(line) - 1) * BET_SIZE
        else:
            # bet on team 1
            if team1_win == 1:
                profit = (us2dec(line) - 1) * BET_SIZE
            else:
                profit = -BET_SIZE
        profits.append(profit)

    avg_profit = tf.reduce_mean(profits)
    return 1 / (avg_profit + BET_SIZE + 1)  ## want to minimize


def calc_team1_odds(line):
    if line > 0:
        line = abs(line)
        team1_odds = 100 / (100 + line)
    else:
        line = abs(line)
        team1_odds = line / (100 + line)
    return team1_odds


def us2dec(line):
    if line > 0:
        dec = 1 + (line / 100)
    else:
        dec = 1 - (100 / line)
    return dec


def eric_loss_function(lines, predictions, team1_wins):
    loss = []
    c = 0.5
    for line, prediction, team1_win in zip(lines, predictions, team1_wins):
        a = tf.math.square(prediction - team1_win)
        b = c * tf.math.square(prediction - (1 / calc_team1_odds(line)))
        loss.append(a - b)
    avg_loss = tf.reduce_mean(loss)
    return avg_loss


def cross_entropy_loss(predictions, team1_wins):
    # predictions = [1.0 if x > .5 else 0.0 for x in predictions]
    # team1_wins = [1.0 if t else 0.0 for t in team1_wins]
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(team1_wins, predictions)
