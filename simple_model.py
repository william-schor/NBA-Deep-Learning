from __future__ import absolute_import
from matplotlib import pyplot as plt
from preprocess import get_data
from convolution import conv2d

import os
import tensorflow as tf
import numpy as np
import random

import nba_loss
import lines
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class Model(tf.keras.Model):
    def __init__(self):
        """
    This model class will contain the architecture for your CNN that 
        classifies images. Do not modify the constructor, as doing so 
        will break the autograder. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """
        super(Model, self).__init__()

        self.batch_size = 64
        self.epochs = 10
        self.num_classes = 2
        self.epsilon = 0.001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.epsilon)

        self.filter1 = tf.Variable(
            tf.random.normal([5, 5, 3, 16], stddev=0.1),
            dtype=tf.float32,
            name="filter1",
        )
        self.conv_bias1 = tf.Variable(
            tf.random.normal([16]), dtype=tf.float32, name="conv_bias1"
        )

        self.filter2 = tf.Variable(
            tf.random.normal([5, 5, 16, 20], stddev=0.1),
            dtype=tf.float32,
            name="filter2",
        )
        self.conv_bias2 = tf.Variable(
            tf.random.normal([20]), dtype=tf.float32, name="conv_bias2"
        )

        self.filter3 = tf.Variable(
            tf.random.normal([5, 5, 20, 20], stddev=0.1),
            dtype=tf.float32,
            name="filter3",
        )
        self.conv_bias3 = tf.Variable(
            tf.random.normal([20]), dtype=tf.float32, name="conv_bias3"
        )

        self.W1 = tf.Variable(
            tf.random.truncated_normal(shape=[320, 64], stddev=0.1),
            dtype=tf.float32,
            name="W1",
        )
        self.b1 = tf.Variable(
            tf.random.truncated_normal(shape=[1, 64], stddev=0.1),
            dtype=tf.float32,
            name="b1",
        )
        self.W2 = tf.Variable(
            tf.random.truncated_normal(shape=[64, 32], stddev=0.1),
            dtype=tf.float32,
            name="W2",
        )
        self.b2 = tf.Variable(
            tf.random.truncated_normal(shape=[1, 32], stddev=0.1),
            dtype=tf.float32,
            name="b2",
        )
        self.W3 = tf.Variable(
            tf.random.truncated_normal(shape=[32, self.num_classes], stddev=0.1),
            dtype=tf.float32,
            name="W3",
        )
        self.b3 = tf.Variable(
            tf.random.truncated_normal(shape=[1, self.num_classes], stddev=0.1),
            dtype=tf.float32,
            name="b3",
        )

    def call(self, inputs, is_testing=False):
        """
        Runs a forward pass on an input batch of images.
        :param inputs: images, shape of (num_inputs, 32, 32, 3); during training, the shape is (batch_size, 32, 32, 3)
        :param is_testing: a boolean that should be set to True only when you're doing Part 2 of the assignment and this function is being called during testing
        :return: logits - a matrix of shape (num_inputs, num_classes); during training, it would be (batch_size, 2)
        """
        # Remember that
        # shape of input = (num_inputs (or batch_size), in_height, in_width, in_channels)
        # shape of filter = (filter_height, filter_width, in_channels, out_channels)
        # shape of strides = (batch_stride, height_stride, width_stride, channels_stride)

        #### LAYER 1 ####
        conv1 = tf.nn.conv2d(
            inputs, self.filter1, [1, 2, 2, 1], padding="SAME"  # (BSx16x16x16)
        )
        conv1 = tf.nn.bias_add(conv1, self.conv_bias1)
        #################

        #### LAYER 2 ####
        moments = tf.nn.moments(conv1, axes=[0, 1, 2])
        norm1 = tf.nn.batch_normalization(
            conv1, moments[0], moments[1], None, None, self.epsilon  # (BSx16x16x16)
        )
        #################

        #### LAYER 3 ####
        relu1 = tf.nn.relu(norm1)  # (BSx16x16x16)
        #################

        #### LAYER 3 ####
        max_pool1 = tf.nn.max_pool(
            relu1, ksize=[3, 3], strides=[1, 2, 2, 1], padding="SAME"  # (BSx8x8x16)
        )
        #################

        #### LAYER 4 ####
        conv2 = tf.nn.conv2d(
            max_pool1, self.filter2, [1, 1, 1, 1], padding="SAME"  # (BSx8x8x20)
        )
        conv2 = tf.nn.bias_add(conv2, self.conv_bias2)
        #################

        #### LAYER 5 ####
        moments = tf.nn.moments(conv2, axes=[0, 1, 2])
        norm2 = tf.nn.batch_normalization(
            conv2, moments[0], moments[1], None, None, self.epsilon  # (BSx8x8x20)
        )
        #################

        #### LAYER 6 ####
        relu2 = tf.nn.relu(norm2)  # (BSx8x8x20)
        #################

        #### LAYER 7 ####
        max_pool2 = tf.nn.max_pool(
            relu2, ksize=[2, 2], strides=[1, 2, 2, 1], padding="SAME"  # (BSx4x4x20)
        )
        #################

        #### LAYER 8 ####
        #### Test layer ####
        if is_testing:
            # My function
            conv3 = conv2d(max_pool2, self.filter3, [1, 1, 1, 1], padding="SAME")
        else:
            conv3 = tf.nn.conv2d(
                max_pool2, self.filter3, [1, 1, 1, 1], padding="SAME"  # (BSx4x4x20)
            )
        conv3 = tf.nn.bias_add(conv3, self.conv_bias3)
        #################

        #### LAYER 9 ####
        moments = tf.nn.moments(conv3, axes=[0, 1, 2])
        norm3 = tf.nn.batch_normalization(
            conv3, moments[0], moments[1], None, None, self.epsilon  # (BSx4x4x20)
        )
        #################

        #### LAYER 10 ####
        relu3 = tf.nn.relu(norm3)  # (BSx4x4x20)
        #################

        #### LAYER 11 ####
        ll_1 = tf.reshape(relu3, [-1, 320])  # Flatten to batchsize by 4*4*20
        ll_1 = tf.matmul(ll_1, self.W1) + self.b1  # (BS, 320) x (320, 64) = (BS x 64)
        ll_1 = tf.nn.dropout(ll_1, 0.3)
        ll_1 = tf.nn.relu(ll_1)
        #################

        #### LAYER 12 ####
        ll_2 = tf.matmul(ll_1, self.W2) + self.b2  # (BS x 64) x (64, 32) = (BS, 32)
        ll_2 = tf.nn.dropout(ll_2, 0.3)
        ll_2 = tf.nn.relu(ll_2)
        #################

        #### LAYER 13 ####
        ll_3 = (
            tf.matmul(ll_2, self.W3) + self.b3
        )  # (BS x 32) x (32, num_classes) = (BS, num_classes)
        #################

        return ll_3

    def loss(self, logits, labels):
        """
        Calculates the model cross-entropy loss after one forward pass.
        :param logits: during training, a matrix of shape (batch_size, self.num_classes) 
        containing the result of multiple convolution and feed forward layers
        Softmax is applied in this function.
        :param labels: during training, matrix of shape (batch_size, self.num_classes) containing the train labels
        :return: the loss of the model as a Tensor
        """
        return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

    def accuracy(self, logits, labels):
        """
        Calculates the model's prediction accuracy by comparing
        logits to correct labels – no need to modify this.
        :param logits: a matrix of size (num_inputs, self.num_classes); during training, this will be (batch_size, self.num_classes)
        containing the result of multiple convolution and feed forward layers
        :param labels: matrix of size (num_labels, self.num_classes) containing the answers, during training, this will be (batch_size, self.num_classes)

        NOTE: DO NOT EDIT
        
        :return: the accuracy of the model as a Tensor
        """
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


def train(model, train_inputs, train_labels):
    """
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: None
    """
    indices = tf.random.shuffle(tf.range(start=0, limit=len(train_labels)))
    train_inputs = tf.image.random_flip_left_right(
        (tf.cast(tf.gather(train_inputs, indices, axis=0), tf.float32))
    )
    train_labels = tf.cast(tf.gather(train_labels, indices, axis=0), tf.float32)

    batch_count = len(train_inputs) // model.batch_size

    for i in range(batch_count):
        with tf.GradientTape() as tape:
            batched_inputs = train_inputs[
                i * model.batch_size : (i + 1) * model.batch_size
            ]
            batched_labels = train_labels[
                i * model.batch_size : (i + 1) * model.batch_size
            ]

            batch_logits = model.call(batched_inputs)
            loss = model.loss(batch_logits, batched_labels)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this can be the average accuracy across 
    all batches or the sum as long as you eventually divide it by batch_size
    """
    batch_logits = model.call(test_inputs, True)
    accuracy = model.accuracy(batch_logits, test_labels)
    return accuracy


def visualize_results(
    image_inputs, probabilities, image_labels, first_label, second_label
):
    """
    Uses Matplotlib to visualize the results of our model.
    :param image_inputs: image data from get_data(), limited to 10 images, shape (10, 32, 32, 3)
    :param probabilities: the output of model.call(), shape (10, num_classes)
    :param image_labels: the labels from get_data(), shape (10, num_classes)
    :param first_label: the name of the first class, "dog"
    :param second_label: the name of the second class, "cat"

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    predicted_labels = np.argmax(probabilities, axis=1)
    num_images = image_inputs.shape[0]

    fig, axs = plt.subplots(ncols=num_images)
    fig.suptitle("PL = Predicted Label\nAL = Actual Label")
    for ind, ax in enumerate(axs):
        ax.imshow(image_inputs[ind], cmap="Greys")
        pl = first_label if predicted_labels[ind] == 0.0 else second_label
        al = first_label if np.argmax(image_labels[ind], axis=0) == 0 else second_label
        ax.set(title="PL: {}\nAL: {}".format(pl, al))
        plt.setp(ax.get_xticklabels(), visible=False)
        plt.setp(ax.get_yticklabels(), visible=False)
        ax.tick_params(axis="both", which="both", length=0)
    plt.show()


def main():

    player_matrix, wl_per_rosters = get_data(roster_file, matrix_file)

    model = Model()

    import time

    def print_red(s, p=False):
        return f"\033[91m {s}\033[00m" if not p else f"\033[91m {s}\033[00m"

    def print_green(s, p=False):
        return f"\033[92m {s}\033[00m" if not p else f"\033[92m {s}%\033[00m"

    start = time.time()
    for _ in range(model.epochs):
        train(model, train_images, train_labels)
    stop = time.time()

    train_time = stop - start

    print("------------------------------------------------")
    print(f"Testing with {len(test_images)} images...")

    ###########################################################
    start = time.time()
    acc = test(model, test_images, test_labels)
    stop = time.time()
    test_time = stop - start
    ###########################################################

    print("------------------------------------------------")

    ###########################################################
    acc = print_green(acc * 100, True) if acc > 0.7 else print_red(acc * 100, True)
    ###########################################################

    tf.print("Accuracy: ", acc, sep="")
    print(f"Training time: {str(train_time)[:6]} seconds")
    print(f"Testing time: {str(test_time)[:6]} seconds")

    # i = np.random.randint(0, high=len(test_images) - 10)
    # visualize_results(
    #     test_images[i : i + 10],
    #     model.call(test_images[i : i + 10]),
    #     test_labels[i : i + 10],
    #     picks[0],
    #     picks[1],
    # )


if __name__ == "__main__":
    main()
