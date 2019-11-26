import os
import tensorflow as tf
import numpy as np
import random

class Model(tf.keras.Model):
	def __init__(self):
		super(Model, self).__init__()
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, name='adam')
    self.batch_size = 64
    self.num_classes = 2

		self.layer1 = tf.keras.layers.Dense(100, input_shape=(len(players),), activation='relu')
    self.layer2 = tf.keras.layers.Dense(num_classes, input_shape=(100,), activation='softmax')

	def call(self, inputs):
    layer_1_output = self.layer1(inputs)
    layer_2_output = self.layer2(layer_1_output)
    return layer_2_output

	def loss(self, logits, labels):
		return tf.nn.softmax_cross_entropy_with_logits(labels, logits)

def train(model, train_inputs, train_labels):
	num_iterations = ((int) (len(train_inputs) / model.batch_size))
	for i in range(num_iterations):
		with tf.GradientTape() as tape:
			logits = model.call(tf.image.random_flip_left_right(train_inputs[i*model.batch_size:(i+1)*model.batch_size]))
			labels = train_labels[i*model.batch_size:(i+1)*model.batch_size]
			labels = tf.compat.v2.squeeze(labels)
			loss = model.loss(logits, labels)
		gradients = tape.gradient(loss, model.trainable_variables)
		model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def test(model, test_inputs, test_labels):
	logits = model.call(test_inputs, True)
	return model.accuracy(logits, tf.compat.v2.squeeze(test_labels))
	
def main():
	data = win_loss_per_roster()
	model = Model()
	
	# Training
	train(data[0:3], data[3])

if __name__ == '__main__':
	main()
