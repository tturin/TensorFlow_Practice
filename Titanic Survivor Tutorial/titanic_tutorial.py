from __future__ import absolute_import, division, print_function, unicode_literals
import functools

import pandas as pd
import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

LABEL_COLUMN = 'survived'
LABELS = [0,1]

def main():
	TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
	TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"
	
	train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
	test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
	
	np.set_printoptions(precision=3, suppress=True)
	
	raw_train_data = get_dataset(train_file_path)
	raw_test_data = get_dataset(test_file_path)
	
	show_batch(raw_train_data)
	
	#If columns not provided, specify columns
	print("\nExplicit columns")
	CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town', 'alone']
	temp_dataset = get_dataset(train_file_path, column_names = CSV_COLUMNS)
	show_batch(temp_dataset)
	
	#Place data into vector
	print("\nVectorize data")
	SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
	DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
	temp_dataset = get_dataset(train_file_path, select_columns = SELECT_COLUMNS, column_defaults = DEFAULTS)
	show_batch(temp_dataset)
	
	#Pack columns together
	print("\nPack columns with PackNumericFeatures class")
	NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']
	packed_train_data = raw_train_data.map(PackNumericFeatures(NUMERIC_FEATURES))
	packed_test_data = raw_test_data.map(PackNumericFeatures(NUMERIC_FEATURES))
	show_batch(packed_train_data)

#General preprocessor class for selecting list of numeric features
#and then packs features into a single column
class PackNumericFeatures(object):
		def __init__(self, names):
			self.names = names
		
		def __call__(self, features, labels):
			numeric_freatures = [features.pop(name) for name in self.names]
			numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_freatures]
			numeric_features = tf.stack(numeric_features, axis = -1)
			features['numeric'] = numeric_features
			
			return features, labels

def get_dataset(file_path, **kwargs):
	dataset = tf.data.experimental.make_csv_dataset(
		file_path,
		batch_size = 5,
		label_name = LABEL_COLUMN,
		na_value = "?",
		num_epochs = 1,
		ignore_errors = True,
		**kwargs)
	return dataset

def show_batch(dataset):
	for batch, label in dataset.take(1):
		for key, value in batch.items():
			print("{:20s}: {}".format(key, value.numpy()))

if __name__ == "__main__":
	main()