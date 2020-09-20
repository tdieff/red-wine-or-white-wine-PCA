import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow
from tensorflow import keras
from keras import models
from keras import layers
from keras import optimizers

def main():
	df, red_white_index = get_transformed_dataframe()
	x = df
	codings = build_model(x)
	generate_scatter(codings, red_white_index)
	return

def get_transformed_dataframe():
	df1 = pd.read_csv('winequality-red.csv', delimiter=';')
	df2 = pd.read_csv('winequality-white.csv', delimiter=';')
	print(df1)
	print(df2)
	df3 = pd.concat([df1, df2])
	df3 = log_transform(df3)
	df3 = df3.apply(scale, axis=0)
	return df3, len(df1)

def log_transform(df):
	df[['citric acid']] += 1
	df = df.transform(lambda x: np.log(x))
	return df

def scale(df):
	return (df - df.mean()) / df.std()

def build_model(x):
	encoder = models.Sequential([layers.Dense(2, input_shape=[12])])
	decoder = models.Sequential([layers.Dense(12, input_shape=[2])])
	autoencoder = models.Sequential([encoder, decoder])
	autoencoder.compile(loss='mse', optimizer=optimizers.SGD(lr=0.1))
	autoencoder.fit(x, x, epochs=20)
	return encoder.predict(x)

def generate_scatter(codings, i):
	plt.scatter(x=codings[:i, 0],
		y=codings[:i, 1],
		c='red',
		s=0.5)
	plt.scatter(x=codings[i:, 0],
		y=codings[i:, 1],
		c='gold',
		s=0.5)
	plt.title('PCA 2D Projection')
	plt.show()
	return

main()
