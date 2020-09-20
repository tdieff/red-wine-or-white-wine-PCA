2D PCA using Keras. See 'analysis.pdf' for more detail.

Author: Taylor Dieffenbach

‘winequality-red.csv’ and ‘winequality-white.csv’ datasets include a number of measurable chemical features of any wine, such as fixed acidity, volatile acidity, citric acid, and residual sugar.

The datasets were loaded and merged into one pandas dataframe, log-transformed, and scaled.

1st and 2nd Principal Components were built from the scaled attributes, using a fully connected feedforward Neural Network autoencoder with two units in the hidden layer and a linear activation functions.

Observations of the two Principal Components are plotted in the plane. Red dots represent red wines, yellow dots represent white wine in the two distinct clusters shown in the PCA 2D projection.

