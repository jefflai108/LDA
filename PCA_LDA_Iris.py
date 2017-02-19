#!/bin/bash python 
"""
@Jeff | 2/18/17
A practice for comparison between LDA and PCA with 2D projection of Iris dataset using scikit-learn. 

http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_vs_lda.html#sphx-glr-auto-examples-decomposition-plot-pca-vs-lda-py

Principal Component Analysis (PCA) applied to this data identifies the combination 
of attributes (principal components, or directions in the feature space) that account 
for the most variance in the data. Here we plot the different samples on the 2 first 
principal components.

Linear Discriminant Analysis (LDA) tries to identify attributes that account for the 
most variance between classes. In particular, LDA, in contrast to PCA, is a supervised 
method, using known class labels.

PCA from scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
LDA from scikit-learn: http://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html#sklearn.discriminant_analysis.LinearDiscriminantAnalysis
"""
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def PCA_LDA_Iris():
	startTime = datetime.now() 

	#The iris dataset is a classic and very easy multi-class classification dataset.
	iris = datasets.load_iris()
	X = iris.data
	y = iris.target
	target_names = iris.target_names

	#PCA with number of components to keep 
	pca = PCA(n_components=2)
	X_pca = pca.fit(X).transform(X)

	lda = LinearDiscriminantAnalysis(n_components=2)
	X_lda = lda.fit(X, y).transform(X)

	# Percentage of variance explained for each components
	print("explained variance ratio (first two components): %s" % str(pca.explained_variance_ratio_))
	print("It takes %s to finish training" % str(datetime.now() - startTime))

	#plotting PCA
	fig1 = plt.figure()
	colors = ['navy', 'turquoise', 'darkorange']
	lw = 2
	for color, i, target_name in zip(colors, [0,1,2], target_names):
		plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title("PCA of IRIS dataset")
	
	#plotting LDA 
	fig2 = plt.figure()
	for color, i, target_name in zip(colors, [0,1,2], target_names):
		plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1], color=color, alpha=.8, lw=lw, label=target_name)
	plt.legend(loc='best', shadow=False, scatterpoints=1)
	plt.title("LDA of IRIS dataset")

	usr_inp = input("Show LDA and PCA plot? (yes or no)")
	if (usr_inp == "yes"): plt.show()	
	fig1.savefig("PCA_Iris.png")
	fig2.savefig("LDA_Iris.png")

if __name__ == '__main__':
	PCA_LDA_Iris()
