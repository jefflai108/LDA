#!/bin/bash/ python
"""
@Jeff | 2/18/17

A practice of Support Vector Machine with Scikit-learn 

http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#example-classification-plot-digits-classification-py

In short, a Support Vector Machine (SVM) is a discriminative classifier formally defined 
by a separating hyperplane. In other words, given labeled training data (supervised learning), 
the algorithm outputs an optimal hyperplane which categorizes new examples.

Some tutorials before implementing: 
1. What is support vector? 
https://www.analyticsvidhya.com/blog/2014/10/support-vector-machine-simplified/
2. What is SVM? 
https://www.analyticsvidhya.com/blog/2015/10/understaing-support-vector-machine-example-code/
3. Math behind SVM 
http://cs229.stanford.edu/notes/cs229-notes3.pdf
"""
import matplotlib.pyplot as plt 
from sklearn import datasets, svm, metrics 
from datetime import datetime 

def SVM_digit(): 
	startime = datetime.now()
	digits = datasets.load_digits()
	print(digits.data.shape)

	#Plotting 
	images_and_labels = list(zip(digits.images, digits.target))
	fig = plt.figure()
	for index, (image, label) in enumerate(images_and_labels[:4]):	
		plt.subplot(2, 4, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title("Training: %i" % label)
	n_samples = len(digits.images)
	data = digits.images.reshape((n_samples, -1))

	#Classifier 
	classifier = svm.SVC(gamma=0.001)

	#learning 
	classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])

	print("It takes %s to train the SVM." % str(datetime.now()-startime))
	
	#predict 
	expected = digits.target[n_samples / 2:]
	predicted = classifier.predict(data[n_samples / 2:])
	print("Classification report for classifier %s:\n%s\n" 
		% (classifier, metrics.classification_report(expected, predicted)))
	print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

	images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
	for index, (image, prediction) in enumerate(images_and_predictions[:4]):
		plt.subplot(2, 4, index+5)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title("Prediction: %i" % prediction)
	usr_input = input("Show training and prediction images? (yes or no)") 
	if (usr_input == 'yes'): plt.show()
	fig.savefig('SVM_digit.png')

if __name__ == "__main__":
	SVM_digit()

