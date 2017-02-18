#!/usr/bin/env python 
from kaldi_io import read_mat_scp
import numpy as np
import scipy.linalg  
np.seterr(divide='ignore', invalid='ignore')
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 

def LDA(): 
	"""
	First, read in i vectors from scp file and create numpy array for 
	male and female speaker respectively. 
	Secondly, calculate within-class and between-class covariances. 
	Lastly, train and return eigenvales and eigenvectors. 

	"""
	#initiate lists 
	list_male = []
	list_female = []

	#Create a list of i-vectors 
	for key,mat in read_mat_scp("exp/ivectors_sre_male/ivector.scp"):
		list_male.append(mat)
	for key,mat in read_mat_scp("exp/ivectors_sre_female/ivector.scp"):
		list_female.append(mat)

	#Transform lists to matrices 
	arr_male = np.vstack(list_male)	
	arr_female = np.vstack(list_female)
 
	#Calculate in-class and between class covariances 
	#First, calculate mean of classes and global mean of the data
	i_vector_dim = arr_male.shape[1] #i-vector dimension
	N_female_ivector = arr_female.shape[0] #No.female ivectors
	N_male_ivector = arr_male.shape[0] #No.male ivectors 
	arr_male_mean = np.mean(arr_male, axis = 0)
	arr_female_mean = np.mean(arr_female, axis = 0) 
	global_mean = np.zeros(i_vector_dim) #global mean 
	for i in range(i_vector_dim):
		global_mean[i] = (arr_male_mean[i]*N_male_ivector+ 
			arr_female_mean[i]*N_female_ivector)/(N_female_ivector+N_male_ivector)

	#Implement Between-class covariance 
	CovB = np.zeros((i_vector_dim, i_vector_dim)) #initiate covariance matrix 
	CovB += np.dot(arr_male_mean-global_mean, (arr_male_mean-global_mean).T)
	CovB += np.dot(arr_female_mean-global_mean, (arr_female_mean-global_mean).T)
	CovB /= 2

	#Implement Within-class covariances 
	CovW1 = np.zeros((i_vector_dim, i_vector_dim))
	CovW2 = np.zeros((i_vector_dim, i_vector_dim)) #initiate covariance matrices
	for i in range(N_male_ivector): 
		CovW1 += np.dot(arr_male[i]-arr_male_mean, (arr_male[i]-arr_male_mean).T)
	CovW1 /= N_male_ivector
	for i in range(N_female_ivector): 
		CovW2 += np.dot(arr_female[i]-arr_female_mean, (arr_female[i]-arr_female_mean).T)
	CovW2 /= N_female_ivector
	CovW = (CovW1+CovW2)/2

	#train LDA 
	eigen_values, eigen_vectors = scipy.linalg.eig(CovB, CovW) 
	return eigen_values, eigen_vectors #return tuples 

def Evaluation(): 
	#Finding threshold value 
	eigen_values, eigen_vectors = LDA()
	P = eigen_vectors[:,-1]
	P = P[:,None] 

	#load data 
	list_M_speaker = []
	list_F_speaker = []
	for key,mat in read_mat_scp("exp/ivectors_sre10_test_male/ivector.scp"):
		list_M_speaker.append(mat)
	for key,mat in read_mat_scp("exp/ivectors_sre10_test_female/ivector.scp"):
		list_F_speaker.append(mat)
	arr_M_speaker = np.vstack(list_M_speaker)	
	arr_F_speaker = np.vstack(list_F_speaker)	
	y1 = np.dot(arr_M_speaker, P) 
	y2 = np.dot(arr_F_speaker, P) 

	Prev_diff = 100
	Mini_threshold = -2
	for threshold in np.arange(-1, 1, 0.001):
		M_error_rate = np.mean(y1 > threshold) 
		F_error_rate = np.mean(y2 < threshold) 
		#print "Male error rate is", M_error_rate, "and Female error rate is", F_error_rate, "while threshold value is", threshold
		Cur_diff = abs(M_error_rate - F_error_rate) 	
		if Cur_diff < Prev_diff: 
			Prev_diff = Cur_diff
			Mini_threshold = threshold 
	print "The optimal threshold for the classifier is", Mini_threshold, "with Error rate difference", Prev_diff

def plot(): 
	#Visualize the data 
        eigen_values, eigen_vectors = LDA()
        P = eigen_vectors[:,-1]
        P = P[:,None]

        #load data
        list_M_speaker = []
        list_F_speaker = []
        for key,mat in read_mat_scp("exp/ivectors_sre10_test_male/ivector.scp"):
                list_M_speaker.append(mat)
        for key,mat in read_mat_scp("exp/ivectors_sre10_test_female/ivector.scp"):
                list_F_speaker.append(mat)
        arr_M_speaker = np.vstack(list_M_speaker)
        arr_F_speaker = np.vstack(list_F_speaker)
        y1 = np.dot(arr_M_speaker, P)
        y2 = np.dot(arr_F_speaker, P)

	fig = plt.figure()
	bins = np.linspace(-2, 2, 250)
        plt.hist(y1, bins, alpha=0.65, label='male')
        plt.hist(y2, bins, alpha=0.65, label='female')
        plt.legend(loc='upper right')
        plt.show()
	fig.savefig('temp.png')

if __name__ == '__main__':
	Evaluation()  
