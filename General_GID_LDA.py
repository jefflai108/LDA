#!/usr/bin/env python 
from kaldi_io import read_mat_scp
import numpy as np 
import collections 	
from datetime import datetime
import scipy.linalg 

def LDA():
	"""
	Generalization of the binary GID classifier. 
	First, create a numpy matrix with male and female speakers. 
	Secondly, create a list of speaker ids corresponding to the ivector matrix.
	Thirdly, compute in-class and between-class covaiances. 
	"""
	###########################################
	#First step, create a numpy matrix with male and female speakers.
	
	startTime = datetime.now()
	#initiate list
	list_speaker = []
	list_index = []

	#Create a list of i-vectors 
	for key,mat in read_mat_scp("exp/ivectors_sre_male/ivector.scp"):
		list_speaker.append(mat)
		list_index.append(key)
	for key,mat in read_mat_scp("exp/ivectors_sre_female/ivector.scp"):
		list_speaker.append(mat)
		list_index.append(key)

	#Transform lists to matrices 
	arr_speaker = np.vstack(list_speaker)	
	
	print "It takes", datetime.now() - startTime, " to complete first step"
	############################################
	#Second step, create a list of speaker ids corresponding to the ivector matrix.

	startTime = datetime.now()
	#create dictionary
	speaker_dic = {}
	with open("/export/b15/janto/kaldi/kaldi/egs/sre10/v1/data/sre/utt2spk") as fopen:
		for line in fopen: 
			(key, id_val) = line.split()
			speaker_dic[key] = id_val
	
	#test 1
	for key, id_val in speaker_dic.iteritems(): 
		print key, ",", id_val
	try: 
		print"length of list_index is", len(list_index)
		print "length of speaker_dic is", len(speaker_dic) 
		print len(list_index) == arr_speaker.shape[0]
	except Exception: 
		pass 
	
	#Create a list_dpeaker_id that corresponds to the order of list_index  
	counter=0; 
	list_speaker_id = []
	for temp_index in list_index: #Double for-loop
		for key, id_val in speaker_dic.iteritems(): 
			if key == temp_index: 
				list_speaker_id.append(id_val)
				break
		counter += 1

	#test 2
	try: 
		for test in list_speaker_id: 
			print test 
		print len(list_speaker_id) == arr_speaker.shape[0]
	except Exception: 
		pass 

	print "It takes", datetime.now() - startTime, " to complete second step"
	############################################
	#Third step, compute in-class and between class covariances.

	startTime = datetime.now()
	#Global mean (vector) of all classes (speakers)
	i_vector_dim = arr_speaker.shape[1] #i-vector dimension
	global_mean = np.zeros(i_vector_dim) 
	arr_speaker_mean = np.mean(arr_speaker, axis = 0)
	#test 3
	try: 
		print len(arr_speaker_mean) == i_vector_dim
	except Exception: 
		print "fail test 3"
	
	#Between-class covariance matrix 
	startTime1 = datetime.now()
	CovB = np.zeros((i_vector_dim, i_vector_dim))
	#Create a list of (list_speaker_id, arr_speaker) tuples 
	X = zip(list_speaker_id, arr_speaker)
	#Create a list with unique speaker-id
	uniq_list_speaker_id = [item for item, count in collections.Counter(list_speaker_id).items() if count > 1] 
	#Loop over every class (first loop)
	counter = 0 #No. of classes 
	mean_class = np.zeros(i_vector_dim) #temp vector 
	temp_list = []
	for i in uniq_list_speaker_id:
		counter += 1
		for (j, z) in X: #mean for each class 
			if j==i: temp_list.append(z)
		temp_mat = np.vstack(temp_list)
		mean_class = np.mean(temp_mat, axis = 0)
		#test 4
		try: 
			print len(mean_class) == i_vector_dim 
		except Exception: 
			print "fail test 4"
		CovB += np.outer(mean_class-global_mean, mean_class-global_mean)
	CovB /= counter #average over M classes 
	print "It takes", datetime.now() - startTime1, "to finish computing Between class covariance"

	#Within-class covariance matrix 
	startTime2 = datetime.now()
	CovW = np.zeros((i_vector_dim, i_vector_dim))
	temp_CovW = np.zeros((i_vector_dim, i_vector_dim))
	#Loop over every class (first loop)
	temp_vector = np.zeros(i_vector_dim) #temp vector 
	mean_class = np.zeros(i_vector_dim) #temp vector 
	counter = 0 #No. of classes 
	for i in uniq_list_speaker_id:
		counter += 1
		for (j, z) in X: #mean for each class 
			if j==i: temp_list.append(z)
		temp_mat = np.vstack(temp_list)
		mean_class = np.mean(temp_mat, axis = 0)
		counter2 = 0 #number of i-vectors of class i 
		for (j, z) in X:
			if j==i: 
				counter2 += 1
				temp_CovW += np.outer(z-mean_class,z-mean_class)
		temp_CovW /= counter2 #average for each classes 
		CovW += temp_CovW 
	CovW /= counter #average over M classes 
	print "It takes", datetime.now() - startTime2, "to finish computing Within class covariance"
	print "It takes", datetime.now() - startTime, " to complete third step"
	print "Start training LDA"

	#train LDA 
	startTime = datetime.now() 
	eigen_values, eigen_vectors = scipy.linalg.eig(CovB, CovW) 
	return eigen_values, eigen_vectors #return tuples 
	print "It takes", datetime.now() - startTime, "to finish training LDA"

def Evaluation(): 
	"""
	Apply LDA() and find projection matrix and hence dimensionality reduction. 
	Find threshold for the classifier. 
	"""
	eigen_values, eigen_vectors = LDA()
	P = eigen_vectors[:,-1] #at most C-1 dimensions
	
	#######################################################################
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
		print "Male error rate is", M_error_rate, "and Female error rate is", F_error_rate, "while threshold value is", threshold
		Cur_diff = abs(M_error_rate - F_error_rate) 	
		if Cur_diff < Prev_diff: 
			Prev_diff = Cur_diff
			Mini_threshold = threshold 
	print "The optimal threshold for the classifier is", Mini_threshold, "with Error rate difference", Prev_diff



if __name__ == '__main__':
	Evaluation()
