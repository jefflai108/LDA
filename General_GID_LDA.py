#!/usr/bin/env python 
from kaldi_io import read_mat_scp
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
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
		CovB += np.dot(mean_class-global_mean, (mean_class-global_mean).T)
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
				temp_CovW += np.dot(z-mean_class,(z-mean_class).T)
		temp_CovW /= counter2 #average for each classes 
		CovW += temp_CovW 
	CovW /= counter #average over M classes 
	print "It takes", datetime.now() - startTime2, "to finish computing Within class covariance"
	print "It takes", datetime.now() - startTime, " to complete third step"
	print "Start training LDA"

	#train LDA 
	startTime = datetime.now() 
	eigen_values, eigen_vectors = scipy.linalg.eig(CovB, CovW) 
	print "It takes", datetime.now() - startTime, "to finish training LDA"
	return eigen_values, eigen_vectors #return tuples 

def Evaluation(): 
	"""
	Apply LDA() and find projection matrix and hence dimensionality reduction. 
	Using Evaluation data to find threshold for the classifier. 
	"""
	eigen_values, eigen_vectors = LDA()
	print "Start Evaluation"
	P = eigen_vectors[:,-1] #at most C-1 dimensions
	P = P[:,None] #make it a column vector

	############################################
	#Speaker Enrollment 
	print "Start Enrolling speaker data"

	#initiate list
	list_enroll_speaker = []
	list_enroll_index = []

	#Create a list of i-vectors 
	for key,mat in read_mat_scp("exp/ivectors_sre10_train_male/ivector.scp"):
		list_enroll_speaker.append(mat)
		list_enroll_index.append(key)
	for key,mat in read_mat_scp("exp/ivectors_sre10_train_female/ivector.scp"):
		list_enroll_speaker.append(mat)
		list_enroll_index.append(key)

	#Transform lists to matrices 
	arr_enroll_speaker = np.vstack(list_enroll_speaker)	
	y_enroll = np.dot(arr_enroll_speaker, P) 	
	
	#test 5
	assert y_enroll.shape[0] == arr_enroll_speaker.shape[0]
	assert y_enroll.shape[1] == P.shape[1]

    #Create dictionary
	speaker_enroll_Dic = {}
	with open("/export/b15/janto/kaldi/kaldi/egs/sre10/v1/data/sre10_train/utt2spk") as fopen:
		for line in fopen: 
			(key, id_val) = line.split()
			speaker_enroll_Dic[key] = id_val

	#Create a list_enroll_speaker_id that corresponds to the order of list_enroll_index 
	list_enroll_speaker_id = []
	for temp_index in list_enroll_index: #Double for-loop
		for key, id_val in speaker_enroll_Dic.iteritems(): 
			if key == temp_index: 
				list_enroll_speaker_id.append(id_val)
				break
	#test 6
	try: 
		print len(list_enroll_speaker_id) == arr_enroll_speaker.shape[0]
	except Exception:
		print "fail test 6"

	############################################	
	#Speaker Testing 
	print "Start inputing testing data"

	#initiate list
	list_test_speaker = []
	list_test_index = []

	#Create a list of i-vectors 
	for key,mat in read_mat_scp("exp/ivectors_sre10_test_male/ivector.scp"):
		list_test_speaker.append(mat)
		list_test_index.append(key)
	for key,mat in read_mat_scp("exp/ivectors_sre10_test_female/ivector.scp"):
		list_test_speaker.append(mat)
		list_test_index.append(key)

	#Transform lists to matrices 
	arr_test_speaker = np.vstack(list_test_speaker)	
	y_test = np.dot(arr_test_speaker, P) 
	
	#test 7
	print "test 7"
	print y_test.shape[0] == arr_test_speaker.shape[0]
	print y_test.shape[1] == P.shape[1]

	############################################	
	#Error rate 

	#inner product of the enrollment data and testing data 
	#S is a matrix with y_enroll.shape[0] by y_test.shape[0] dimensions
	S = np.dot(y_enroll, y_test.T)
	
    #Create target-(non)target matrix 
	Uniq_spk = []
	Uniq_ivector = []
	with open("/export/b15/janto/kaldi/kaldi/egs/sre10/v1/data/sre10_test/trials") as fopen:
		for line in fopen: 
			(spk, i_vector_id, binary) = line.split()
			if spk not in Uni_spk: 
				Uniq_spk.append(spk)
			if i_vector_id not in Uniq_ivector: 
				Uniq_ivector.append(i_vector_id)			

	objective = np.zeros((len(Uniq_spk), len(Uniq_ivector)))
	
############################################################################

	#test 8
	print "test 8"
	assert len(Uniq_spk) == len(list_enroll_speaker_id)
	assert len(Uniq_ivector) == len(list_test_speaker)
	assert objectvie.shape == S.shape 

	#Uniq_spk should have the same order as list_enroll_speaker_id
	#Uniq_ivector should have the same order as list_test_speaker

############################################################################

	#test 9
	print "test 9"
	print "The m dimension for the objective matrix is", objective.shape[0] 
	print "The n dimension for the objective matrix is", objective.shape[1] 
	
	temp_row = -1
	temp_column = -1
	with open("/export/b15/janto/kaldi/kaldi/egs/sre10/v1/data/sre10_test/trials") as fopen:
		for line in fopen:
			(spk, i_vector_id, binary) = line.split()			
			temp_row = Uniq_spk.index(spk)
			temp_column = Uniq_ivector.index(i_vector_id)
			if binary == "target": 
				objective[temp_row][temp_column] = 1
			elif binary == "nontarget":
				objective[temp_row][temp_column] = -1
	
	for row in range(objective.shape[0]): 
		for column in range(objective.shape[1]):
			if objective[row][column] != 1 and objective[row][column] != -1: 
				objective[row][column] = 0 
	
	#test 10
	print "test 10"
	for row in range(objective.shape[0]): 
		for column in range(objective.shape[1]):
			print objective[row][column]

	#Finding threshold 
	threshold = 0
	C = np.zeros((len(Uniq_spk), len(Uniq_ivector)))
	C = S > threshold 

	#Compare C with Objective to calculate error rate
	#Loop over both matrix

if __name__ == '__main__':
	Evaluation()
