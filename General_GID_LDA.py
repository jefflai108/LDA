#!/usr/bin/env python 
from kaldi_io import read_mat_scp
import numpy as np 
np.seterr(divide='ignore', invalid='ignore')
import collections 	
from datetime import datetime
import scipy.linalg 
import pickle  
import time 

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
	
	#Store the eigen_values, eigen_vectors in a seperate file
	file_Name = "Eigens"
	f = open(file_Name, "wb")
	pickle.dump(eigen_values, f)
	pickle.dump(eigen_vectors, f)
	f.close()
	
	return eigen_values, eigen_vectors #return tuples 

def Evaluation(): 
	"""
	Apply LDA() and find projection matrix and hence dimensionality reduction. 
	Using Evaluation data to find threshold for the classifier. 
	"""
	startTime = datetime.now()
	
	#import eigen_values and eigen_vectors 
	file_Name = "Eigens"
	f = open(file_Name, "rb")
	eigen_values = pickle.load(f)
	eigen_vectors = pickle.load(f)
	f.close()
	#eigen_values, eigen_vectors = LDA()
	print "Start Evaluation"
	P = eigen_vectors[:,-1] #at most C-1 dimensions
	P = P[:,None] #make it a column vector
	
	############################################
	
    #Create target-(non)target matrix 
	Uniq_spk = []
	Uniq_ivector = []
	with open("/export/b15/janto/kaldi/kaldi/egs/sre10/v1/data/sre10_test/trials") as fopen:
		for line in fopen: 
			(spk, i_vector_id, binary) = line.split()
			if spk not in Uniq_spk: 
				Uniq_spk.append(spk)
			if i_vector_id not in Uniq_ivector: 
				Uniq_ivector.append(i_vector_id)

	print "The original length of Uniq_spk is", len(Uniq_spk) #4267

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

	############################################

	print "The original length of list_enroll_speaker_id is", len(list_enroll_speaker_id) #11936

	new_Uniq_spk = []
	new_list_enroll_speaker = []
	for spk_id in list_enroll_speaker_id: 
		if spk_id in Uniq_spk: 
			index = list_enroll_speaker_id.index(spk_id)
			new_Uniq_spk.append(spk_id)
			new_list_enroll_speaker.append(list_enroll_speaker[index])

	#test 5
	print "test 5"
	print "length of the new_Uniq_spk is", len(new_Uniq_spk); 
	print "length of the new list_enroll_speaker is", len(new_list_enroll_speaker)
	assert len(new_Uniq_spk) == len(new_list_enroll_speaker), "fail test 5"
	
	#Transform lists to matrices 
	arr_enroll_speaker = np.vstack(new_list_enroll_speaker)	
	y_enroll = np.dot(arr_enroll_speaker, P) 	
	
	#test 6
	print "test 6"
	print len(new_Uniq_spk) == arr_enroll_speaker.shape[0]
	assert y_enroll.shape[0] == arr_enroll_speaker.shape[0], "fail test 6"
	assert y_enroll.shape[1] == P.shape[1], "fail test 6"

#################################################	
	#Speaker Testing 
	print "Start inputing testing data"
	print "The original length of Uniq_ivector is", len(Uniq_ivector) #767

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

#################################################

	new_Uniq_ivector = []
	for dummy in list_test_index: 
		if dummy in Uniq_ivector:
			index = list_test_index.index(dummy)
			new_Uniq_ivector.append(list_test_speaker[index])

	#Transform lists to matrices 
	arr_test_speaker = np.vstack(new_Uniq_ivector)	
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
	objective = np.zeros((len(new_Uniq_spk), len(new_Uniq_ivector)))			
	
	#test 8
	print "test 8"
	assert objective.shape == S.shape, "fail test 8"

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
			temp_row = new_Uniq_spk.index(spk)
			temp_column = new_Uniq_ivector.index(i_vector_id)
			if binary == "target": 
				objective[temp_row][temp_column] = 1
			elif binary == "nontarget":
				objective[temp_row][temp_column] = -1
	
	for row in xrange(objective.shape[0]): 
		for column in xrange(objective.shape[1]):
			if objective[row][column] != 1 and objective[row][column] != -1: 
				objective[row][column] = 0 
	
	#test 10
	print "test 10"
	for row in objective: 
		print row 

###########################################################################
	#Finding threshold 
	(mini_false_rej, mini_false_accep) = (-100, -100) 
	prev_diff = 100
	(false_rej, false_accep) = (0, 0)
	for threshold in np.arrangenp.arange(-2, 2, 0.01):
		C = np.zeros((len(new_Uniq_spk), len(new_Uniq_ivector)))
		C = S > threshold 

	#Compare C with Objective to calculate error rate
	#False rejection, also called a type I error, is a mistake occasionally made by 
	#biometric security systems. In an instance of false rejection, the system fails to 
	#recognize an authorized person and rejects that person as an impostor.
		N_target = 0 
		N_match = 0 
		for row in xrange(objective.shape[0]): 
			for column in xrange(objective.shape[1]):
				if objective[row][column] == 1: 
					N_target += 1
					if C[row][column] == 0:
						N_match += 1

		false_rej = N_match/N_target*100
	#False Acceptance, also called a type II error. A system's FAR typically is stated 
	#as the ratio of the number of false acceptances divided by the number of 
	#identification attempts.
		N_nontarget = 0
		N_match = 0
		for row in xrange(objective.shape[0]): 
			for column in xrange(objective.shape[1]):
				if objective[row][column] == -1: 
					N_target += 1		
					if C[row][match] == 1:
						N_match += 1 
		false_accep = N_match/N_nontarget*100
		
		diff = abs(false_reg-false_accep)
		if diff < prev_diff: 
			prev_diff = diff 
			mini_false_rej = false_rej
			mini_false_accep = false_accep

	#Ends for loop 
	print "It takes", datetime.now-startTime, "to finish evaluation and find the optimal threshold." 
	print "The optimal threshold for the M-class classifier is", threshold, "with false rejection", mini_false_rej, "and false acceptance", mini_false_accep

if __name__ == '__main__':
	Evaluation()
