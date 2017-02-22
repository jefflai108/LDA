#!/bin/bash python 
"""
Testing pickly functionality for later use in General_GID_LDA.py

Tutorial: 
https://pythontips.com/2013/08/02/what-is-pickle-in-python/
http://stackoverflow.com/questions/15463387/pickle-putting-more-than-1-object-in-a-file
"""
import pickle 
import numpy as np 

a = np.array(['test value 1','test value 2','test value 3'])
b = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
file_Name = "testfile_pickle"

# open the file for writing 
f = open(file_Name, "wb")
pickle.dump(a, f)
pickle.dump(b, f)
f.close()

# we open the file for reading
f = open(file_Name, "rb")
c = pickle.load(f)
d = pickle.load(f)
f.close()

assert (c==a).all(), "failed test 1"
print(b)
print()
print(d) 
