#!/bin/bash python 
"""
@Jeff | 2/18/17

A Python implementation of Principle Component Analysis (PCA). 

I implemented Linear Discriminant Analysis (LDA) for my research, so I guess 
it should be time for some PCA!

Math tutorial before actually implementing: 
http://cs229.stanford.edu/section/gaussians.pdf

To understand understand one curcial proof for covariance matrix: See Appendix 1 in the above link 

Tutorial for PCA: 
http://sebastianraschka.com/Articles/2014_pca_step_by_step.html#taking-the-whole-dataset-ignoring-the-class-labels
"""
import numpy as np 
from matplotlib import pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch

###########################################################
#Generate 40 3-dimensional samples randomly drawn from a multivariate Gaussian distribution. 
#where one half (i.e., 20) samples of our data set are labeled ω1 and the other half ω2

np.random.seed(1234) #Random seed 

mean_vec1 = np.array([0,0,0])
covariance_mat1 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class1_sample = np.random.multivariate_normal(mean_vec1, covariance_mat1, 20).T
assert class1_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

mean_vec2 = np.array([0,0,0])
covariance_mat2 = np.array([[1,0,0],[0,1,0],[0,0,1]])
class2_sample = np.random.multivariate_normal(mean_vec2, covariance_mat2, 20).T
assert class2_sample.shape == (3,20), "The matrix has not the dimensions 3x20"

#Plotting. Ref: http://matplotlib.org/api/figure_api.html
fig1 = plt.figure(figsize=(8,8))
ax = fig1.add_subplot(111, projection='3d')
plt.rcParams['legend.fontsize'] = 10   
ax.plot(class1_sample[0,:], class1_sample[1,:], class1_sample[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
ax.plot(class2_sample[0,:], class2_sample[1,:], class2_sample[2,:], '^', markersize=8, alpha=0.5, color='red', label='class2')
plt.title('Samples for class 1 and class 2')
ax.legend(loc='upper right')
show_pig = input("Show data plotted in 3D? (yes/no)")
if show_pig == "yes": plt.show()
fig1.savefig("PCA_1.png")

###########################################################
#Taking the whole dataset ignoring the class labels
all_class = np.concatenate((class1_sample, class2_sample), axis=1)
assert all_class.shape == (3,40), "The matrix has not the dimensions 3x40"

#Computing the d-dimensional mean vector
mean_x = np.mean(all_class[0,:])
mean_y = np.mean(all_class[1,:])
mean_z = np.mean(all_class[2,:])

mean_vector = np.array([[mean_x],[mean_y],[mean_z]])
print('Mean Vector:\n', mean_vector)

#Compute scatter matrix 
scatter_matrix = np.zeros((3,3))
for i in range(all_class.shape[1]):
	scatter_matrix += (all_class[:,i].reshape(3,1) - mean_vector).dot((all_class[:,i].reshape(3,1) - mean_vector).T)
print("Scatter matrix:\n", scatter_matrix)

#Computing eigenvectors and corresponding eigenvalues
eig_val_sc, eig_vec_sc = np.linalg.eig(scatter_matrix)
for i in range(len(eig_val_sc)):
    eigvec = eig_vec_sc[:,i].reshape(1,3).T
    print('Eigenvector {}: \n{}'.format(i+1, eigvec))
    print('Eigenvalue {} from scatter matrix: {}'.format(i+1, eig_val_sc[i]))

###########################################################
#Visualizing Eigenvectors 
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

fig2 = plt.figure(figsize=(7,7))
ax = fig2.add_subplot(111, projection='3d')

ax.plot(all_class[0,:], all_class[1,:], all_class[2,:], 'o', markersize=8, color='green', alpha=0.2)
ax.plot([mean_x], [mean_y], [mean_z], 'o', markersize=10, color='red', alpha=0.5)
for v in eig_vec_sc.T:
    a = Arrow3D([mean_x, v[0]], [mean_y, v[1]], [mean_z, v[2]], mutation_scale=20, lw=3, arrowstyle="-|>", color="r")
    ax.add_artist(a)
ax.set_xlabel('x_values')
ax.set_ylabel('y_values')
ax.set_zlabel('z_values')
plt.title('Eigenvectors')
show_pig = input("Show eigenvectors plotted in 3D? (yes/no)")
if show_pig == "yes": plt.show()
fig2.savefig("PCA_2.png")

###########################################################
#Sorting the eigenvectors by decreasing eigenvalues and drop the lowest one 
for ev in eig_vec_sc:
    np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_sc[i]), eig_vec_sc[:,i]) for i in range(len(eig_val_sc))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
for i in eig_pairs:
    print(i[0])

#Choosing k, i.e. 2 in our case, eigenvectors with the largest eigenvalues
matrix_w = np.hstack((eig_pairs[0][1].reshape(3,1), eig_pairs[1][1].reshape(3,1)))
print('Matrix W:\n', matrix_w)

###########################################################
#Transforming the samples onto the new subspace
transformed = matrix_w.T.dot(all_class)
assert transformed.shape == (2,40), "The matrix is not 2x40 dimensional."

#Plotting 
fig3 = plt.figure()
plt.plot(transformed[0,0:20], transformed[1,0:20], 'o', markersize=7, color='blue', alpha=0.5, label='class1')
plt.plot(transformed[0,20:40], transformed[1,20:40], '^', markersize=7, color='red', alpha=0.5, label='class2')
plt.xlim([-4,4])
plt.ylim([-4,4])
plt.xlabel('x_values')
plt.ylabel('y_values')
plt.legend()
plt.title('Transformed samples with class labels')
user_input = input("Show the transformation in 3D? (yes/no)")
if user_input == "yes": plt.show()
fig3.savefig("PCA_3.png")





