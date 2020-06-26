from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import math
import pylab as plt
from os import listdir
from os.path import isfile, join
import kmeans_GMM_PCA as gmm

np.set_printoptions(threshold=np.nan)

def main():
	l = 2
	k = 8

	data_all_classes = load_bovw_train_set()
	data_all_classes_combined = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis=0)
	mean = np.mean(data_all_classes_combined, axis = 0)
	#// Build cov matrix with training data all classes
	cov_mat = np.cov(data_all_classes_combined.T)

	#// Do Eigen analysis on training data all classes
	eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
	for i in eigen_vectors:
		if (np.linalg.norm(i) - 1) == 0.001:
			print("alert!")
	
	index = eigen_values.argsort()[::-1]
	eigen_values_ordered = eigen_values[index]
	eigen_vectors_ordered = eigen_vectors[:,index]
	# plot_eigen_values(eigen_values_ordered, l)

	#// Pick the corresponding eigen vectors for each case and make reduced data of training and test data
	selected_eigen_vectors = np.asmatrix(eigen_vectors_ordered[:,:l])

	train_data_reduced = perform_data_reduction(data_all_classes, selected_eigen_vectors, l, mean)
	train_data_reduced_combined = np.concatenate((train_data_reduced[0], train_data_reduced[1], train_data_reduced[2]), axis=0)
	# plot_univariate_data(train_data_reduced, l)
	# plot_bivariate_data(train_data_reduced, l)
	
	mu_each_class_each_k = np.zeros((3, k, l))
	pi_each_class_each_k = np.zeros((3, k))
	sigma_each_class_each_k = np.zeros((3, k, l, l)) 

	for class_index in range(len(train_data_reduced)):
		print ("class", class_index)
	#// Do K - Means with reduced training data
		centers = find_initial_random_centers(train_data_reduced[class_index], k, l)
		centers_after_Kmeans, whcluster = gmm.computeKMC(train_data_reduced[class_index], centers, k)
		cluster_data = gmm.groupdata(train_data_reduced[class_index], centers_after_Kmeans, whcluster, k)
		
	#// Build GMM on training data
		log_likelihood_all_iters, pi_final_each_k, mu_final_each_k, sigma_final_each_k = gmm.compute_GMM(centers_after_Kmeans, train_data_reduced[class_index], cluster_data, k)
		plot_iters_vs_loglikelihood(log_likelihood_all_iters, 3, k, class_index, l)
		pi_each_class_each_k[class_index] = pi_final_each_k
		mu_each_class_each_k[class_index] = mu_final_each_k
		sigma_each_class_each_k[class_index] = sigma_final_each_k
		
	# Build Confusion matrix with test data
	data_all_classes_test = load_bovw_test_set()
	test_data_reduced = perform_data_reduction(data_all_classes_test, selected_eigen_vectors, l, mean)
	confusion_matrix = gmm.compute_confusion_matrix(test_data_reduced, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, 0)
	print("\nconfusion_matrix")
	gmm.print_matrix(confusion_matrix)

	performance_matrix = gmm.find_performance_matrix(confusion_matrix)
	print("\nperformance_matrix")
	gmm.print_matrix(performance_matrix)

	class_accuracy = gmm.find_accuracy(confusion_matrix)
	print("\nclass accuracy: ", class_accuracy)

def plot_iters_vs_loglikelihood(log_likelihood_all_iters, opt_type, k, i, l):
    plt.plot(log_likelihood_all_iters)
    if i == 0:
        title = "Class A"
    elif i == 1:
        title = "Class B"
    elif i == 2:
        title = "Class C"
    plt.title(str(title))
    plt.xlabel('Iterations')
    plt.ylabel('log likelihood')
    if opt_type == 1:
        opt_type = "NLS"
    elif opt_type == 2:
        opt_type = "RD"
    elif opt_type == 3:
        opt_type = "Scene_Images"
    elif opt_type == 4:
        opt_type = "Cell Scene_Images"
    plt.savefig(str(opt_type) + "_l_" + str(l) + "_K_" + str(k) + "_" + str(title) + ".png")
    plt.clf()

def plot_eigen_values(eigen_values, l):
	x = list(range(len(eigen_values)))
	plt.scatter(x, eigen_values, marker='o')
	plt.title("Eigenvalues in Descending Order")
	plt.ylabel('Eigenvalues')
	plt.savefig("Eigen_values_DO")
	plt.clf()

def plot_univariate_data(train_data_reduced, l):
	x = list(range(50))
	plt.scatter(x, train_data_reduced[0], marker='o')
	plt.title("Univariate data distribution")
	plt.ylabel('Points in One dimentional Space')
	plt.savefig("l" + str(l) + "_" + ".png")
	plt.clf()

def plot_bivariate_data(train_data_reduced, l):
	xs = [x[0] for x in train_data_reduced[0]]
	ys = [x[1] for x in train_data_reduced[0]]
	plt.scatter(xs, ys)
	plt.title("Bivariate data distribution")
	plt.xlabel('X value')
	plt.ylabel('Y value')
	plt.savefig("Bivariate_data_distribution")
	plt.clf()

def perform_data_reduction(data_all_classes, selected_eigen_vectors, l, mean):
	data_all_classes_reduced = []
	for class_index in range(len(data_all_classes)):
		# class_mean = np.mean(data_all_classes[class_index], axis = 0)
		data_each_class_reduced = np.zeros((50, l))

		for i in range(len(data_all_classes[class_index])):
			mean_sub_xn = np.asmatrix(data_all_classes[class_index][i] - mean)
			# data_each_class_reduced[i] = np.matmul(mean_sub_xn, selected_eigen_vectors)
			data_each_class_reduced[i] = mean_sub_xn.dot(selected_eigen_vectors)

		data_all_classes_reduced.append(data_each_class_reduced)

	# data_all_classes_reduced_combined = np.concatenate((data_all_classes_reduced[0], data_all_classes_reduced[1], data_all_classes_reduced[2]), axis=0) 
	# print(np.cov(data_all_classes_reduced_combined.T))
	return data_all_classes_reduced

def load_bovw_train_set():
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Train_BovW_Feature_Vectors/"
    train_folder = ["coast", "industrial_area", "pagoda"]
    data_all_classes = []
    for scene_type in train_folder:
        all_txt_files_path = join(dir_path, scene_type)
        data_each_class = []
        for each_img_file_name in listdir(all_txt_files_path):
            txt_path = join(all_txt_files_path, each_img_file_name)
            #cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)
            bovw_each_img = np.loadtxt(txt_path, delimiter=',')
            data_each_class.append(bovw_each_img)
        data_each_class = np.array(data_each_class)
        data_all_classes.append(data_each_class)
 
    return data_all_classes

def load_bovw_test_set():
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Test_BovW_Feature_Vectors/"
    train_folder = ["coast", "industrial_area", "pagoda"]
    data_all_classes = []
    for scene_type in train_folder:
        all_txt_files_path = join(dir_path, scene_type)
        data_each_class = []
        for each_img_file_name in listdir(all_txt_files_path):
            txt_path = join(all_txt_files_path, each_img_file_name)
            #cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31)
            bovw_each_img = np.loadtxt(txt_path, delimiter=',')
            data_each_class.append(bovw_each_img)
        data_each_class = np.array(data_each_class)
        data_all_classes.append(data_each_class)
    
    return data_all_classes

def find_initial_random_centers(data, k, d):  
	n = len(data) # Number of training data points
	centers = []
	random_center_indexes = np.random.choice(n, k, replace=False)
	for i in random_center_indexes:
		centers.append(data[i])
	centers = np.resize(centers, (k, d))

	return centers

def computeKMC(data, centers, k):   
    curr_centers = deepcopy(centers) 
    n = data.shape[0] #number of rows (data points)
    prev_centers = np.zeros(curr_centers.shape)  # to store old centers
    distances = np.zeros((n, k))      
    error = np.linalg.norm(curr_centers - prev_centers) #sum of distances from all (curr_centers, prev_centers) is expected to be zero.   
    itr = 0
    while error >= 0.001:
        # Euclidean distance between each data point and each centers. 
        # distance[] is a (num of data points * num of clusters) matrix.
        itr = itr + 1
        for i in range(k):
            distances[:,i] = np.linalg.norm(data - curr_centers[i], axis=1) 
        whcluster = np.argmin(distances, axis=1) #whcluster contains the information about which cluster, each data point belongs to.
        prev_centers = deepcopy(curr_centers)
        #MStep Finding the new cluster center with new points..
        for i in range(k):
            if(np.any(whcluster == i)):         # If there is no data associated with cluster, the center remain the same..
                curr_centers[i] = np.mean(data[whcluster == i], axis=0)
            else:
                print("no cluster associated in iteration number", itr)
                continue;
        error = np.linalg.norm(curr_centers - prev_centers)

    return curr_centers, whcluster

main()
