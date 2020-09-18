from copy import deepcopy
import numpy as np
import math
# import pylab as plt
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import kmeans_GMM as gmm
import pandas as pd 

np.set_printoptions(threshold=np.nan)

def load_LS_train_data():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/train_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/train_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/train_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    train_data_all_classes = [np_data_class1, np_data_class2, np_data_class3] #3 dimentional - 3 * training_class_size * 2
    
    return train_data_all_classes

def load_LS_test_data():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/test_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/test_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass1_Bayesian_Classifier/Latest/LS/test_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    test_data_all_classes = [np_data_class1, np_data_class2, np_data_class3] #3 dimentional - 3 * training_class_size * 2
    
    return test_data_all_classes
	
def train_batch_perceptron(two_classes):
    # plot_two_dim_pts(two_classes[1])
    # plot_two_dim_pts(two_classes[0])
    w = np.asarray([1, 1, 1]) #Initializing the wight vector
    eta = 0.1
    num_misclassified_pts = []
    iters = 0
    misclassified_pts = []
    for i in range(2):
    	for each_pt in two_classes[i]:
            each_pt = np.insert(each_pt, 0, 1)
            g_of_x = np.dot(w, each_pt)
            if i == 0:
                y = -1
            elif i == 1:
                y = 1
            if y * g_of_x < 0:
                misclassified_pts.append(y * each_pt)

    num_misclassified_pts.append(len(misclassified_pts))
    misclassified_pts = np.array(misclassified_pts)

    while len(misclassified_pts) != 0:
        iters += 1
        # if iters == 150:
        #     plot_three_dim_pts(misclassified_pts)
        #     break
        w = w + (eta * np.sum(misclassified_pts, axis = 0))
        misclassified_pts = []
        for i in range(2):
            for each_pt in two_classes[i]:
                each_pt = np.insert(each_pt, 0, 1)
                g_of_x = np.dot(w, each_pt)
                if i == 0:
                    y = -1
                else:
                    y = 1
                if y * g_of_x < 0:
                    misclassified_pts.append(y * each_pt)
        misclassified_pts = np.array(misclassified_pts)
        num_misclassified_pts.append(len(misclassified_pts))
    print(iters)
    print(w)
    # draw_line(w)
    return w

def test_which_class(x,  weight_vector_0_1, weight_vector_0_2, weight_vector_1_2):
    class_labels = [0, 0, 0]
    x = np.insert(x, 0, 1)
    g_of_x = np.dot(weight_vector_0_1, x)
    if g_of_x <= 0:
        class_labels[0] += 1
    else:
        class_labels[1] += 1

    g_of_x = np.dot(weight_vector_0_2, x)
    if g_of_x <= 0:
        class_labels[0] += 1
    else:
        class_labels[2] += 1

    g_of_x = np.dot(weight_vector_1_2, x)
    if g_of_x <= 0:
        class_labels[1] += 1
    else:
        class_labels[2] += 1

    return class_labels.index(max(class_labels))

def compute_confusion_matrix(data_all_classes_test, weight_vector_0_1, weight_vector_0_2, weight_vector_1_2):
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(3):
        for j in range(len(data_all_classes_test[i])):
            class_num = test_which_class(data_all_classes_test[i][j],  weight_vector_0_1, weight_vector_0_2, weight_vector_1_2)
            confusion_matrix[i][class_num] += 1

    return confusion_matrix

def plot_two_dim_pts(two_dim_pts):
    xs = [x[0] for x in two_dim_pts]
    ys = [x[1] for x in two_dim_pts]
    plt.scatter(xs, ys)
    # plt.xlabel('Iterations')
    # plt.ylabel('log likelihood')
    plt.savefig("asdf")
    # plt.clf()

def plot_three_dim_pts(three_dim_pts):
    xs = [x[1] for x in three_dim_pts]
    ys = [x[2] for x in three_dim_pts]
    plt.scatter(xs, ys)
    # plt.xlabel('Iterations')
    # plt.ylabel('log likelihood')
    plt.savefig("asdf")
    # plt.clf()

def draw_line():
    plt.plot(x, x + 0, linestyle='solid')


train_data_all_classes = load_LS_train_data()
data_all_classes_test = load_LS_test_data()
weight_vector_0_1 = train_batch_perceptron([train_data_all_classes[0], train_data_all_classes[1]])
weight_vector_0_2 = train_batch_perceptron([train_data_all_classes[0], train_data_all_classes[2]])
weight_vector_1_2 = train_batch_perceptron([train_data_all_classes[1], train_data_all_classes[2]])

confusion_matrix = compute_confusion_matrix(data_all_classes_test, weight_vector_0_1, weight_vector_0_2, weight_vector_1_2)
print("\nconfusion_matrix")
gmm.print_matrix(confusion_matrix)

performance_matrix = gmm.find_performance_matrix(confusion_matrix)
print("\nperformance_matrix")
gmm.print_matrix(performance_matrix)

class_accuracy = gmm.find_accuracy(confusion_matrix)
print("\nclass accuracy: ", class_accuracy)






