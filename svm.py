#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:49:59 2018

@author: ganesan
"""
import pandas as pd
import numpy as np
import math
from sklearn import svm, datasets
import matplotlib.pyplot as plt
import Loadclasses as loadcls
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.model_selection import GridSearchCV
import kmeans_GMM_PCA as gmm
import PCA_complete as PCA


#Linear Kernel
def kernel_fn(data_all_classes, data_all_classes_test, C, gamma):
    
    clf = svm.SVC(kernel='linear') # Linear Kernel
    data_train = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis = 0)
    data_test = np.concatenate((data_all_classes_test[0],data_all_classes_test[1], data_all_classes_test[2]), axis = 0)
    colorUse = ("red", "green", "blue")
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    cls0 = np.zeros(len(data_all_classes[0]))
    cls1 = np.zeros(len(data_all_classes[1]))
    cls2 = np.zeros(len(data_all_classes[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_train = np.concatenate((cls0, cls1, cls2), axis = 0)
    cls0 = np.zeros(len(data_all_classes_test[0]))
    cls1 = np.zeros(len(data_all_classes_test[1]))
    cls2 = np.zeros(len(data_all_classes_test[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_test = np.concatenate((cls0, cls1, cls2), axis = 0)
    x_min, x_max = data_train[:, 0].min() - 1, data_train[:, 0].max() + 1
    y_min, y_max = data_train[:, 1].min() - 1, data_train[:, 1].max() + 1
    h = abs((x_max / x_min)/100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    # Create the SVC model object
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(data_train, cls_train)
    Z = svc.predict(X_plot)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(data_train[:, 0], data_train[:, 1], c=cls_train, cmap=plt.cm.Set1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVM: Linear Kernel')
    y_pred = svc.predict(data_test)
    print(confusion_matrix(cls_test,y_pred))  
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  


    
    #Polynomial kernel
#    parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    svc = svm.SVC(kernel='rbf', C=C,gamma = gamma, decision_function_shape='ovr').fit(data_train, cls_train)
#    clf.fit(data_train, cls_train)
    Z = svc.predict(X_plot)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(data_train[:, 0], data_train[:, 1], c=cls_train, cmap=plt.cm.Set1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVM: RBF Kernel')
    y_pred = svc.predict(data_test)
    print(confusion_matrix(cls_test,y_pred))  
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    kernel_fn_another_two(data_all_classes, data_all_classes_test, C, gamma)
 

def bovw_predict(data_all_classes, data_all_classes_test, C, gamma):
    clf = svm.SVC(kernel='linear') # Linear Kernel
    data_train = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis = 0)
    data_test = np.concatenate((data_all_classes_test[0],data_all_classes_test[1], data_all_classes_test[2]), axis = 0)
    cls0 = np.zeros(len(data_all_classes[0]))
    cls1 = np.zeros(len(data_all_classes[1]))
    cls2 = np.zeros(len(data_all_classes[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_train = np.concatenate((cls0, cls1, cls2), axis = 0)
    cls0 = np.zeros(len(data_all_classes_test[0]))
    cls1 = np.zeros(len(data_all_classes_test[1]))
    cls2 = np.zeros(len(data_all_classes_test[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_test = np.concatenate((cls0, cls1, cls2), axis = 0)
    svc = svm.SVC(kernel='linear', C=C, decision_function_shape='ovr').fit(data_train, cls_train)
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(conf_mat)  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    
    svc = svm.SVC(kernel='rbf', C=C, gamma = gamma, decision_function_shape='ovr').fit(data_train, cls_train)
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(conf_mat)  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    
    
    svc = svm.SVC(kernel='poly', degree = 4, gamma = gamma, C=C, decision_function_shape='ovr').fit(data_train, cls_train)
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(conf_mat)  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))
    
    svc = svm.SVC(kernel='sigmoid',gamma = gamma, C=C, decision_function_shape='ovr').fit(data_train, cls_train)
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(conf_mat)  
    print(perf_mat)
    print(classification_report(cls_test,y_pred))
    


#Linear Kernel
def kernel_fn_another_two(data_all_classes, data_all_classes_test, C, gamma):
    clf = svm.SVC(kernel='linear') # Linear Kernel
    data_train = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis = 0)
    data_test = np.concatenate((data_all_classes_test[0],data_all_classes_test[1], data_all_classes_test[2]), axis = 0)
    colorUse = ("red", "green", "blue")
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    cls0 = np.zeros(len(data_all_classes[0]))
    cls1 = np.zeros(len(data_all_classes[1]))
    cls2 = np.zeros(len(data_all_classes[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_train = np.concatenate((cls0, cls1, cls2), axis = 0)
    cls0 = np.zeros(len(data_all_classes_test[0]))
    cls1 = np.zeros(len(data_all_classes_test[1]))
    cls2 = np.zeros(len(data_all_classes_test[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_test = np.concatenate((cls0, cls1, cls2), axis = 0)
    x_min, x_max = data_train[:, 0].min() - 1, data_train[:, 0].max() + 1
    y_min, y_max = data_train[:, 1].min() - 1, data_train[:, 1].max() + 1
    h = abs((x_max / x_min)/100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    # Create the SVC model object
    svc = svm.SVC(kernel='poly', degree = 4, gamma = gamma, C=C, decision_function_shape='ovr').fit(data_train, cls_train)
    Z = svc.predict(X_plot)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(data_train[:, 0], data_train[:, 1], c=cls_train, cmap=plt.cm.Set1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVM: Polynomial Kernel')
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(confusion_matrix(cls_test,y_pred))  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    
    #Polynomial kernel
#    parameters = [{'kernel': ['rbf'], 'gamma': [1e-4, 1e-3, 0.01, 0.1, 0.2, 0.5],'C': [1, 10, 100, 1000]}, {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
#    clf = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=5)
    svc = svm.SVC(kernel='sigmoid',gamma = gamma, C=C, decision_function_shape='ovr').fit(data_train, cls_train)
#    clf.fit(data_train, cls_train)
    Z = svc.predict(X_plot)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
    plt.scatter(data_train[:, 0], data_train[:, 1], c=cls_train, cmap=plt.cm.Set1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVM: sigmoid Kernel')
    y_pred = svc.predict(data_test)
    print(confusion_matrix(cls_test,y_pred))
    print(classification_report(cls_test,y_pred))  
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(confusion_matrix(cls_test,y_pred))  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    
def supportvectors(data_all_classes, data_all_classes_test, C, gamma):
    data_train = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis = 0)
    data_test = np.concatenate((data_all_classes_test[0],data_all_classes_test[1], data_all_classes_test[2]), axis = 0)
    cls0 = np.zeros(len(data_all_classes[0]))
    cls1 = np.zeros(len(data_all_classes[1]))
    cls2 = np.zeros(len(data_all_classes[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_train = np.concatenate((cls0, cls1, cls2), axis = 0)
    cls0 = np.zeros(len(data_all_classes_test[0]))
    cls1 = np.zeros(len(data_all_classes_test[1]))
    cls2 = np.zeros(len(data_all_classes_test[2]))
    cls0.fill(0)
    cls1.fill(1)
    cls2.fill(2)
    cls_test = np.concatenate((cls0, cls1, cls2), axis = 0)
    x_min, x_max = data_train[:, 0].min() - 1, data_train[:, 0].max() + 1
    y_min, y_max = data_train[:, 1].min() - 1, data_train[:, 1].max() + 1
    h = abs((x_max / x_min)/100)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    X_plot = np.c_[xx.ravel(), yy.ravel()]
    # Create the SVC model object
    C = 1.0 # SVM regularization parameter
    svc = svm.SVC(kernel='rbf', C=C, gamma = gamma, decision_function_shape='ovr').fit(data_train, cls_train)
    Z = svc.predict(X_plot)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
#    plt.contourf(xx, yy, Z, cmap=plt.cm.tab10, alpha=0.3)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
           linestyles=['--', '-', '--'])
    plt.scatter(svc.support_vectors_[:, 0], svc.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='b')
    plt.scatter(data_train[:, 0], data_train[:, 1], c=cls_train, cmap=plt.cm.Set1)
    plt.xlabel('X axis')
    plt.ylabel('Y axis')
    plt.xlim(xx.min(), xx.max())
    plt.title('SVM: RBF Kernel')
    y_pred = svc.predict(data_test)
    conf_mat = confusion_matrix(cls_test,y_pred)    #svm call 
    perf_mat = gmm.find_performance_matrix(conf_mat)
    print(confusion_matrix(cls_test,y_pred))  
    print( perf_mat)
    print(classification_report(cls_test,y_pred))  
    
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

	data_all_classes_reduced_combined = np.concatenate((data_all_classes_reduced[0], data_all_classes_reduced[1], data_all_classes_reduced[2]), axis=0) 
	print(np.cov(data_all_classes_reduced_combined.T))
	return data_all_classes_reduced

def main():
    data_all_classes = loadcls.load_LS_trainingset()
    data_all_classes_test = loadcls.load_LS_test_set()
    C = 1
    gamma = 0.05
    pca = 1
    l = 2
    data_all_classes_bovw = loadcls.load_bovw_train_set()
    data_all_classes_test_bovw = loadcls.load_bovw_test_set()
#    kernel_fn_another_two(data_all_classes, data_all_classes_test)
    kernel_fn(data_all_classes, data_all_classes_test, C, gamma)

#    supportvectors(data_all_classes, data_all_classes_test, C, gamma)
    data_all_classNLS = loadcls.load_NLS_trainingset()
    data_all_class_NLStest = loadcls.load_NLS_test_set()
    data_all_classNLS_some = [data_all_classNLS[0], data_all_classNLS[1]]
    data_all_class_NLStest_some = [data_all_class_NLStest[0], data_all_class_NLStest[1]]
#    bovw_predict(data_all_classes_bovw, data_all_classes_test_bovw, C, gamma)
    
#    if (pca == 1):
#        data_all_classes = data_all_classes_bovw
#        data_all_classes_combined = np.concatenate((data_all_classes[0], data_all_classes[1], data_all_classes[2]), axis=0)
#        mean = np.mean(data_all_classes_combined, axis = 0)
#        #// Build cov matrix with training data all classes
#        cov_mat = np.cov(data_all_classes_combined.T)
#    
#        #// Do Eigen analysis on training data all classes
#        eigen_values, eigen_vectors = np.linalg.eig(cov_mat)
#        for i in eigen_vectors:
#            if (np.linalg.norm(i) - 1) == 0.001:
#                print("alert!")
#        
#        index = eigen_values.argsort()[::-1]
#        eigen_values_ordered = eigen_values[index]
#        eigen_vectors_ordered = eigen_vectors[:,index]
#        # plot_eigen_values(eigen_values_ordered, l)
#    
#        #// Pick the corresponding eigen vectors for each case and make reduced data of training and test data
#        selected_eigen_vectors = np.asmatrix(eigen_vectors_ordered[:,:l])
#        selected_eigen_vectors1 = np.asmatrix(eigen_vectors_ordered[:,28:])
#        
#        train_data_reduced = perform_data_reduction(data_all_classes, selected_eigen_vectors, l, mean)
#        train_data_reduced_combined = np.concatenate((train_data_reduced[0], train_data_reduced[1], train_data_reduced[2]), axis=0)
#        
#        data_all_classes_test = data_all_classes_test_bovw
#        test_data_reduced = perform_data_reduction(data_all_classes_test, selected_eigen_vectors, l, mean)
#        kernel_fn(train_data_reduced, test_data_reduced, C, gamma)

    print ("here")

main()