#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 21:05:37 2018

@author: ganesan
"""

from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import kmeans_GMM_PCA as gmm
import Loadclasses as loadcls


def find_Sb_and_Sw(data_two_classes, N_cls):
    mean_two_classes, scatter_two_classes = [], []
    if(N_cls != 2): print("Number of classes should be 2")
    for i in range(N_cls):
        mean = np.mean(data_two_classes[i], axis = 0)
        cov_mat = np.cov(data_two_classes[i].T)
        N = len(data_two_classes[i])
        scatter = (N -1) * cov_mat      #N-1 is multiplied because np.cov uses N-1 only for finding covariance..
        mean_two_classes.append(mean)
        scatter_two_classes.append(scatter)
    within_cls_scatterM = scatter_two_classes[0] + scatter_two_classes[1]   #Sw
#    mu1_minus_mu2 = np.zeros((2,2))[np.]
    mu1_minus_mu2_noaxis = np.subtract(mean_two_classes[0], mean_two_classes[1])
    mu1_minus_mu2 = np.subtract(mean_two_classes[0], mean_two_classes[1]) [np.newaxis]
    mu1_minus_mu2_transp = mu1_minus_mu2.T
    btw_cls_scatterM = np.multiply(mu1_minus_mu2_transp, mu1_minus_mu2)
    print ("here")
    return within_cls_scatterM, btw_cls_scatterM, mu1_minus_mu2

def find_project_vector(within_cls_scatterM, btw_cls_scatterM, mu1_minus_mu2):
    mu1_minus_mu2_transp = mu1_minus_mu2.T
    inv_within_cls_scatterM = np.linalg.inv(within_cls_scatterM)
    w = inv_within_cls_scatterM.dot(mu1_minus_mu2_transp)           #procedure 1.. 
    
    i_sw_sb = inv_within_cls_scatterM.dot(btw_cls_scatterM)         #procedure 2.. we can use either eigen_vector_max or w...
    eigen_values, eigen_vectors = np.linalg.eig(i_sw_sb)
    index = eigen_values.argsort()[::-1]
    eigen_values_ordered = eigen_values[index]
    eigen_vectors_ordered = eigen_vectors[:,index]
    eigen_values_ordered_tr = eigen_vectors_ordered.T
    eigen_vector_max = eigen_values_ordered_tr[0]
    return w, eigen_vector_max

def find_projected_points(data_some_classes, w, eigen_vector_max):
    proj_vect = np.array(eigen_vector_max)[np.newaxis]
    w_t = w.T
    cls1 = data_some_classes[0]
    cls2 = data_some_classes[1]
    proj_cls1, proj_cls2 = [], []
    proj_cls1_np = np.zeros((len(cls1),w.shape[1]))
    proj_cls2_np = np.zeros((len(cls2),w.shape[1]))

    for i in range(len(cls1)):
        xn1 = np.asmatrix(cls1[i])
        xn2 = np.asmatrix(cls2[i])
        proj_cls1_np[i] =  xn1.dot(w)
        proj_cls2_np[i] =  xn2.dot(w)
        proj_c1 = w_t.dot(cls1[i])
        proj_c2 = w_t.dot(cls2[i])
#        proj_cls = np.concatenate(proj_vect)
        proj_cls1.append(proj_c1)
        proj_cls2.append(proj_c2)
        
    proj_cls_1 = np.array(proj_cls1)
    proj_cls_2 = np.array(proj_cls2)
#    proj_cls1_cls2 = [proj_cls_1, proj_cls_2]
    proj_cls1_cls2 = [proj_cls1_np, proj_cls2_np]
#    cls1_project = np.multiply(proj_vect, cls1)
    return proj_cls1_cls2

def find_projected_pt(x, w):
#    proj_vect = np.array(w)[np.newaxis]
#    proj_x = proj_vect.dot(x)
    xn = np.asmatrix(x)
    proj_x = xn.dot(w)
    
    return proj_x



def find_initial_random_centers(data, k, d):  
    n = len(data) # Number of training data points
    centers = []
    random_center_indexes = np.random.choice(n, k, replace=False)
    for i in random_center_indexes:
        centers.append(data[i])
    centers = np.resize(centers, (k, d))
    return centers

#def compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k):
#    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
#    for i in range(3):
#        for j in range(len(data_all_classes_test[i])):
#            class_num = test_which_class(data_all_classes_test[i][j], pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k)
#            confusion_matrix[i][class_num] += 1
#
#    return confusion_matrix

def test_which_class(x, pi_each_class_each_k_combnn, mu_each_class_each_k_combnn, sigma_each_class_each_k_combnn, k, two_cls):
    prob_xi_each_class = []
    for i in range(two_cls):
        summation = 0
        for j in range(k):
            nix = gmm.norm_pdf_multivariate(x, mu_each_class_each_k_combnn[i][j], sigma_each_class_each_k_combnn[i][j])
            summation += pi_each_class_each_k_combnn[i][j] * nix
        prob_xi_each_class.append(summation)
    
    return prob_xi_each_class.index(max(prob_xi_each_class))
#def similar_to_confusion_matrix(data_all_classes_test, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k):
#    count_cls = len(data_all_classes_test)
#    tot_test_pts =  len(data_all_classes_test[0])
#    wh_class_arr = np.zeros((tot_test_pts,count_cls))
#    for i in range(count_cls):
#        for j in range(tot_test_pts):
#            class_num = test_which_class(data_all_classes_test[i][j], pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, count_cls)
#            wh_class_arr[j][i] = class_num
#    return wh_class_arr

def compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k_diff_combination, mu_each_class_each_k_diff_combination, sigma_each_class_each_k_diff_combination, W_diff_combinn, k):
    print ("here")
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    count_cls = len(data_all_classes_test)
#    two_clas =  len(cls01_cls12_cls02_test[0])
    count_two_clas = len(pi_each_class_each_k_diff_combination[0])
    tot_test_pts = len(data_all_classes_test[0])
    for i in range(count_cls):
        test_cls_xx = data_all_classes_test[i]
        for j in range(len(test_cls_xx)):
            x = test_cls_xx[j]
            clsno = []
            for l in range(count_cls):
                proj_x = find_projected_pt(x, W_diff_combinn[l])
                class_num = test_which_class(proj_x, pi_each_class_each_k_diff_combination[l], mu_each_class_each_k_diff_combination[l], sigma_each_class_each_k_diff_combination[l], k, count_two_clas)            
                
                if(l == 0):
                    cls_index = [0,1]
                    clsno.append(cls_index[class_num])
                if(l == 1):
                    cls_index = [1,2]
                    clsno.append(cls_index[class_num])
                if(l == 2):
                    cls_index = [0,2]
                    clsno.append(cls_index[class_num])
                
            all_whc = np.array(clsno)
            counts = np.bincount(all_whc)
            wh_cls = np.argmax(counts)
            confusion_matrix[i][wh_cls] +=1
    return confusion_matrix

#            proj_test_some_cls = find_projected_points(data_some_classes_test, w, eigen_vector_max)
#            wh_class_arr = similar_to_confusion_matrix(proj_test_some_cls, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k)
#            print ("here")
    
def plot_decision_boundary(data_all_classes, pi_each_class_each_k_diff_combination, mu_each_class_each_k_diff_combination, sigma_each_class_each_k_diff_combination, W_diff_combinn, k):
    count_cls = len(data_all_classes)
#    two_clas =  len(cls01_cls12_cls02_test[0])
    class_names = ["A", "B", "C"]
    count_two_clas = len(pi_each_class_each_k_diff_combination[0])
    colorUse = ("red", "green", "blue")
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    groupName = ("Class1", "Class2", "Class3")
    markSym = ('o', '^', 's')
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
    plt.figure()
    cls_xmin,cls_xmax,cls_ymin,cls_ymax = [],[],[],[]
    for clsno in range(count_cls):
        xval = data_all_classes[clsno][:,0]
        yval = data_all_classes[clsno][:,1]
        xmax = np.max(xval)
        ymax = np.max(yval)
        xmin = np.min(xval)
        ymin = np.min(yval)
        cls_xmin.append(xmin)
        cls_ymin.append(ymin)
        cls_xmax.append(xmax)
        cls_ymax.append(ymax)
    xmin1 = min(cls_xmin) - ((min(cls_xmax) - min(cls_xmin)) * 0.25)
    ymin1 = min(cls_ymin) - ((min(cls_ymax) - min(cls_ymin)) * 0.25)
    xmax1 = max(cls_xmax) + ((min(cls_xmax) - min(cls_xmin)) * 0.25)
    ymax1 = max(cls_ymax) + ((min(cls_ymax) - min(cls_ymin)) * 0.25)
    print (xmin1, ymin1, xmax1, ymax1)
    x0,y0,x1,y1,x2,y2 = [], [],[],[],[],[]    
    for a in np.arange(xmin1, xmax1, (xmax1-xmin1)/200.0):
        for b in np.arange(ymin1, ymax1, (ymax1-ymin1)/200.0):
            xy = [a,b]
            pt_in_space = np.array(xy)
            clsno = []
            for l in range(count_cls):
                proj_x = find_projected_pt(pt_in_space, W_diff_combinn[l])
                class_num = test_which_class(proj_x, pi_each_class_each_k_diff_combination[l], mu_each_class_each_k_diff_combination[l], sigma_each_class_each_k_diff_combination[l], k, count_two_clas)            
                
                if(l == 0):
                    cls_index = [0,1]
                    clsno.append(cls_index[class_num])
                if(l == 1):
                    cls_index = [1,2]
                    clsno.append(cls_index[class_num])
                if(l == 2):
                    cls_index = [0,2]
                    clsno.append(cls_index[class_num]) 
            all_whc = np.array(clsno)
            counts = np.bincount(all_whc)
            whclass = np.argmax(counts)
            if(whclass==0):
               x0.append(a)
               y0.append(b) 
            elif(whclass==1):
               x1.append(a)
               y1.append(b) 
            else: 
               x2.append(a)
               y2.append(b)
    plt.scatter(x0,y0,alpha=0.4,marker='s', edgecolors=catColor[0], facecolor=catColor[0], s=50)
    plt.scatter(x1,y1,alpha=0.4,marker='s', edgecolors=catColor[1], facecolor=catColor[1], s=50)
    plt.scatter(x2,y2,alpha=0.4,marker='s', edgecolors=catColor[2], facecolor=catColor[2], s=50)
    for clsno in range(count_cls):
        xval = data_all_classes[clsno][:,0]
        yval = data_all_classes[clsno][:,1]
        plt.scatter(xval,yval,alpha=1.0,marker=markSym[clsno],edgecolors=colorUse[clsno],facecolor=colorUse[clsno],s=5,label=class_names[clsno])
    plt.legend()
    plt.xlabel('Xaxis')
    plt.ylabel('Yaxis')
#    plt.axes().set_aspect('equal', 'box')
    plt.savefig("1.png", dpi = 500)
#    plt.savefig("Type"+str(class_names[0])+str(class_names[1])+str(class_names[2])+".png")    
    
    
def abline(slope, intercept):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    print ("X-Y",x_vals, y_vals)
    plt.plot(x_vals, y_vals, '--')
    return x_vals, y_vals
    
def plot_classes(proj_cls_1, proj_cls_2):
#    print (np.zeros_like(proj_cls_1))
    plt.plot(proj_cls_1, np.zeros_like(proj_cls_1),color ='r', label = 'c1')
    plt.plot(proj_cls_2, np.zeros_like(proj_cls_2),color ='g', label = 'c2')
    plt.show()
    
def plot_scatter(class1, class2, eigen_max, proj_classes):
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    colorUse = ("red", "green", "blue")
    markSym = ('o', '^', 's')
    plt.scatter(class1[:,0], class1[:,1],alpha = 1,marker=markSym[0],c = colorUse[0], s=50)
    plt.scatter(class2[:,0], class2[:,1],alpha = 1,marker=markSym[1],c = colorUse[1], s=50)
#    center1 = np.mean(class1,axis = 0)
#    center2 = np.mean(class2,axis = 0)
    plt.xlabel('Xaxis')
#    print (center1, center2)
#    plt.scatter(center1[0], center1[1], marker='*', c='b', s=100)
#    plt.scatter(center2[0], center2[1], marker='*', c='b', s=100)
#    x_pts = [center1[0], center2[0]]
#    y_pts = [center1[1], center2[1]]
#    plt.plot(x_pts, y_pts, marker = 'o')
    slope = eigen_max[0] / eigen_max[1]
    print (eigen_max)
    print ("slope", slope)
#    pt_1 = [(center1[0]+center2[0])/2, (center1[1]+center2[1])/2]
    pt_1 = [0, 0]
    y_int = pt_1[1] - slope * pt_1[0]   
    x_vals, y_vals = abline(slope, y_int)
    x_pts, y_pts = [], []
    x_pt1, y_pt1 = 0, 0
    for i in range(len(class1)):
        
        x_pt1 = class1[i].dot(eigen_max)
#        x_pt1  = x_pt1 + eigen_max[0]
        y_pt1 = y_int + slope * x_pt1
#        y_pt1 = y_pt1 + eigen_max[0]
        x_pts.append(x_pt1)
        y_pts.append(y_pt1)
    plt.scatter(x_pts, y_pts,alpha=0.4,c = 'b', s=50)
#    print (x_pts, y_pts)
    x_pts, y_pts = [], []
    for i in range(len(class2)):
        
        x_pt1 = class2[i].dot(eigen_max)
#        x_pt1  = x_pt1 + eigen_max[0]
        y_pt1 = y_int + slope * x_pt1
#        y_pt1 = y_pt1 + eigen_max[0]
        x_pts.append(x_pt1)
        y_pts.append(y_pt1)
    plt.scatter(x_pts, y_pts,alpha=0.4,c = 'g', s=50)
    plt.ylabel('Yaxis') 
    
def plot_scatter1(class1, class2, eigen_max, proj_classes):
    catColor = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF')
    colorUse = ("red", "green", "blue")
    markSym = ('o', '^', 's')
    plt.scatter(class1[:,0], class1[:,1],alpha = 1,marker=markSym[0],c = colorUse[0], s=50)
    plt.scatter(class2[:,0], class2[:,1],alpha = 1,marker=markSym[1],c = colorUse[1], s=50)
    center1 = np.mean(class1,axis = 0)
    center2 = np.mean(class2,axis = 0)
    plt.xlabel('Xaxis')
    print (center1, center2)
    plt.scatter(center1[0], center1[1], marker='*', c='b', s=100)
    plt.scatter(center2[0], center2[1], marker='*', c='b', s=100)
    x_pts = [center1[0], center2[0]]
    y_pts = [center1[1], center2[1]]
    plt.plot(x_pts, y_pts, marker = 'o')
    slope = (center2[1] - center1[1]) / (center2[0] - center1[0])
    pt_1 = [(center1[0]+center2[0])/2, (center1[1]+center2[1])/2]
    m_perp = (-1 / slope)
    y_int = pt_1[1] - m_perp * pt_1[0]   
    abline(m_perp, y_int)
    plt.ylabel('Yaxis')   

    
def main():
    l = 1
    opt_type = 2    #data set option
    k = 4           # Number of clusters
    btw_classes = 4     #plot contour plots between the selected classes
    dimension_of_data = 2          #please specify the dimension of data here.. 
    #Loads the required classes..    
    if(opt_type == 1):
        data_all_classes = loadcls.load_LS_trainingset()
        data_all_classes_test = loadcls.load_LS_test_set()
    elif(opt_type == 2):
        data_all_classes = loadcls.load_NLS_trainingset()
        data_all_classes_test = loadcls.load_NLS_test_set()
    elif(opt_type == 3):
        data_all_classes = loadcls.load_bovw_train_set()
        data_all_classes_test = loadcls.load_bovw_test_set()
     #arrange the information
#    elif (btw_classes == 4):
#        data_some_classes = data_all_classes
#        btw_num_of_classes = 3
#        class_names = ['C1','C2','C3']
#        selected_colors = [0, 1, 2]
     
    mean_two_classes, scatter_two_classes = [], []
    
    if( btw_classes == 4):
        data_cls0_cls1 = [data_all_classes[0], data_all_classes[1]]
        data_cls1_cls2 = [data_all_classes[1], data_all_classes[2]]
        data_cls0_cls2 = [data_all_classes[0], data_all_classes[2]]
#        data_cls0_cls1_test = [data_all_classes_test[0], data_all_classes_test[1]]
#        data_cls1_cls2_test = [data_all_classes_test[1], data_all_classes_test[2]]
#        data_cls0_cls2_test = [data_all_classes_test[0], data_all_classes_test[2]]
        
        cls01_cls12_cls02 = [data_cls0_cls1, data_cls1_cls2, data_cls0_cls2]
#        cls01_cls12_cls02_test = [data_cls0_cls1_test, data_cls1_cls2_test, data_cls0_cls2_test]
        proj_cls_all_combination = []
        pi_each_class_each_k_diff_combination, mu_each_class_each_k_diff_combination,sigma_each_class_each_k_diff_combination  = [], [], []
        W_diff_combinn = []
        for i in range(len(cls01_cls12_cls02)):
            data_some_classes_train = cls01_cls12_cls02[i]
            btw_num_of_classes = 2
            within_cls_scatterM, btw_cls_scatterM, mu1_minus_mu2 = find_Sb_and_Sw(data_some_classes_train, btw_num_of_classes)
            w, eigen_vector_max = find_project_vector(within_cls_scatterM, btw_cls_scatterM, mu1_minus_mu2)
            proj_tr_some_cls = find_projected_points(data_some_classes_train, w, eigen_vector_max)
            proj_cls_all_combination.append(proj_tr_some_cls)
            W_diff_combinn.append(w)             #eigen_vector_max is used for projection and w is not used.. we can interchange...
            train_data_reduced = proj_tr_some_cls
            
            mu_each_class_each_k = np.zeros((2, k, l))
            pi_each_class_each_k = np.zeros((2, k))
            sigma_each_class_each_k = np.zeros((2, k, l, l)) 
            
            for class_index in range(len(train_data_reduced)):
                print ("class", class_index)
            #// Do K - Means with reduced training data
                centers = find_initial_random_centers(train_data_reduced[class_index], k, l)
                centers_after_Kmeans, whcluster = gmm.computeKMC(train_data_reduced[class_index], centers, k)
                cluster_data = gmm.groupdata(train_data_reduced[class_index], centers_after_Kmeans, whcluster, k)
                
            #// Build GMM on training data
                log_likelihood_all_iters, pi_final_each_k, mu_final_each_k, sigma_final_each_k = gmm.compute_GMM(centers_after_Kmeans, train_data_reduced[class_index], cluster_data, k)
#                plot_iters_vs_loglikelihood(log_likelihood_all_iters, 3, k, class_index, l)
                pi_each_class_each_k[class_index] = pi_final_each_k
                mu_each_class_each_k[class_index] = mu_final_each_k
                sigma_each_class_each_k[class_index] = sigma_final_each_k
                
            pi_each_class_each_k_diff_combination.append(pi_each_class_each_k)
            mu_each_class_each_k_diff_combination.append(mu_each_class_each_k)
            sigma_each_class_each_k_diff_combination.append(sigma_each_class_each_k)
            
        conf_mat = compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k_diff_combination, mu_each_class_each_k_diff_combination, sigma_each_class_each_k_diff_combination, W_diff_combinn, k)
        gmm.print_matrix(conf_mat)
        performance_matrix = gmm.find_performance_matrix(conf_mat)
        print("\nperformance_matrix")
        gmm.print_matrix(performance_matrix)
        class_accuracy = gmm.find_accuracy(conf_mat)
        print("\nclass accuracy: ", class_accuracy)
        
#        plot_decision_boundary(data_all_classes, pi_each_class_each_k_diff_combination, mu_each_class_each_k_diff_combination, sigma_each_class_each_k_diff_combination, W_diff_combinn, k)
#        plot_classes(proj_cls_all_combination[0][0], proj_cls_all_combination[0][1])
        plot_scatter(cls01_cls12_cls02[0][0], cls01_cls12_cls02[0][1], W_diff_combinn[0],proj_cls_all_combination[0])
        
        
            
#            data_some_classes_test = cls01_cls12_cls02_test[i]
#            proj_test_some_cls = find_projected_points(data_some_classes_test, w, eigen_vector_max)
#            wh_class_arr = similar_to_confusion_matrix(proj_test_some_cls, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k)
    print ("here")
      
#        plot_classes(proj_cls_1, proj_cls_2)
        
        
        
            
main()
