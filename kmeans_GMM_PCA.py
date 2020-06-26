from copy import deepcopy
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
import math
import pylab as plt
from os import listdir
from os.path import isfile, join

colors = (u'#ff0000', u'#228b22', u'#0000cd', u'#FF8000', u'#B23AEE', u'#8FBC8F', u'#BCEE68', u'#00EEEE', u'#EE6A50', u'#838B8B', u'#CDAA7D', u'#1C86EE', u'#8B3E2F',u'#6495ED',u'#8B8878',u'#00BFFF')

#Loads Non-Linearly separable data
def load_NLS_trainingset():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/train_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/train_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/train_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    #dataf = np.concatenate((pd_data_class1, pd_data_class2, pd_data_class3)) #2 dimentional - 1500 * 2
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    data_all_classes = [np_data_class1, np_data_class2, np_data_class3] #3 dimentional - 3 * training_class_size * 2
    
    return data_all_classes

def load_NLS_test_set():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/test_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/test_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/NLS/test_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    #dataf = np.concatenate((pd_data_class1, pd_data_class2, pd_data_class3)) #2 dimentional - 1500 * 2
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    data_all_classes_test = [np_data_class1, np_data_class2, np_data_class3] #3 dimentional - 3 * training_class_size * 2
    
    return data_all_classes_test
    
#Loads Real-world speech data 
def load_speech_trainingset():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/train_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/train_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/train_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    #dataf = np.concatenate((pd_data_class1, pd_data_class2, pd_data_class3))
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    data_all_classes = [np_data_class1, np_data_class2, np_data_class3]
    
    return data_all_classes

def load_speech_test_set():
    pd_data_class1 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/test_class1.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class2 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/test_class2.txt", header = None, delimiter=' ', usecols=(0, 1))
    pd_data_class3 = pd.read_csv("/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/RD/test_class3.txt", header = None, delimiter=' ', usecols=(0, 1))
    #dataf = np.concatenate((pd_data_class1, pd_data_class2, pd_data_class3))
    np_data_class1 = np.array(pd_data_class1)
    np_data_class2 = np.array(pd_data_class2)
    np_data_class3 = np.array(pd_data_class3)
    data_all_classes_test = [np_data_class1, np_data_class2, np_data_class3]
    
    return data_all_classes_test

def load_scene_image_training_set():
    train_folder = ["coast", "industrial_area", "pagoda"]    
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Scene_Images/Train_Feature_Vectors_Stripped"
    i = 0
    trnclsf = []
    for scene_type in train_folder:
        all_features_path = join(dir_path, scene_type)
        frame = pd.DataFrame()
        dataA = []
        for each_img_file_name in listdir(all_features_path):
            img_path = join(all_features_path, each_img_file_name)
            cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
            data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
            dataA.append(data_1)
            i = i + len(data_1)
        frame = pd.concat(dataA)
        frame_ip = [tuple(i) for i in frame.as_matrix()]
        dataf = np.array(frame_ip)
        trnclsf.append(dataf)
    return trnclsf

def load_scene_image_test_set():
    train_folder = ["coast", "industrial_area", "pagoda"]    
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Scene_Images/Test_Feature_Vectors_Stripped"
    i = 0
    trnclsf = []
    for scene_type in train_folder:
        all_features_path = join(dir_path, scene_type)
        frame = pd.DataFrame()
        dataA = []
        a = sorted(listdir(all_features_path))
        if scene_type == "coast":
            a = a[1:]
        for each_img_file_name in a:
            img_path = join(all_features_path, each_img_file_name)
            cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23)
            data_1 = pd.read_csv(img_path, header = None, delimiter=',', usecols = cols)
            dataA.append(data_1)
            i = i + len(data_1)
        frame = pd.concat(dataA)
        frame_ip = [tuple(i) for i in frame.as_matrix()]
        dataf = np.array(frame_ip)
        trnclsf.append(dataf)
    return trnclsf

def load_scene_image_test_set_():
    train_folder = ["coast", "industrial_area", "pagoda"]    
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Scene_Images/Test_Feature_Vectors_Stripped"
    i = 0
    data_all_classes_test = []
    for scene_type in train_folder:
        all_features_path = join(dir_path, scene_type)
        data_each_class = []
        a = sorted(listdir(all_features_path))
        if scene_type == "coast":
            a = a[1:]
        for each_img_file_name in a:
            img_path = join(all_features_path, each_img_file_name)
            data_1 = np.loadtxt(img_path, delimiter=',', dtype = 'int')
            data_each_class.append(data_1)
        
        data_each_class = np.array(data_each_class)
        data_all_classes_test.append(data_each_class)

    return data_all_classes_test

def load_bovw_train_set():
    dir_path = "/Users/3pi/Documents/Pattern Recognition/Ass2_KMeans_GMM/Train_BovW_Feature_Vectors/"
    train_folder = ["coast", "industrial_area", "pagoda"]
    data = []
    for scene_type in train_folder:
        all_txt_files_path = join(dir_path, scene_type)
        for each_img_file_name in listdir(all_txt_files_path):
            txt_path = join(all_txt_files_path, each_img_file_name)
            cols = (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24, 25, 26, 27, 28, 29, 30, 31)
            data_1 = pd.read_csv(txt_path, header = None, delimiter=',', usecols = cols)
            data_1 = np.asarray(data_1)
            data.append(data_1)
    return data

def find_initial_random_centers(data, k, color_id):
    n = data.shape[0] # Number of training data points
    c = data.shape[1] # Number of features in the data (number of dimentions)
    mean = np.mean(data, axis=0) #[x_component_mean, y_component_mean]
    std = np.std(data, axis = 0) #[x_component_std, y_component_std]
    centers = []
    for i in range(k):
        centers = np.append(centers, data[np.random.randint(n)], axis = 0)
    centers = np.resize(centers, (k, 24))
#    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=100)
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
#        curr_centers = curr_centers[~np.isnan(curr_centers)]
        error = np.linalg.norm(curr_centers - prev_centers)

    return curr_centers, whcluster
    
def groupdata(data, centers_after_Kmeans, whcluster, k):
    cluster_data = []
    cluster_data = np.array([data[whcluster==i] for i in range(k)])
    return cluster_data
    
def contour_plot(data, centers, cluster_d, k, tot_rows):
    cat_color = (u'#FFAFAF', u'#BBFFB9', u'#BBB9FF', u'#ffff00')
    colors = (u'#ff0000', u'#228b22', u'#0000cd', u'#FF8000', u'#B23AEE', u'#8FBC8F', u'#BCEE68', u'#00EEEE', u'#EE6A50', u'#838B8B', u'#CDAA7D', u'#1C86EE', u'#8B3E2F',u'#6495ED',u'#8B8878',u'#00BFFF')

    for i in range(k):
#        print ("cluster", i, cluster_d[i])
        xy_max = np.max(cluster_d[i], axis = 0)     #max two points, x,y in cluster i 
        xy_min = np.min(cluster_d[i], axis = 0)    #min two points, x,y in cluster i
        cov_mat = np.cov(cluster_d[i][:,0],cluster_d[i][:,1])
        meanc = np.mean(cluster_d[i], axis = 0)
#        probc = len(cluster_d[i]) / tot_rows
#        print xy_max, xy_min
        plt.scatter(cluster_d[i][:,0], cluster_d[i][:,1],c = colors[i], s = 5)
#        plt.scatter(cluster_d[i][:,0], cluster_d[i][:,1], s = 5)
        upts = cluster_d[i][:,0]
        vpts = cluster_d[i][:,1]
        xpts, ypts = [], []
#        upts = np.arange(xy_min[0], xy_max[0], (xy_max[0] - xy_min[0]) * dx) #find all points from xmin to xmax in steps of 0.005
#        vpts = np.arange(xy_min[1], xy_max[1], (xy_max[1] - xy_min[1]) * dx) #find all points from ymin to ymax in steps of 0.005
        upts = np.linspace(xy_min[0], xy_max[0], 400)
        vpts = np.linspace(xy_min[1], xy_max[1], 400)
        xpts, ypts = np.meshgrid(upts, vpts) #build a mesh using the above points
        zpts = []
        for j in range(len(xpts)):
            zx = [] 
            for k in range(len(ypts)):
#                gx = np.linalg.norm([xpts[j][k], ypts[j][k]] - centers[i])
                gx = find_gix([xpts[j][k], ypts[j][k]], meanc, cov_mat)
                zx.append(gx)
            zpts.append(zx)
        plt.contour(xpts,ypts,zpts, 4, colors = colors[i])
#        plt.contour(xpts,ypts,zpts, 4)
        
        
def find_gix(Xi, Mui, cov_matrix):
    x = np.subtract(Xi, Mui)
    xt = x.T
    inv_covmat = np.linalg.inv(cov_matrix)
    det_cov = np.linalg.det(cov_matrix)
    if det_cov == 0:
        det_cov = 0.0000001
    nix = -(1 / 2) * np.matmul(np.matmul(xt, inv_covmat), x) - (1 / 2) * np.log(det_cov) - np.log(2*np.pi)
    return nix


def main():
    opt_type = 3 #data set option
    k = 16   # Number of clusters
    btw_classes = 4 #plot contour plots between the selected classes
    
    #Loads the required classes..    
    if(opt_type == 1):
        data_all_classes = load_NLS_trainingset()
        data_all_classes_test = load_NLS_test_set()
    elif(opt_type == 2):
        data_all_classes = load_speech_trainingset()
        data_all_classes_test = load_speech_test_set()
    elif opt_type == 3:
        data_all_classes = load_scene_image_training_set()
        #data_all_classes_test = load_scene_image_test_set_()
        #np.save("output_data_all_classes_test.npy", data_all_classes_test)
    elif opt_type == 4:
        data_all_classes = load_bovw_train_set()
    
    #arrange the information
    if (btw_classes == 1):
        data_some_classes = [data_all_classes[0], data_all_classes[1]]
        btw_num_of_classes = 2
        class_names = ['C1','C2','']
        selected_colors = [0, 1]
    elif (btw_classes == 2):
        data_some_classes = [data_all_classes[1], data_all_classes[2]]
        btw_num_of_classes = 2
        class_names = ['C2','C3','']
        selected_colors = [1, 2]
    elif (btw_classes == 3):
        data_some_classes = [data_all_classes[0], data_all_classes[2]]
        btw_num_of_classes = 2
        class_names = ['C1','C3','']
        selected_colors = [0, 2]
    elif (btw_classes == 4):
        data_some_classes = data_all_classes
        btw_num_of_classes = 3
        class_names = ['C1','C2','C3']
        selected_colors = [0, 1, 2]
    
    if btw_num_of_classes == 3 and opt_type == 4:
        mu_each_class_each_k = np.zeros((3, k, 32))
        pi_each_class_each_k = np.zeros((3, k))
        sigma_each_class_each_k = np.zeros((3, k, 32, 32))    
    elif btw_num_of_classes == 3 and opt_type == 3:
        mu_each_class_each_k = np.zeros((3, k, 24))
        pi_each_class_each_k = np.zeros((3, k))
        sigma_each_class_each_k = np.zeros((3, k, 24, 24))    
    elif btw_num_of_classes == 3:
        mu_each_class_each_k = np.zeros((3, k, 2))
        pi_each_class_each_k = np.zeros((3, k))
        sigma_each_class_each_k = np.zeros((3, k, 2, 2))   

    for i in range(btw_num_of_classes):
        print ("class", i)
        centers = find_initial_random_centers(data_some_classes[i], k, selected_colors[i])
        centers_after_Kmeans, whcluster = computeKMC(aqdata_some_classes[i], centers, k)
        cluster_data = groupdata(data_some_classes[i], centers_after_Kmeans, whcluster, k) #cluster data contains data of each cluster seperately
        log_likelihood_all_iters, pi_final_each_k, mu_final_each_k, sigma_final_each_k = compute_GMM(centers_after_Kmeans, data_some_classes[i], cluster_data, k)
        plot_iters_vs_loglikelihood(log_likelihood_all_iters, opt_type, k, i)
        pi_each_class_each_k[i] = pi_final_each_k
        mu_each_class_each_k[i] = mu_final_each_k
        sigma_each_class_each_k[i] = sigma_final_each_k

        np.save("output_pi_k_16.npy", pi_each_class_each_k)
        np.save("output_mu_k_16.npy", mu_each_class_each_k)
        np.save("output_sigma_k_16.npy", sigma_each_class_each_k)

    # pi_each_class_each_k = np.load("output_pi_k_2.npy")
    # mu_each_class_each_k = np.load("output_mu_k_2.npy")
    # sigma_each_class_each_k = np.load("output_sigma_k_2.npy")

    data_all_classes_test = np.load("output_data_all_classes_test.npy")
    confusion_matrix = compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, opt_type)
    print("\nconfusion_matrix")
    print_matrix(confusion_matrix)

    performance_matrix = find_performance_matrix(confusion_matrix)
    print("\nperformance_matrix")
    print_matrix(performance_matrix)

    class_accuracy = find_accuracy(confusion_matrix)
    print("\nclass accuracy: ", class_accuracy)

def print_matrix(matrix):
    for i in range(len(matrix)):
        print(matrix[i])

def test_which_class(x, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, opt_type):
    prob_xi_each_class = []
    for i in range(3):
        summation = 0
        for j in range(k):
            nix = norm_pdf_multivariate(x, mu_each_class_each_k[i][j], sigma_each_class_each_k[i][j])
            summation += pi_each_class_each_k[i][j] * nix
        prob_xi_each_class.append(summation)
        
    if opt_type == 3:
        return prob_xi_each_class
    else:
        return prob_xi_each_class.index(max(prob_xi_each_class))

def compute_confusion_matrix(data_all_classes_test, pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, opt_type):
    confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for each_class in range(3):
        if opt_type == 3:
            for each_img in range(len(data_all_classes_test[each_class])):
                prob_each_img_each_class = np.zeros(3)
                for each_24d_feature in range(len(data_all_classes_test[each_class][each_img])):
                    prob_each_img_each_class += np.log(np.array(test_which_class(data_all_classes_test[each_class][each_img][each_24d_feature], pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, opt_type)))
                class_num = np.where(prob_each_img_each_class == (max(prob_each_img_each_class)))
                confusion_matrix[each_class][class_num[0][0]] += 1
        else:
            for j in range(len(data_all_classes_test[each_class])):
                class_num = test_which_class(data_all_classes_test[each_class][j], pi_each_class_each_k, mu_each_class_each_k, sigma_each_class_each_k, k, 0)
                confusion_matrix[each_class][class_num] += 1

    return confusion_matrix

def find_performance_matrix(confusion_matrix):
    performance_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for class_num in range(3):
        performance_matrix[class_num][0] = find_precision(confusion_matrix, class_num)
    for class_num in range(3):
        performance_matrix[class_num][1] = find_recall_rate(confusion_matrix, class_num)
    for class_num in range(3):
        performance_matrix[class_num][2] = find_f_score(performance_matrix[class_num])

    #find mean precision, mean recall rate and mean f score
    for metric in range(3):
        performance_matrix[3][metric] = (performance_matrix[0][metric] + performance_matrix[1][metric] + performance_matrix[2][metric]) / 3

    round_off_performance_matrix(performance_matrix)

    return performance_matrix

def find_accuracy(confusion_matrix):
    total_samples = 0
    for class_num in range(3):
        for metric in range(3):
            total_samples += confusion_matrix[class_num][metric]

    class_accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[2][2]) / total_samples * 100

    return class_accuracy

def round_off_performance_matrix(performance_matrix):
    for i in range(len(performance_matrix)):
        for j in range(len(performance_matrix[i])):
            performance_matrix[i][j] = round(performance_matrix[i][j], 2)


def find_precision(confusion_matrix, class_num):
    total_samples_classfied_as_class_num = 0
    for i in range(3):
        total_samples_classfied_as_class_num += confusion_matrix[i][class_num]
    if total_samples_classfied_as_class_num == 0:
        return 0
    precision_rate = (confusion_matrix[class_num][class_num] / total_samples_classfied_as_class_num) * 100

    return precision_rate

def find_recall_rate(confusion_matrix, class_num):
    total_samples_in_class = 0
    for j in range(3):
        total_samples_in_class += confusion_matrix[class_num][j]
    if total_samples_in_class == 0:
        return 0
    recall_rate = (confusion_matrix[class_num][class_num] / total_samples_in_class) * 100

    return recall_rate

def find_f_score(array):
    precision = array[0]
    recall = array[1]
    if (precision * recall) == 0:
        return 0
    f_score = (precision * recall) / ((precision + recall) / 2)

    return f_score

def plot_iters_vs_loglikelihood(log_likelihood_all_iters, opt_type, k, i):
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
    plt.savefig(str(opt_type) + "_K_" + str(k) + "_" + str(title) + ".png")
    plt.clf()

def compute_GMM(centers_after_Kmeans, data, cluster_data, k):
    mu_init_each_k = centers_after_Kmeans
    pi_init_each_k = compute_init_pi(cluster_data, k)
    sigma_init_each_k = compute_init_sigma(data, cluster_data, k)
    prob_each_Xn_each_k = compute_prob_each_Xn_each_k_1(data, mu_init_each_k, pi_init_each_k, sigma_init_each_k, k)
    total_prob_each_Xn = np.sum(prob_each_Xn_each_k, axis = 1)
    log_likelihood_old = np.sum(np.log(total_prob_each_Xn))
    log_likelihood_new = 0
    log_likelihood_all_iters = [log_likelihood_old]
    while (log_likelihood_new - log_likelihood_old) > 0.0001:
        log_likelihood_old = log_likelihood_new
        log_likelihood_new, prob_each_Xn_each_k, total_prob_each_Xn, pi_new_each_k, mu_new_each_k, sigma_new_each_k = compute_log_likelihood(data, prob_each_Xn_each_k, total_prob_each_Xn, k)
        log_likelihood_all_iters.append(log_likelihood_new)
    print(log_likelihood_all_iters)
    return log_likelihood_all_iters, pi_new_each_k, mu_new_each_k, sigma_new_each_k

def compute_log_likelihood(data, prob_each_Xn_each_k, total_prob_each_Xn, k):
    responsibility_term_each_Xn_eack_k = estimation_step(prob_each_Xn_each_k, total_prob_each_Xn, k)
    responsibility_term_each_Xn_eack_k = np.nan_to_num(responsibility_term_each_Xn_eack_k)
    pi_new_each_k, mu_new_each_k, sigma_new_each_k = maximization_step(data, responsibility_term_each_Xn_eack_k, k)
    prob_each_Xn_each_k =  compute_prob_each_Xn_each_k(data, mu_new_each_k, pi_new_each_k, sigma_new_each_k, k)
    prob_each_Xn_each_k = np.nan_to_num(prob_each_Xn_each_k)
    total_prob_each_Xn = np.sum(prob_each_Xn_each_k, axis = 1)
    log_likelihood_new = np.sum(np.log(total_prob_each_Xn))
    log_likelihood_new = np.nan_to_num(log_likelihood_new)

    return(log_likelihood_new, prob_each_Xn_each_k, total_prob_each_Xn, pi_new_each_k, mu_new_each_k, sigma_new_each_k)

def maximization_step(data, responsibility_term_each_Xn_eack_k, k):
    eff_num_pts_in_each_k = np.sum(responsibility_term_each_Xn_eack_k, axis = 0)
    pi_new_each_k = eff_num_pts_in_each_k / len(data)
    mu_new_each_k = np.zeros((1, k, len(data[0])))
    sigma_new_each_k = np.zeros((1, k, len(data[0]), len(data[0])))
    for i in range(k):
        numerator = np.zeros((1, len(data[0])))
        for j in range(len(data)):
            numerator += data[j] * responsibility_term_each_Xn_eack_k[j][i]
        mu_new_each_k[0][i] = numerator / eff_num_pts_in_each_k[i]
        numerator_1 = np.zeros((len(data[0]), len(data[0])))
        for j in range(len(data)):
            x_mu_new = np.matrix(data[j] - mu_new_each_k[0][i])
            numerator_1 += responsibility_term_each_Xn_eack_k[j][i] * (x_mu_new.T * x_mu_new)

        sigma_new_each_k[0][i] = numerator_1 / eff_num_pts_in_each_k[i]
        sigma_new_each_k[0][i] = np.nan_to_num(sigma_new_each_k[0][i])
   
        # for p in range(24):
        #     for q in range(24):
        #         if p != q:
        #             sigma_new_each_k[0][i][p][q] = 0

    return pi_new_each_k, mu_new_each_k, sigma_new_each_k

def estimation_step(prob_each_Xn_each_k, total_prob_each_Xn, k):
    responsibility_term_each_Xn_eack_k = np.zeros((len(prob_each_Xn_each_k), k))
    for i in range(len(prob_each_Xn_each_k)):
        for j in range(k):
            responsibility_term_each_Xn_eack_k[i][j] = prob_each_Xn_each_k[i][j] / total_prob_each_Xn[i]

    return responsibility_term_each_Xn_eack_k

def compute_prob_each_Xn_each_k(data, mu_each_k, pi_each_k, sigma_each_k, k):
    prob_each_Xn_each_k = np.zeros((len(data), k))
    for i in range(k):
        nix = []
        for Xi in data:
            nix.append(norm_pdf_multivariate(Xi, mu_each_k[0][i], sigma_each_k[0][i]))
        nix = np.asarray(nix)
        prob_each_Xn_each_k[:, i] = pi_each_k[i] * nix #probability for each data point belonging to cluster k

    return prob_each_Xn_each_k

def compute_prob_each_Xn_each_k_1(data, mu_each_k, pi_each_k, sigma_each_k, k):
    prob_each_Xn_each_k = np.zeros((len(data), k))
    for i in range(k):
        nix = []
        for Xi in data:
            a = norm_pdf_multivariate(Xi, mu_each_k[i], sigma_each_k[0][i])
            nix.append(a)
        nix = np.asarray(nix)
        prob_each_Xn_each_k[:, i] = pi_each_k[i] * nix #probability for each data point belonging to cluster k
    
    return prob_each_Xn_each_k

def norm_pdf_multivariate(x, mu, sigma):
    size = len(x)
    if size == len(mu) and (size, size) == sigma.shape:
        det = np.linalg.det(sigma)
        if det <= 0:
            return 0
        norm_const = 1.0/ (math.pow((2*math.pi),float(size)/2) * math.pow(det,1.0/2))
        x_mu = np.matrix(x - mu)
        inv = np.linalg.inv(sigma)        
        result = math.pow(math.e, -0.5 * (x_mu * inv * x_mu.T))
        return norm_const * result
    else:
        raise NameError("The dimensions of the input don't match")

def compute_init_sigma(data, cluster_data, k):
    sigma_init_each_k = np.zeros((1, k, len(data[0]), len(data[0])))
    for i in range(k):
        cov_mat = np.cov(cluster_data[i].T)
        cov_mat = np.nan_to_num(cov_mat)
        sigma_init_each_k[0][i] = cov_mat
        # for p in range(24):
        #     for q in range(24):
        #         if p != q:
        #             sigma_init_each_k[0][i][p][q] = 0
    return sigma_init_each_k

def compute_init_pi(cluster_data, k):
    pi_init_all_clusters = [0] * k
    total_data_points = 0
    for i in range(k):
        total_data_points += len(cluster_data[i])
    for i in range(len(cluster_data)):
        pi_init_all_clusters[i] = len(cluster_data[i]) / total_data_points
    return pi_init_all_clusters

