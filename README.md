# Perceptron-FDA-SVM
In the first part of the assignment, we implemented PCA to reduce the dimensions of 32D BoVW
to different lower dimensional space, built GMM and performed classification on it. We got
51.3% accuracy with 12 principal components.
Next we performed FDA and we got slightly better results on BoVW data. If the data is not
overlapping or concentric (one inside the other), FDA is better than PCA because it finds out
better representation by finding the direction of higher separation. However, FDA gives only one
direction in which separation is maximum. So, clearly we could see that FDA didn't work
properly on non linearly separable data because projected data in the chosen direction were
overlapping. Using GMM, we got 100% accuracy on the same dataset.
We also implemented Perceptron algorithm on linearly separable data and got 99.7% accuracy.
Although the data is linearly separable, perceptron does not give 100% accuracy because it does
not choose optimal separating hyperplane between the classes.
In order to get an optimal separating hyperplane, SVM is used. SVM on nonlinearly separable
data gives 100% accuracy. But on BoVW, even SVM with RBF kernel could not provide good
accuracy because features were overlapping. Even in the higher dimensions, features are
overlapping and not well separated.
