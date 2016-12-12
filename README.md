# Overview

This is a machine learning library, made from scratch.  

It uses:
* `numpy`: for handling matrices/vectors
* `scipy`: for various mathematical operations
* `cvxopt`: for convex optimization
* `networkx`: for handling graphs in decision trees

It contains the following functionality:
* **Supervised Learning:**
  * Linear and Logistic regression
    * Closed form, Gradient descent, and SGD solvers
    * Regularization
  * Support Vector Machines
    * Soft and hard margins
    * Kernels
  * Tree Methods
    * CART (classificiation and regression)
    * PRIM
    * AdaBoost
    * Gradient Boost
    * Random Forests
  * Kernel Methods
    * Nadaraya average
    * Local linear regression
    * Kernel density classification
  * Discriminant Analysis
    * LDA, QDA, RDA
  * Prototype Methods
    * KNN
    * LVQ
    * DANN
  * Perceptron
* **Unsupervised Learning**
  * K means/mediods clustering
  * PCA
  * Gaussian Mixtures
* **Model Selection and Validation**

# Examples
Examples are shown in two dimensions for visualisation purposes, however, all methods can handle high dimensional data.
## Regression

* Linear and logistic regression with regularization

![Imgur](http://i.imgur.com/YJl0DfM.png)

![Imgur](http://i.imgur.com/eOarDws.png)

## Support Vector Machines

* Support vector machines maximize the margins between classes

![Imgur](http://i.imgur.com/Uw4puZ1.jpg)

* Using kernels, support vector machines can produce non-linear decision boundries.  The RBF kernel is shown below

![Imgur](http://i.imgur.com/dpSlL5z.jpg)

![Imgur](http://i.imgur.com/9Fw80Ex.png)

* An alternative learning algorithm, the perceptron, can linearly separate classes.  It does not maximize the margin, and is severely limited.

![SLiMG Image](https://i.sli.mg/PI5jJl.png)

## Tree Methods

* The library contains a large collection of tree methods, the basis of which are a decision trees for classification and regression

![Imgur](http://i.imgur.com/Mmkehxq.png)

These decision trees can be aggregated, and the library supports the following ensemble methods:
* AdaBoosting
* Gradient Boosting
* Random Forests

## Kernel Methods

Kernel methods estimate the target function by fitting seperate functions at each point using local smoothing of training data

* Nadaraya–Watson estimation uses a local weighted average

![Imgur](http://i.imgur.com/QptSDUu.png)

* Local linear regression uses weighted least squares to locally fit an affine function to the data

![Imgur](http://i.imgur.com/JM7VeQ2.png)

* The library also supports kernel density estimation (KDE) of data which is used for kernel density classification

![Imgur](http://i.imgur.com/VtAbSWs.png)

## Discriminant Analysis

* Linear Discriminant Analysis creates decision boundries by assuming classes have the same covariance matrix.
* LDA can only form linear boundries

![SLiMG Image](https://i.sli.mg/ukWqRT.png)

* Quadratic Discriminant Analysis creates deicion boundries by assuming classes have indepdent covariance matrices.
* QDA can form non-linear boundries.

![SLiMG Image](https://i.sli.mg/9jjM9f.png)

* Regularized Discriminant Analysis uses a combination of pooled and class covariance matrices to determine decision boundries.

![SLiMG Image](https://i.sli.mg/dEeLC2.png)

## Prototype Methods

* K-nearest neighbors determines target values by averaging the k-nearest data points.  The library supports both regression and classification.

![SLiMG Image](https://i.sli.mg/BGNG04.png)

* Learning vector quantization is a prototype method where prototypes are iteratively repeled by out-of-class data, and attracted to in-class data

![SLiMG Image](https://i.sli.mg/Ll8yl6.png)

* Discriminant Adaptive Nearest Neighbors (DANN). DANN adaptively elongates neighborhoods along boundry regions.
* Useful for high dimensional data.

![SLiMG Image](https://i.sli.mg/RVveSp.png)

## Unsupervised Learning

* K means and K mediods clustering.  Partitions data into K clusters.

![SLiMG Image](https://i.sli.mg/eBRfDT.png)

* Gaussian Mixture Models.  Assumes data are generated from a mixture of Gaussians and estimates those Gaussians via the EM algorithm.  The decision boundry between two estimated Gaussians is shown below.

![SLiMG Image](https://i.sli.mg/E75MG3.png)

* Principal Component Analysis (PCA) Transforms given data set into orthonormal basis, maximizing variance.

![SLiMG Image](https://i.sli.mg/45c9FN.png)


