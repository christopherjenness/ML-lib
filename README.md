# Overview

![Travis](https://travis-ci.org/christopherjenness/ML-lib.svg?branch=master) [![Coverage Status](https://coveralls.io/repos/github/christopherjenness/ML-lib/badge.svg?branch=master)](https://coveralls.io/github/christopherjenness/ML-lib?branch=master)

This is a machine learning library, made from scratch.  

It uses:
* `numpy`: for handling matrices/vectors
* `scipy`: for various mathematical operations
* `cvxopt`: for convex optimization
* `networkx`: for handling graphs in decision trees

It contains the following functionality:
* **Supervised Learning:**
  * Linear and Logistic regression
    * Regularization
    * Solvers
      * Gradient descent
      * Steepest descent
      * Newton's method
      * SGD
      * Backtracking line search
      * Closed form solutions
  * Support Vector Machines
    * Soft and hard margins
    * Kernels
  * Tree Methods
    * CART (classificiation and regression)
    * PRIM
    * AdaBoost
    * Gradient Boost
    * Random Forests
  * Kernel Smoothing Methods
    * Nadaraya average
    * Local linear regression
    * Local logistic regression
    * Kernel density classification
  * Discriminant Analysis
    * LDA, QDA, RDA
  * Naive Bayes Classification
    * Gaussian
    * Bernoulli
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

* Linear and logistic regression with regularization.  Closed form, gradient descent, and SGD solvers.

![Imgur](http://i.imgur.com/dtihcxa.png)

![Imgur](http://i.imgur.com/MDecAmb.png)


## Support Vector Machines

* Support vector machines maximize the margins between classes

![Imgur](http://i.imgur.com/wojgsUN.png)

* Using kernels, support vector machines can produce non-linear decision boundries.  The RBF kernel is shown below

![Imgur](http://i.imgur.com/crDrds0.png)

![Imgur](http://i.imgur.com/NJ2oKls.png)

* An alternative learning algorithm, the perceptron, can linearly separate classes.  It does not maximize the margin, and is severely limited.

![Imgur](http://i.imgur.com/0XtFnWk.png)

## Tree Methods

* The library contains a large collection of tree methods, the basis of which are decision trees for classification and regression

![Imgur](http://i.imgur.com/Mf3KRCl.png)

These decision trees can be aggregated and the library supports the following ensemble methods:
* AdaBoosting
* Gradient Boosting
* Random Forests

## Kernel Methods

Kernel methods estimate the target function by fitting seperate functions at each point using local smoothing of training data

* Nadaraya–Watson estimation uses a local weighted average

![Imgur](http://i.imgur.com/EsqDMsS.png)

* Local linear regression uses weighted least squares to locally fit an affine function to the data

![Imgur](http://i.imgur.com/1hiVYKw.png)

* The library also supports kernel density estimation (KDE) of data which is used for kernel density classification

![Imgur](http://i.imgur.com/7pGHjf0.png)

## Discriminant Analysis

* Linear Discriminant Analysis creates decision boundries by assuming classes have the same covariance matrix.
* LDA can only form linear boundries

![Imgur](http://i.imgur.com/J9M3OBH.png)

* Quadratic Discriminant Analysis creates deicion boundries by assuming classes have indepdent covariance matrices.
* QDA can form non-linear boundries.

![Imgur](http://i.imgur.com/QpWG7UJ.png)

* Regularized Discriminant Analysis uses a combination of pooled and class covariance matrices to determine decision boundries.

![Imgur](http://i.imgur.com/AQ7bYWU.png)

## Prototype Methods

* K-nearest neighbors determines target values by averaging the k-nearest data points.  The library supports both regression and classification.

![Imgur](http://i.imgur.com/L7svJaA.png)

* Learning vector quantization is a prototype method where prototypes are iteratively repeled by out-of-class data, and attracted to in-class data

![Imgur](http://i.imgur.com/tSC85zu.png)

* Discriminant Adaptive Nearest Neighbors (DANN). DANN adaptively elongates neighborhoods along boundry regions.
* Useful for high dimensional data.

![Imgur](http://i.imgur.com/jyiq2z8.png)

## Unsupervised Learning

* K means and K mediods clustering.  Partitions data into K clusters.

![Imgur](http://i.imgur.com/cwLxmyR.png)

* Gaussian Mixture Models.  Assumes data are generated from a mixture of Gaussians and estimates those Gaussians via the EM algorithm.  The decision boundry between two estimated Gaussians is shown below.

![Imgur](http://i.imgur.com/3c0RAmj.png)

* Principal Component Analysis (PCA) Transforms given data set into orthonormal basis, maximizing variance.

![Imgur](http://i.imgur.com/un3ItuG.png)


