# MLFromScratch
Implementation of widely used ML techniques from scratch

## Contents
- [Logistic Regression](#logistic-regression)
    - [Binary Classification](#binary-classification)


## Logistic Regression

keywords : classification, supervised learning, discriminative model, non-linear, gradient descent.

Opposed to its name, logistic regression is a supervised learning algorithm used for classification. 

### Binary Classification

<p align="center">
<img src="logistic-regression/images/theory.jpg" alt="theory" width="400"/>
</p>

- [This]() file contains the implementation of a binary class logistic regression algorithm. 
- The algorithm uses binary cross-entropy loss function. The loss function and its gradient implementation can be found [here]().
- Since it is a **non-linear formulation**, there is no closed form solution and descent methods are used to get to the loss function minimizer.

<br/>

<p align="center">
<img src="logistic-regression/images/lossfunc.png" alt="loss function" width="200"/>
<img src="logistic-regression/images/discriminator.png" alt="discriminator" width="200"/>
</p>
<p align = "center">
Left: Convergence of loss function value, Right: Discriminator learned from training data.
</p>
