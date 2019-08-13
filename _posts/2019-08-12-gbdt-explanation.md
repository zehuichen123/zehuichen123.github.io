---
layout: default
title: Gradient Boosting Decision Tree Explanation
date: 2019-08-12 13:12 +0800
---

GBDT(Gradient Boosting Decision Tree) is a kind of iterative algorithm. It consists of multiple decision tree, and the result is derived from these trees. In this post, we will cover the following parts from intuitive way, in other words, not so mathematical :)

- Decision Tree(Regression DT and Classification DT)
- Bagging and Boosting
- An Example of GBDT
- Difference between GBDT and Adaboost
- Shrinkage

---

## Decision Tree

When it comes to decision tree, most people may think about C45 classification tree. However, GBDT doesn't use classification DT as its base tree, but the regression decision tree. The reason of this is that GBDT is to learn the residual of the prediction, and probably CDT can not learn such a uncertainty value which enables us to sum it up. So what's the difference between CDT and RDT?

For CDT, it iteratively choose different threshold for each feature, and find the one which yields the best improvement for entropy(maximize entropy here can be understood as let each node only holds one kind of instance). In most cases, we may not let every node holds only one kind instance, hence early stopping is introduced (For example the node can not be divided because it must hold at least $n$ instances).

The process of RDT is similar to CDT, however, for each node we can compute the prediction value. Take Age prediction as example, the prediction value is equal to the average of all instances which are assigned to this node. What's more, the judge value/ objecitve function is not the entropy but the prediciton error.

---

## Bagging and Boosting

Bagging and boosting are probably the most important topics in ensemble methods. Random forests is the representation of bagging. The main intuition is to randomly select samples from dataset and train your model. When obtaining plentiy of models, just average them up and you will get your answer (Ok, this is not the formal definition of Bagging, actually, it is a part of how random forests works). In view of our topic is GBDT, I won't spend much ink on Bagging. So what's boosting? Boosting is also utilizing multiple base learner to predict but totally different. The base learner inside GBDT is not to learn the final prediction value but the residual value of the previous base learners. Take age prediction as example, say A's true age is 16. If the first base learner predict A's age is 10 then the second learner is to learn the residual value of true age and the previous prediction value (16 - 10 = 6). 

---

## Example of GBDT

Let's take age prediction as an example, yeah, I love this example. Say we have 4 people A,B,C,D and their ages 14, 16, 24, 26. If we choose a DT to predict the age, the result may probably looks like this:

<center><img src="/images/gbdt/dt.png" height="300"></center>
<center><i>An illustraction of an decision tree.</i></center>
Then we utilize GBDT to learn this task, and we limit the maximum number of nodes is 2, and n_estimators=2, then we may get this figure.

<center><img src="/images/gbdt/gbdt.png"></center>
<center><i>An illustraction of an GBDT.</i></center>
Note that in the second tree, it not learn the value of 14, 16, 24, 26 but the residual value from the previous base learners.

---

## GBDT and AdaBoost

Adaboost and GBDT are both boosting model. So what's the difference between them?

For Adaboost, it compute the weight for each instance based on the previous prediction, and assign more weight on those which are wrongly classified and assign low penalty to those classified correctly. Actually, the intuition of Bootstrap and Adaboost are the same. But what about GBDT? GBDT is to learn the residual function of previous prediction. In some ways, you can think about it as learning something that is hard to learn. 

---

## Shrinkage

Shrinkage can be viewed as a competitive regularization method for boosting. The idea behind shrinkage is more little steps towards final result is better/less likely to overfitting than large steps towards optimal. It just assure that each tree may only learn part of truth. And multiple trees can learn better.

No Shrinkage:
<center>
$$
y_{n+1} \to residual(\sum_{i=1}^n y_i)\\
y_{pred} = sum(y1,..., y_{n+1})
$$
</center>
With shrinkage:
<center>
$$
y_{n+1} \to residual(\sum_{i=1}^n y_i)\\
y_{pred} = sum(y1,..., y_{n}) + step *y_{n+1}
$$
</center>