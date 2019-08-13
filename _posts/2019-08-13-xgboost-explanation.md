---
layout: default
title: XGBoost Explanation
date: 2019-08-13 13:09 +0800
---

In most kaggle competition, especially in tabular data ones, Gradient Boosting Machine(GBM) ha s shown its competitive performance, even surpass neural network by a large margin. Believe it or not, IEEE-CIS Fraud Detection[1] is an example. In this post, we will mainly focus on the algorithm on XGBoost and Gradient Boosting.

---

## Intro to XGBoost

XGBoost[2] is the abbreviation of eXtreme Gradient Boosting. The features of XGBoost are

- Based on the boosting tree which can deal with sparse data
- Weighted function for searching best split point
- Parallel and distributed computation
- Blocked data for high efficiency computation

---

## Fundamental Concepts

### Optimazation in Functional Space

In supervised learning problem, our target is to learn a best assumption $F^*(x)\in H$ which have the minimum general error.

$$ F^*(X) = arg min_{F(X)}E_{y, X}\Psi(y, F(X)) $$

$\Psi(y, F(X))$ is some kind of loss function.

since we don't know union distribution $P(X, Y)$, we can only use mean error in training data to perform nondestructive analysis. If we choose different hypothesis, we will get different $P$, and therefore, different mean error.Then our problem will become an optimazation in a $N$ dimensional space:

$$ min\Psi(P) = \Psi(y, F(x_1), F(x_2), ..., F(x_N)) $$

which means, our optimization target is to minimize the expectation of loss function which specified on $y$ in the marginal distribution of $x$. 

If our loss function is tractable, we can use gradient-based optimization methods to find the point $P^{\star}$ in space $P$ to minimize $\Psi(P)$. At this time, the assumption is what we want for $F^{*}$.

### Forward Stagewise Additive Modeling

FSAM is the core of boosting algorithm. The main idea of ensemble method is to train multiple models and combine them together to form a competitve model. In Boosting algorithms, we often generate base learners iteratively and add them together to form the final predictor.

$$ F_{m-1}(x) = \sum_{k=1}^{m-1}\alpha_k f_k(x) $$

In the next iteration, we are going to train $f_m(x)$ to learn the residual

$$ (\alpha_m, f_m(x)) = \text{arg min} E_{y,X} \Psi(y, F_{m-1}(X) + \alpha f(x)) \\ F_m{x} = F_{m-1}(x)+ \alpha_m f_m(x)$$

---

## Gradient Boosting

When talking about search for next residual, there is a simple way in optimization theory--linear search. Suppose our loss function is first-order tractable. Then we can compute the gradient for $P_{m-1}$ with respect to loss function and get the minus gradient $\rho_{m-1}$.

$$ \nabla \Psi(P_{m-1}) = -\frac{\partial\Psi(P_{m-1})}{\partial P_{m-1}} \\ =-\frac{\partial \Psi(F_{m-1}(X))}{\partial F_{m-1}(X)}$$

Then search point $P_m$ which meets the minimum loss in the minus gradient direction

$$ P_m = \text{arg min}_{\lambda} \Psi (P_{m-1} - \lambda \nabla \Psi(P_{m-1}))\\ = F_{m-1}(X) - \lambda_m \frac{\partial \Psi (F_{m-1}(X))}{\partial F_{m-1}(X)}$$

You may ask, why not just to learn the residual given from loss function directly?

I found the answer from book \<Statistical Learning Methods> by Li hang. When loss function is square error or exponantial error, the optimization will be easy. But what about other loss function format? Hence, Freidaman proposed gradient boosting method, which is the approximation of fastest descent algorithm. The core of this idea is to utilize the minus gradient at this point as the approximation of residual to learn a base tree.

---

## XGBoost

### Loss function for XGBoost

XGBoost is tree-based boosting algorithm and it optimize the original loss function and adds regularization term

$$ \Psi (y, F(X)) = \sum_{i=1}^N \Psi(y_i, F(X_i)) + \sum_{m=0}^T \Omega(f_m) \\ =  \sum_{i=1}^N \Psi(y_i, F(X_i)) + \sum_{m=0}^T (\gamma L_m + \frac{1}{2}\lambda\lvert\lvert\omega\lvert\lvert^2)$$

Among which $L_m$ is the number of leaves of $m^{th}$ iterative tree and $\omega$ is the output of each leave node in $f_m$.

XGBoost is also an addtive model. However, instead of to fit the minus gradient at $F_{m-1}(X)$, XGBoost learns the Talyor expansion at this point with respect to loss function and minimize this loss error to train base learner. 

Hhh, remember the last paragraph in the previous chapter, the intuition of computing minus gradient is that the residual loss is hard to optimize. But for XGBoost, it use Talyor expansion to conquer this issue

$$ \Psi (y, F(X)) = \sum_{i=1}^N \Psi(y_i, F_{m-1}(x_i) + f_m(x_i)) + \sum_{m=0}^T \Omega(f_m) \\ \qquad \qquad \qquad\qquad \approx \sum_{i=1}^N \Psi(y_i, F_{m-1}(X_i) + g_i f_m(x_i) + \frac{1}{2}h_i f_m^2(x_i)) + \sum_{m=0}^T \Omega(f_m) $$

among which $g_i$ is the first-order gradient at $P_{m-1}(X)$ with respect to $F_{m-1}(x_i)$ and $h_i$ is the second-order gradient at $P_{m-1}(X)$.

Since $\Psi(y_i, F_{m-1})$ is constant as for $m^{th}$ iteration, just move it outside of paranthesis

$$ \Psi_m = A +  \sum^N_{i=1}[g_i f_m(x_i) + \frac{1}{2}h_i f_m^2(x_i)] + \Omega(f_m)$$

### Convert sample-based loss into node-based

Before writing this post, I am wondering why need us to compute the optimal output for each leave node? Just find the best split point to reach the best information gain may be ok. However, instead of using entropy gain of information, XGBoost proposes a new target function which directly optimize loss function. To make it convinent when computing the best split point when constructing trees, we need to first convert the sample-based loss function into node-based ones. If you don't know what is sample-based and node-based, let's see this example:

Suppose at iteration $m$, we've construct one tree which has $L$ leave nodes($l_1, l_2, ..., l_L$), and assume that $I_j = (i\lvert q(x_i) = j)$ represents that the index of samples which are assigned to the  $j^{th}$ leave node, and $q$ denotes the result from the $m^{th}$ tree. Then we can convert the original sample-based loss function into node-based one. Please check the iterative symbol of sum function.

$$ \hat{\Psi}_m = \sum_{i=1}^N[g_i f_m(x_i) + \frac{1}{2}h_i f_m^2(x_i))] \\ \qquad \qquad \qquad = \sum_{j=1}^L[(\sum_{i\in I_j}g_i)\omega_j + \frac{1}{2}(\lambda + \sum_{i\in I_j}h_i)\omega^2_j] + \gamma L$$

Then, for each leave node, we can rewrite the function to make it simpler:

$$ \hat{\Psi} = \sum_{j=1}^L f(\omega_j) + \gamma L $$

where $f(\omega_j) = (\sum_{i\in I_j} \omega_j) + \frac{1}{2}(\lambda + \sum_{i\in I_j}h_i)\omega^2_j$ 

Now, we just need to compute leave node to get the optimal solution. Further, however can we find the optimal solution given these leave node parameters? An intuitive way is to use gradient based search: let its gradient equal to zero:

$$ \omega^{\star} = -\frac{\sum_{i\in I_j}g_j}{\lambda + \sum_{i\in I_j}h_j}$$

Until here, let's compute the expected minimum loss function here:

$$ \hat{\Psi}_m(q) = -\frac{1}{2}\sum_{j=1}^L \frac{G_j^2}{\lambda + H_j} + \gamma L$$

where $G_j = \sum_{i\in I_j}g_j$ and $H_j = \sum_{i\in I_j}h_j $. 

So here we are. Instead of iterate each samples, we just need to iterate leave nodes to compute the target loss function.

### Split Condition

Similar to GBDT, XGBoost here compute the decreasement of loss function before and after the split.

$$ \Delta\Psi = \frac{1}{2}[\frac{G_L^2}{\lambda + H_L} + \frac{G_R^2}{\lambda + H_R} - \frac{G^2}{\lambda + H}] -\gamma$$

---

## Summary

In this post, we've went through the whole process of XGBoost as well as something related to gradient boosting. However, there do exists something that we didn't cover, espeically about split methods and how to boost the speed of search. I decided to discuss these topic in next post, about LightGBM which advanced XGBoost with faster speed. And its core idea is to optimize the search phase.

---

## Reference

[1] <a href="https://www.kaggle.com/c/ieee-fraud-detection/">IEEE-CIS Fraud Detection- A kaggle Competition</a>

[2] [XGBoost: A Scalable Tree Boosting System]()