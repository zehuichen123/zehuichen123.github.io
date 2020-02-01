---
layout: default
title: Uncertainty in Deep Learning
date: 2020-01-20 10:57 +0800
---

This post is my learning notes for paper <a href="http://mlg.eng.cam.ac.uk/yarin/blog_2248.html"><Uncertainty in Deep Learning></a> by **YARIN GAL**.

# Chapter1 Introduction: The Importance of Knowing What We Don't Know

## Model Uncertainty

- When encountering *out of distribution* data, model should yield out high uncertainty, conveying low confidence)
- noisy data
- Uncertainty in model parameters that best explain the observed data((a large number of possible models might be able to explain a given dataset, in which case we might be uncertain which model parameters to choose to predict with)
- structure uncertainty(what model structure should we use? how do we specify our model to extrapolate / interpolate well?)

