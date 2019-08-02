---
title: Gaussian Mixture Model for Variational Autoencoder
date: 2019-08-02  13:29:22
---

This is learning notes and some thoughts for paper *Deep Unsupervised Clustering with Gaussian MIxture Variational Autoencoders* (ICLR2017 Rejected).

This topic is highly related to what I am currently focusing on: performing density estimation on latent space derived from autoencoders and constraining with GMM prior, which aims at achieveing better clustering performance and competitive modeling strength.

## Motivation of GMVAE

1. Over-regularization of VAE (with an isotropic Gaussian)
2. Extended: Inference in models with complicated latent structures can be difficult (VAE make it with single Gaussian, GMVAE extends into multiple Gaussian)

