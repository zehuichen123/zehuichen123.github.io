---
layout: default
title: Segmentation Related Models and Loss
date: 2019-08-09 13:43 +0800
---

In this post, we will cover about some techniques about image segmentation, from FCN to DeepLab, from cross entropy loss to lovász-softmax loss and so on.

---

## Models

### Fully Convolutional Net

The first model is Fully Convolutional Nets(FCN) [1]. The main contribution of FCN can be three folds:

- Without any fully connected layers.
- Enlarge the size of image: deconvolutional layer.
- Skip structure; ensemble various depth features.

<img src="/images/segment_topics/fcn.png">

Without using any DNN layers, FCN allows us to receive different size of images. Besides, FCN provides us with some upsampling methods, since we need to enlarge the size of feature map into original image size. 

> ### Upsample
>
> In the application of computer vision, after the feature extraction by CNN, the output data will usually become smaller (mainly due to pooling operation). However, sometimes we want the size of our output images can be exactly the same as input images. Hence, the operation to enlarge the size of image into higher resolution projection is called upsample. 

Here, we only focus on Transposed convolution. Actually, deconvolution is a special convolutional operation. If we defold the input images as one dimensional vector $X$ and the output images as the same $Y$, the convolutional operation can be represented as $Y = CX$, which is to say, the deconvolution can be rewriten as $X = C^TY$. Finally, what we do is to pad zeros around center points and perform new convolutional operations. 

---

As we mentioned before, deconvolutional operation can not extactly recover original information, even with the same filter parameters(actually is a non-inversable). This means FCN is unable to maintain positional information for each pixel. There are two totally different ways to tackle with this issue:

- Encoder-Decoder Framework, and enable shortcut connection between encoder and decoder. U-Net is one of those classical models.
- Apply dilated/atrous convolutions, which enables us to remove pooling layers.

---

### U-Net

The structure of U-Net[2] is simple as well.

<img src="/images/segment_topics/unet.png">

Briefly, there are two main points that makes U-Net peforms better than FCN, IMO. 

Firstly, FCN applys summeration for multiple feature maps while U-Net chooses concatenation. Mathematically, summeration will lose information, right?

Another point is concatenation enables U-Net to consider more global, but worse resolution as well as more local but better resolution information.

---

### DeepLab v3

DeepLab[3] removes pooling layers and replaces it with dialated convolutions. The dialated/atrous convolution can be illustrated as following figures:

<center><img src="/images/segment_topics/deeplab_1.png" height="200"></center>

<center><i>Atrous convolution with kernel size 3 × 3 and different rates. Standard convolution corresponds to atrous convolution with rate = 1. Employing large value of atrous rate enlarges the model’s field-of-view, enabling object encoding at multiple scales.</i></center>

The main contribution of DeepLabv3 System is listed as follows:

- employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple scales.
- Advanced *Atrous Spatial Pyramid Pooling* module, which probe convolutional features at mutiple scales, with image-level features encoding global context.

The above points are proposed to deal with two challenges:

- Multiple Pooling or convolution worse the feature resolution, which leads to the uncertainty of positional information.
- Multiple scale objects in the same images.

<img src="/images/segment_topics/deeplab_2.png">

<img src="/images/segment_topics/deeplab_3.png">

<center><i>Parallel modules with atrous convolution (ASPP), augmented with image-level features.</i></center>

---

### Some Thoughts about These Models

For most Kaggle image segmentation competitions, you will find U-Net is probably the most popular framework , not one of. The reason of this can be concluded:

- U-Net itself is one competitve model in image segmentation.
- Actually, U-Net provides us with a extendable framework, an encoder-decoder framework, which enables us to replace the encoder/decoder structure with different architecture, like ResBlock, SENet or EffientNet. The easiness of U-Net to ensemble other advanced framework is probably the most important reason that makes it so popular.

---

## Loss

The loss in image segmentation is also an important topic. The extremely imbalanced positive dataset poses great challenges to our training phase. A carefully designed loss can benefit you with better performance.

### Log Loss

The log loss is actually binary cross entropy loss, which is widely used in binary classificaiton tasks. The formula can be written as

$$ L = -y \cdot log(y') - (1-y)\cdot log(1-y)$$

However, this loss function holds an evident drawback: when the positive samples are much less than negative samples, the model can not learn very well from positive samples due to lack of information.

### DICE Loss

Firstly, we will define the similarity between two shapes. We use A, B to denote the points inside these two shapes. Then,

$$DSC(A, B) = 2 \lvert A \cap B\lvert / \lvert A \cup B \lvert $$

So the loss can be written as

$$ DL_2 = 1 - \frac{\sum_{n=1}^Np_n r_n + \epsilon}{\sum_{n=1}^N p_n + r_n + \epsilon} - \frac{\sum_{n=1}^N(1-p_n)(1-r_n)+\epsilon}{\sum_{n=1}^N2-p_n-r_n + \epsilon}$$

Here is an implementation with Keras

```python
def dice_coef(y_true, y_pred, smooth=1.):
  intersaction = K.sum(y_true * y_pred, axis=[1,2,3])
  union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3])
  return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

def dice_coef_loss(y_true, y_pred):
  return 1 - dice_coef(y_true, y_pred, smooth=1.)
```

### Focal Loss

$$f(n) = \begin{cases} -(1-y')^\gamma \text{log}y', &\text{if y=1} \\ -y'^\gamma\text{log}(1-y'), & \text{if y=0}\end{cases}  $$

In focal loss, we focus more on samples which is hard to classify, and assign low penalty to those samples which is easier. Take $\gamma = 2 $ as an example,

- for positive samples, if our prediction is 0.97, then it must be an easy-to-classify sample, hence $(1-0.97)^\gamma$ will be very small. On the contrary, if our prediction is 0.3, then it would be a hard-to-classify sample, then $(1-0.3)^\gamma$ will be quite large (at least larger than the previous one)
- vice versa

Additionally, focal loss also use $\alpha$, called balance factor to balance the distribution between positive and negative samples. 

$$f(n) = \begin{cases} -\alpha (1-y')^\gamma \text{log}y', &\text{if y=1} \\ -(1-\alpha)y'^\gamma\text{log}(1-y'), & \text{if y=0}\end{cases}$$

```python
def focal_loss(y_true, y_pred):
  gamma = 0.75
  alpha = 0.25
  pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
  pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
  
  pt_1 = K.clip(pt_1, 1e-3, .999)
  pt_2 = K.clip(pt_0, 1e-3, .999)
  
  return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
  
```

### Lovasz-Softmax Loss

IoU (Jaccard index) is actually intractable, since it needs equal operation. However, Jaccard loss can be performed with Lovasz extension, which enables discrete space into continuous space, which will be tractable. 

BTW, lovasz-softmax loss is used during fine-tuning phase. A common practice is to train the model with BCE loss/ DICE loss a few epochs and then turn the loss to Lovasz loss + DICE/BCE loss.

## Reference

[1] [Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)

[2] [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

[3] [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587)

[4] [Kaggle: A Data Competition Platform](www.kaggle.com)

