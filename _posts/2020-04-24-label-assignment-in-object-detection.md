---
layout: post
title: Label Assignment in Object Detection
date: 2020-04-24 18:32 +0800
---

在最初的Faster R-CNN做法中，我们是通过框与框之间的IoU来决定anchor的正负标签的，但是这种使用IoU一刀砍的方式来决定label未免有些粗鲁，所以后续出现各种花式的label assignment，就打算整理一下～

首先说说Fast R-CNN一类，通过计算anchor与GT之间的IoU，把IoU > fg_thres(0.7)作为正样本，IoU < bg_thres(0.3)作为负样本，IoU在bg_thres ~ fg_thres(0.3 ～ 0.7)之间的认为是ignore样本，看起来还挺合理～但是不免存在如下几种问题：

- 问题一：如果一个框在0.3以下那它就不能被回归到0.5以上了吗？如果它被回归到了我们还把它看作负样本，那是不是某种程度上在deteriorate模型的表现呢？
- 问题二：与GT IoU0.9的框 与GT IoU0.55的框，应该都给1吗？他们其实是有好坏之分的，但是在使用IoU一刀砍的环境下，没法体现出来这种优劣之分。
- 问题三：由于我们的anchor设置是predefine的，那么可能无法保证每一个GT都存在很好的anchor和它匹配，这样就会造成一个问题：每个GT与他周围的anchor的IoU的重叠分布是不同的，这样如果我们一刀切就会导致不同GT样本被分配到的anchor不均衡。虽然现在的实现里我们会保证找到和GT匹配度最高的anchor（无论它与该GT的IoU是多少）设为正样本，但是依然没有从本质上解决这个问题。

对于问题一，我目前看到了两篇文章其实想解决这个问题，一个是HAMBox，一个是Learning from noisy labels for one stage object detection。

### TopK

首先说一下TopK吧，是我在实习的时候学到的一种方法，据说是当年fpp拿冠军的时候分享的方法，即在assign anchor label的时候，对于每个GT，我们都找到topk的样本把他们当成正样本，这样做的好处也很明显：对于小物体GT，可能和它匹配的anchor很少，这个时候如果还根据IoU来找正样本的anchor，那么可能打物体被分配了很多anchor来训练，而小物体被assign到的anchor数就比较少（突然发现可以从data imbalance的角度来看这个问题hhh），这种方法可以看作通过一种dynamic的方式来动态改变IoU阈值来划分正负样本～同时我们保证了不同大小的物体都能够得到一定的anchor进行训练。topk方法可以看作是对问题一的解决方法。

### Learning from Noisy Anchor

那么问题二该怎么办呢？一个很直观的想法：那我能不能根据回归出来的结果来动态的决定我的anchor的质量好坏呢？如果我能给他回归的很好，那这个anchor的确很棒，如果我回归的不是很行，那就让这个anchor爬。Learning from noisy anchor for one stage object detection干的就是这么一件事情～这篇文章的出发点就是从问题一出发的，不过是从另外一个角度来思考问题一：就是我们看作是正样本的那些anchor，是不是都很棒呢？是不是也有不是很好的样本嘞？那么怎么决定一个anchor是不是noisy的呢？作者提出了可以从回归后和GT的IoU以及分类的score来看，于是乎，这两个值乘起来，就作为我评价anchor质量的评判标准啦（文中叫做cleanliness）～所以这个anchor的质量怎么反应到我的网络中呢？首先作者把它作为cls的label放到了focal loss里面，也就是不再用0/1作label，而是使用这个cleanliness作为label，同时它还参与到regression的部分，作为权重加权回归，也就是好的anchor我们就多回归，noisy的anchor我们就少回归～所以说这篇文章实际上是从学习的过程中动态的决定anchor的label，根据网络学习的结果来决定这个anchor是好的还是坏的。感觉还是挺合理哈～

8⃣️过，这篇文章其实都是把重点放在了刚开始就positive的anchor，而negative的anchor永世不可超生，也就是依然没法解决我们问题一中提到的问题。

### HAMBox

接着就可以说一下HAMBox了～这篇文章是在人脸检测上做的，文章里说在他们的model里，anchor是没有ignore区间的，低于0.35的就都是负样本，高于0.35的就都是正样本，不过我没有怎么读过人脸检测的文章，不知道这种做法是不是common practice，但是这样assign的话，我们之前提到的问题一就更加显著，即一旦有一个anchor低于0.3的IoU，它就是负样本，我分类的时候就要抑制它，这样可能会deteriorate模型的分类能力。

而HAMBox发现，最终回归出来的IoU大于0.5的检测框里，有很大的比例来自于这些刚开始的anchor与GT的IoU低于0.35的anchor。那岂不是错杀了一大批anchor？所以HAMBox提出了一种anchor补偿的策略，本质上是在训练的过程中，动态的修改那些被回归好的样本。所以HAMBox主要解决的就是我们说的问题一的问题，我会在训练的过程中，动态的把那些本身和GT重叠度不高但是最后回归的好的anchor设为正样本。具体的做法其实和topk有点类似，作者会在训练的过程中，对每个GT动态的补偿到k个anchor，也就是说，HAMBox不是像topk那样直接把所有topk都看成正样本（不过topk好像fpp也没有发paper，具体怎么做的我也不知道🤷‍♂️），而是根据回归的好坏把某些模型能够回归出来的anchor设为正样本。其实你可以看作 HAMBox的重点在于如何处理刚开始被assign为负的样本，而Learning from noisy anchor看作如何处理刚开始被assign为正的样本～

### ATSS

ATSS虽然同样在解决正负标签assign的问题，但是入手的角度是从anchor based和anchor free的区别出发。文章从RetinaNet和FCOS出发，把FCOS上的结构都在RetinaNet上加了一遍，同时对比了classification和regression分支两者的区别，最后发现RetinaNet和FCOS的区别在于：正负样本Assign的规则导致了最终点数上的差距。也就是label assignment的问题。（当然ATSS在一步一步分析这两者区别的过程也是这篇文章很棒的一部分，可以看到思路非常清晰的，不愧是shifeng大佬orz）至于ATSS的做法其实是从统计意义上来思考正负样本的定义。直观的理解就是，如果你把每个GT周围的anchor与它的IoU统计一下，其实是可以形成一个分布的，那么我通过取这个分布上的某个分位数来决定每个GT的IoU Thres岂不是很合理？其实如果再想想你可以发现ATSS也和topk有着异曲同工之妙，他们都是从每个GT出发，去argmax GT周围的的anchor，而不是 argmax proposal周围的GT。可能从GT出发来决定anchor的正负才是更为合理的方式？

写到这里差不多就结束了，最后回过头来看，目前的几种解决label assignment的方式可以归纳为两种：

1. 动态的根据回归后的结果决定anchor的好坏
2. 从每个GT出发assign label，而不是从anchor出发



[1] HAMBox: Delving into Online High-quality Anchors Mining for Detecting Outer Faces, Yang Liu, et.al

[2] Bridging the Gap Between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection, Shifeng Zhang, et.al

[3] Learning from Noisy Anchors for One-stage Object Detection, Hengduo Li, et.al

