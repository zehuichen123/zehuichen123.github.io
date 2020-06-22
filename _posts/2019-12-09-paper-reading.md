---
layout: default
title: Object Detection Paper Reading
date: 2019-12-09 15:27 +0800
---

目录：

1. Bridging the Gap between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection
2. Shape-aware Feature Extraction for Instance Segmentation
3. Region Proposal by Guided Anchoring
4. Feature Selective Anchor-Free Module for Single-Shot Object Detection
5. RDSNet: A New Deep Architecture for Reciprocal Object Detection and Instance Segmentation
6. Revisiting Feature Alignment for One-stage Object Detection
7. RepPoints: Point Set Representation for Object Detection
8. Is Sampling Heuristics Necessary in Training Deep Object Detectors?
9. Multiple Anchor Learning for Visual Object Detection
10. Learning from Noisy Anchors for One-Stage Object Detection
11. Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving
12. Bounding Box Regression with Uncertainty for Accurate Object Detection
13. HAMBox: Delving into Online High-quality Anchors Mining for Detecting Outer Faces
14. Revisiting the Sibling Head in Object Detector
15. Rethinking Classifcation and Localization for Object Detection

## Bridging the Gap between Anchor-based and Anchor-free Detection via Adaptive Training Sample Selection

这篇文章主要argue的一个点就是其实anchor-based和anchor-free本质上是没啥区别的，当然有一个前提条件，就是how to define positive and negative training samples。

这篇文章给了一个实验，也就是RetinaNet和FCOS的先导试验，就是它先把FCOS上的东西基本上都加到RetinaNet上，结果从32.5涨到了37.0。然后现在的RetinaNet和FCOS只有两个区别，一个是如何生成训练样本，一个是是否predefine anchor。然后当作者把RetinaNet的sampling methods从IoU-based改成Spatial and scale constraints后，FCOS和RetinaNet的表现结果一致了。这就表明其实有没有anchor是不重要的，你从点开始回归还是从框回归，最后并不影响。我觉得这个点发现的挺有意思的。

然后就是作者提出的ATSS了，具体想法就是先找到所有level的possible candidates的positive，然后计算他们与GT的IOU，接着计算mean和var得到对应IoU threshold。最后select对应满足条件的anchor。结果还是挺感人的，retina涨了2个点左右，FCOS涨了一个点多。

最后discussion的时候作者还提到一点，就是每个点有几个anchor在用ATSS的时候并不重要，但是在用IoU的时候还是很重要的。至于原因我觉得可能是IoU定义sample的时候真的就只有IoU作为标准，所以此时多几个准的框会好很多，但是ATSS的时候，是用center来定义的，所以对于同样的center，你出几个anchor结果都是一样的。

## Shape-aware Feature Extraction for Instance Segmentation

个人感觉出发点很好的一篇文章，可惜做法实在太水了。。。

其实本质上算是 segmentation-based ROIAlign，因为很多时候我们align feature的时候，因为物体畸形很多地方的feature是无效甚至会 deteriorates后面的分类和regression效果，文中举了一只猫和一只狗在一起的例子，普通的ROI Align把他们两都分成了猫（还是狗）。所以作者觉得在ROIAlign的时候如果能够知道segmentation同时进行ROIAlign，效果肯定会变好（废话）。所以作者的做法真的就是把segmentation的prediction合并到ROIAlign时候的feature map上了... 真xx简单....当然因为我们第一次ROIAlign的时候是没有segmentation的，所以作者说那我们可以像Cascade一样再来一个refine module，把第一次生成的segmentation接给第二次reg和seg的feature map，然后万事大吉。最后结果提了1.5个点左右吧，然后 casscade本来就会再涨0.5个点，所以最后也就1个点...

但是讲道理，那个intuition还是挺好的...

## Region Proposal by Guided Anchoring

之前看过一篇Cascade RPN，是将cascade的思想从RCNN放到了RPN中，通过两次iterative的预测RPN的regression，从而得到更好的proposal。这篇GA-RPN实际上是比Cascade RPN更早的一篇文章，主要思想其实也是十分直观的的，首先预测出可能出anchor的点，也就是物体的中心，这部分实际上跟FCOS的centerness有点相似，但是使用这个centerness的方法不一样吧...然后再根据可能出anchor的点预测它的h和w。当然直接预测形状实际上是不合理的，因为他们用的都是同一个CNN得到的feature，而这个时候每个点的感受野都是一样的，为啥能够预测出来不同的anchor的shape呢？所以作者提出了一个**Feature Adaption**模块，用来根据得到的anchor shape接一个1x1的conv得到offset然后重新过一遍deformable的卷积，这使得得到的proposal考虑到了anchor的shape，也就是feature和anchor的consistancy（feature的形状要和anchor的形状一致）。这里陈恺dalao提到了设计anchor的两条准则，一是alignment，即anchor的中心应该与feature的中心对齐，其次是consistency，即anchor的shape与feature的shape也要对齐。其实这点一直是one stage算法的问题，即one stage一直会存在misalign的问题，因为每个点出的proposal必然会离开生成的proposal，从而降低了生成的proposal的质量。而FCOS之所以使用centerness也是因为只在centerness出anchor避免了feature misalignment的问题。

## Feature Selective Anchor-Free Module for Single-Shot Object Detection

首席在一次paper reading让我看的一篇文章，打算最近把anchor free的文章都看一遍，看看能不能对这个方向有点自己的见解。很久很久之前在上Ng的课的时候他就说one-stage会是未来，那个时候我也不懂object detection是个啥，现在想想one-stage的确有超过two-stage的势头，从FCOS到之前那篇ATSS，现在大家都开始在思考two stage的这个宝贝anchor到底有没有必要。FCOS的centerness的确是个好东西，我觉得它从另外一个角度避免了feature misalignment的问题，而并非像two stage通过ROI Align来解决。但是它的感受野依然是一个局限，因为FCOS每个点的感受野，或者说feature的大小都一样，那么为啥有的点可以出大proposal，有的点可以出小propsoal呢？这个就是之前GA-RPN中提到的consisitency的问题。突然想到TridentNet里面用不同的dialated ratio来解决不同size的object检测问题。那么能不能在one-stage中也糅合Tridentnet的思想呢？话说TridentNet还是我刚来的时候看的，待会重新去看看，看看能不能有啥想法。

好了，上面都是废话，现在说一下这篇文章。其实就是点出框，然后回归点到边框的距离。但是实际上这个做法是有挺多问题的，首先是有效范围，FSAF中没有centerness来指导框的回归和分类，所以作者自己定义了只在0.2原始框大小的范围内进行梯度回传，这其实就是在一定程度上避免回归出较远的框。其次由于丢失了anchor这个东西，所以对于instance分配在FPN的哪一个level上就是一个问题，原来大家都是用IoU来进行instance分配到哪个level上的，现在没有了anchor自然就没有办法计算IoU了，那怎么分配呢？作者提出了在FPN上用loss来决定。也即如果容易在那个level回归的话，那么loss肯定比其他level小，那么我们就回传loss比较小的那个分支的loss。**不过有一个问题我一直没有搞清楚，就是对于不同level的回归target肯定不一样啊，高level的target肯定要比低level的target要小？那么loss相对来说应该从一开始就低一点？那这样不是所有的都跑去回归高level的target了？还是说都是project到原来size去算loss的？其实我觉得这个选择与初始化有关，应该一开始大家都train，train到后面选择loss最小的那个分支继续训练～** 所以感觉应该还是理解有所偏差，所以之后估计还要自己看看或者问问别人...

## RDSNet: A New Deep Architecture for Reciprocal Object Detection and Instance Segmentation

也是出发点很reasonable的一篇文章，考虑的是bbox的regression其实是不合理的，用l1回归这个框不是最直接的方式，因为框本身是根据mask定义出来的，所以reg和cls分支应该结合起来。但是之前像cascade/HTC虽然探索了这部分的问题，但是计算量相当大，本文则是以一种相对较小的开销达到了比较好的效果。

其实结构还是挺好懂的，但是那个metric learning的图画的我看懵了，重读了好几遍才算理清楚...理清楚后发现想法真的很直接，如果你有这个intuition，你也会选择这样做...

首先作者将FPN的object stream和pixel stream给分开了，两边单独管理object和mask。那么这两者如何交流呢？就有了第一个Instance-agnositc to Instance-aware module。主要就是，object分支还多出一个2k x d的representation。干嘛呢？用来做卷积核，来卷pixel stream的图，对于每一个object都卷一次，这样就能train出每一个object的mask了。所以作者认为这是把object level的信息给了mask...emmm 勉强吧 谁知道你object多出的是个啥...然后再根据object level出的框来裁剪我们的mask。为啥要裁呢？不然你train的时候那么大一片区域结果只有那么一小块有效面积，就很难被train好，所以作者就想用bbox出的框把想要的框给裁出来单独训练。但是你一开始的框可能不那么准，所以我们多裁不如少裁，顺便还能带点noisy增加分割的鲁棒性。。。但是我觉得这步真的有点挫...名字起的倒挺好听的，叫from translation-invariant to translation-variant。然后最后一步是通过已有的mask获得新的bbox，即Mask assisted Object Detection模块。然后作者发现直接根据mask来重新裁出bbox效果并不好，尤其是在小物体上表现很差。所以作者把这个问题转化为用贝叶斯来搞，即

$$ P(X=i|M') = \frac{P(X=i)P(M'|X=i)}{\sum_{t=1}^{w}P(X=t)P(M'|X=t)} $$

这里的$M$就是之前的mask logits，然后作者用一个discrete Gaussian distribution来近似 $ P(X=i)$ , 然后用一个2s+1的卷积核来预测  $ P(m^x_{i-s,...i+s}\vert X=i) $ ，从而保证端到端的训练。

最后的结果也还可以，因为本身这个问题我估计就不是太大。但是我觉得做的还是不够优雅，虽然的确做到了我们想要做的东西...不过我觉得应该还能找到更好的方法来做这个问题，以至于我一直觉得AdaptIS是非常优雅的，可惜单纯的segmentation想要达到好的instance level还是有点难呀~

## Revisiting Feature Alignment for One-stage Object Detection

## && 

## RepPoints: Point Set Representation for Object Detection

两篇放在一起写吧，都是今天看的。之所以一起写是因为两篇的实现非常相似，虽然出发点不同，但是实际上最后结构都差不多，除了一个用点一个用的box，但是最后都变成了使用deformable conv。不过还是觉得知乎上RepPoint也太不友好了...毕竟我觉得AlignDet里的insight Repoint文章提都没提，一口就是一个你是我baseline比的方法，但是实际上不是结构问题，个人觉得AlignDet是从One-stage出发考虑feature alignment的问题，而Reppoints明显是在表达点比bbox好...好了个人吐槽完毕，总的说一下两篇文章。

在说AlignDet之前先讲一下之前提到的Misalignment的问题，其实object detection两个问题，一个是中心点的misalign,一个是shape的misalign。中心点的misalign我觉得centerness应该解决的差不多了，不然FCOS也不会涨那么多..第二个点就是shape的misalign。什么事shape的misalign呢？就是说大物体和小物体的感受野不应该是一样的，这也是one-stage和two-stage的一个区别之一。AlignDet在文章中做了一个实验就是去掉FPN后RetinaNet疯狂掉点但是FasterRCNN只掉了1，2个点。这也体现出One-stage Detector的确需要不同的感受野。为了克服这个问题，AlignDet提出可以先出一个offset来align shape，通过Deformable Conv来进行所谓的ROIAlign，之后就是one-stage的标准步骤了。那么RepPoint呢？主要argue的点在于bbox不是最自然表现物体的方式，虽然它是最简单的，所以他们提出要用点来表示物体。怎么做呢？还是学一个offset，用9个点来表示一个物体，同时把物体的中心作为该物体的预测点。但是很巧合的是，RepPoint也是先出一次Offset之后Align再过one-stage，跟AlignDet里面的做法完全一样，所以RepPoint中的baseline（即跟bbox比）是AlignDet也的确没错啦～但是怎么说呢，我觉得他们两的确是两个出发点，但是最后得到了同样的结构...写到最后，我自己也觉得，好像没啥意思...hhh

## Is Sampling Heuristics Necessary in Training Deep Object Detectors?

文章读起来给人一种 “啥？这就写完了？”的感觉。简单来说就是感觉，的确切入点很新颖，however，提出的方法让我觉得有些简陋哈哈哈。不过的确work了，这点是无法否认的，当初kaiming不也是改了initlization的方式让训练work的嘛，可能我只是不大适应叭。

首先作者在RetinaNet上做了一个小实验，就是去掉focal loss是否能让模型训练起来。of course not! 那么有没有啥办法呢？首先需要降低cls loss的权重，不然会出现gradient exploding。其次，作者还观察到一个点，就是background loss会突然drop，因为网络发现如果给background全预测0的时候，loss最低，所以网络会直接优化过去。怎么解决这个问题呢？作者提出通过修改initlization最后一层bias的方式。如果提高最后一层网络的bias，那么最后网络的输出就会在0.01附近，这样就可以减少网络对负样本的attention，即$ b = - log \frac{1 - \pi}{\pi}$， 取$\pi = 1e^{-5}$时取得Retina的最优mAP35.6。然后作者又提到，他发现现在整个网络预测的cls score普遍低于之前的score，所以作者降低了min_det_score到0.005，效果提升至36.4。至此，一个sampling free的object detection model训练完成。也就是说，sampling free主要靠换bias。不知道正常imbalance problem有没有人试过这种方法...我觉得是有点神奇。至于作者后面提出的几个mechanism：optimal bias initialization、guided loss、class adaptive threshold都没有给我耳目一新的感觉，尤其是后两个哈哈哈，我感觉有点拍脑袋想出来的方法。为啥呢，因为后两个也没能解释为啥这样设置，那我就算你超参叭...最后能够在cascade RCNN上提1个点，说明的确是有效的。

这篇文章的确发掘出了我们平时没有注意到的问题，OHEM、focal loss的确很有效，但是没有他们真的就train不了了吗？看起来也不是这样。这也算是在imbalance problem in object detection的路上挺有意思的一次探索吧！

## Multiple Anchor Learning for Visual Object Detection

一篇把A放到B上work的文章，而且work的很好...比FreeAnchor还高1个点。那么它究竟是怎么work的呢？

在开始之前我们需要了解什么是Multiple Instance Learning。它其实也是一种监督学习，但是和分类不同的是它的分类是不是per sample给一个label，而是一个bag给一个label。举一个🌰来说，我们要判别，一个视频中有没有出现篮球，那么如果这个视频由一万张图组成，那么如果有一张图中有出现篮球，那么这个篮球就有篮球，反之，如果1w张图中都没有出现篮球，那我们就说这个视频中没有出现篮球。所以我们训练数据的时候拿到的是1个视频和1个label（这个视频中有没有篮球）。

那么这篇文章是如何把MIL用到object detection中的呢？

作者首先提出现在的模型大多都没有将cls score和localization联系起来的，即cls score并不表达regression的回归效果，但是实际上我们希望能够表达，因为这样我们的NMS需要根据cls score来remove冗余的预测框。那么这就会存在可能回归出来的框很好，但是cls score较低，或者cls score很高，但是回归出来的框不是很理想，这些情况都对最后的NMS造成了负面影响。解决这个问题存在两种方法，第一种是给更好的score，也就是IoUNet想做的事情，第二种也就是作者想做的，用更适合的anchor来训练网络。当然这个说法就是FreeAnchor（作者解释是因为MLE不适合用来解非凸问题，所以FreeAnchor也不好）。那么怎么才是好的呢？作者提出了学多个anchor，每次选出top-k score的anchor用来回归，也就是从anchor bag中选若干个anchor进行训练，这样就在优化的过程中保证了给出的anchor是高score高reg，不然就不会被选到。但是这就存在一个问题，如果我每次都训练多个anchor回归一个物体，那么最后inference呢？选谁好？所以作者提出了一个selection depression optimization，是啥呢？通俗的说就是一开始会一个bag里有多个anchor，随着iter增加逐渐减少anchor，直到最后只剩一个。当然对于anchor的选择，作者还提出了另外一个方法，也就是anchor depression，好像是对anchor进行重排？这里没有看太懂...总的来说就是把MIL应用到object detection里的文章，长点还是有的，比RetinaNet涨了5个点。。。还是有点猛嗷。。。至于为啥涨，我觉得是因为第一MIL本身就work，第二给的anchor多了学习的能力增强也会带动AP上涨吧，总是看完之后没有太大感觉，倒是觉得之前自己一直想要把cls score和location解藕，我觉得如果真能把这两个东西搞成线性相关，倒是没有啥节藕的必要了吧，那应该也会接着涨点。

同时刚才想到一个点，就是其实神经网络判别物体类别靠的应该是某些关键点，那么roi如果一开始坐落在某些关键点上，那么给出的cls score可能会很大，但是实际上它reg不一定准，所以可能会导致cls score和localization的mismatch。如果我们能够一定程度上解决这个问题，那应该也能涨点。

## Learning from Noisy Anchors for One-Stage Object Detection

这篇文章读了两遍，第一遍是在圣诞节看的，只是粗略的扫了一遍，因为之前的一些实验突然想起来这篇文章，跟zz讨论了一下label assignment的问题，就想起来这篇文章。今天跑模型的时候就抽空又把这篇文章重新读了一遍。其实很简单的一篇文章，但是形式的确挺优雅。

出发点就是对于卡IoU来给anchor assign 1/0的方法实际上是不够好的，因为这样会引入noisy，因为明显GT IoU0.55和 GT IoU0.95的anchor不应该是一种anchor。那如何降低这种noisy呢？两种选择：第一种就是不用卡IoU的方式来定义anchor，第二种就是reweight学习的权重，对于有noisy的anchor就少点权重。

那么怎么从这两点来解决noisy的问题呢？这篇文章首先提出了一个cleanliness score，首先它应该能够作为一种表征回归的难易或者这个anchor好坏的东西，其次应该能够作为一种reweight的参考。综合考虑这两点，作者把classification和localization两部分结合生成了cleanliness score:

$$ c = \alpha \cdot \text{loc_a} + (1 - \alpha) \cdot \text{cls_c} $$

其中 loc_a 为 location accuracy, 即为与GT的IoU, cls_c为cls分支的score。这里作者提到对于每个GT只取它附近的top k计算cleanliness score。也就是其他地方的anchor的cleanliness都取0。

有了cleanliness score怎么使用呢？首先把它作为一种新的label，而非0/1。那么对于RetinaNet的focal loss就可以改写为

$$ \text{BCE}(p, c) = -c \cdot w_p \cdot \text{log}(p) - (1-c)\cdot w_n \cdot \text{log}(1-p) $$

这样cleanliness score实际上被作为一种soft label进入了我们的loss计算中，这在某种程度上其实是算改变了定义label的方式。其次，还需要reweight 我们每个anchor，对于noisy anchor给的权重就小一点。这个reweight的coef呗定义为 

$$ r = (\alpha \cdot f(\text{loc_a}) + (1-\alpha) \cdot f(\text{cls_c}))^{\gamma} $$

然后用 $r$ 去penalize regression分支的smooth l1 loss。所以最后的形式就变成了

$$ L_{cls} = \sum^{A_{pos}}_{i}r_i BCE(p_i, c_i) + \sum_{j}^{A_{neg}}BCE(p_j, c_j) \\
L_{reg} = \sum_{i}^{A_{pos}} r_i \text{ smooth_}l_1 $$

最后作者还做了一个实验就是cls和loc是否存在很好的联系，即对于高cls score,它与GT的IoU是不是也比较高。这样就可以保证在NMS阶段能够得到更好的效果，似乎的确是有点效果的。

对于label assignment的问题，因为最近还在做Knowledge distillation的东西，所以其实teacher给出来的label是否也可以作为一种新的label定义方式呢？因为通过teacher给出的label实际上也是一种更适合学习的神经网络的对label的定义方式？

## Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving
## &&
## Bounding Box Regression with Uncertainty for Accurate Object Detection

两篇文章都是探究在localization阶段，也就是regression分支上如何输出网络回归的uncertainty进而作用于接下来的不论是NMS还是cocoeval scoring阶段。

两篇文章都是在Regression阶段输出一个分布而非一个确定的值。前者用了Negative Log Likelihood来优化，后者用来KL Loss来优化。

第一篇文章优化每个sample落在对应输出上的分布上的似然，优化函数为：

$$ L_x = - \sum^W_{i=1}\sum^H_{j=1}\sum^K_{k=1}\gamma_{ijk}\text{log}(N(x^G_{ijk}|\mu_{t_w}(x_{ijk}, \delta(x_{ijk}))+ \epsilon) $$

其中W、H为image grid，K为每个grid的anchor数目，只有当grid上的点与某一GT的重叠度最大时，$\gamma$才为1，其他时候为0。通过优化这个NLL从而让网络学到uncertainty。最后用法也很简单，乘到score上。

第二篇的思路也很直观，把GT也看作一个分布，只不过是一个var为0的Dirac delta分布。然后优化输出分布与GT分布的KL散度。

$$ L_{reg} = D_{KL}(P_D(x)|| P(x)) \\ = \frac{(x_g-x_e)^2}{2\sigma ^2} + \frac{\text{log}(\sigma^2)}{2} + \frac{\text{log}(2\pi)}{2} - H(P_D(x)) $$

由于$H(P_D(x))$是不参与求导的，所以与优化无关，最终损失函数正比于

$$ L_{reg} \propto \frac{(x_g - x_e)^2}{2\sigma^2} + \frac{1}{2}\text{log}(\sigma^2) $$

之后这篇文章里面还提到了Var Voting，其实算是一种集成框的形式，所以必涨点。不是我这里想提的内容，所以不讲。

那么如何衡量学到的var是否真的有效呢？

由于我之前一直在后面这篇文章上做些实验，干脆放上来分享一下。

<center><img src="/images/paper2020_1/res152_proposal.png" height="600"></center>
这张图是每条边预测的var与对应边和GT的差值的图，这里只取IoU大于0.5的作图，不然几乎看不到线性关系= = 由于我们推出来的var实际上是在学GT与预测值差值的绝对值，所以我们期望最后学出来的diff与var应该是符合线性关系，也就是应该比较均匀的分布在红线周围，但实际上模型学出来的var overestimate了，也就是说，即使diff很小，网络依然可能会预测出较大的var，这也是为啥只取了大于0.5的作图，不然scatter的点可能会覆盖整个x轴...

<center><img src="/images/paper2020_1/res152_iou_proposal.png" height="500"></center>
这张图是我今天看到Gaussian YoloV3那篇文章里面画的图所以想画张一样的看看效果。的确惨目忍睹啊...在低IoU的框中，网络预测出的var打了个折，那应该是咋样呢？看下别人Gaussian YoloV3里面的图就知道了...不过后来想了一下，值得注意的是，如果IoU都已经低于0.5了，那么我们还应该把这种prediction看成是与该GT匹配的框吗？同时对于IoU小于0.5的框我们是不train的，所以出现这种情况也是合理的吧

<center><img src="/images/paper2020_1/gaussianyolo.png" height="400"></center>
感觉KL Loss学的还是不够好啊...最近在考虑要不要试一波他这个NLL，看起来还是比KL强点...

## HAMBox: Delving into Online High-quality Anchors Mining for Detecting Outer Faces

话说好久没怎么写reading notes，不过最近看的paper感觉都没有看到啥比较有insight的paper，倒是这篇HAMBox挺有意思的，准备写一下。

首先这篇文章是关于anchor label assign的，哈哈哈，是不是想到了之前看的那篇Learning from noisy labels for object detection？不过这两篇的确有着相似之处，比如说都是用anchor与GT的IoU来加权Regression分支。这点其实我之前也试过，Faster R-CNN用proposal与GT的iou加权RCNN的回归分支，好像能够在COCO涨个0.2～0.3个点左右，不过这个涨幅感觉不够大啊，倒是Davis组的那篇同时考虑了cls score与IoU融合的加权好像能涨更多。

其实这篇文章主要的贡献当然不是那个focal loss for regression，实际上是说的是label assign的问题。在之前那篇文章中，作者提出用cleanliness score来加权classification分支的训练，提出了一种类似于focal loss的损失函数。这篇文章并非直接用loss入手，而是从给的数据集样本分布入手，人为加入anchor（或者更准确的说动态更新IoU阈值assign anchor的label，说到这又让我想到了ATSS 哈哈哈，ATSS是使用anchor计算分布来动态assign。啊哈，突然感觉这三篇文章都在研究一个问题！准备改天写个仔细点的文章分析一下这三个东西的共同点和区别，ATSS都是在去年12月看的了....）

当然，这篇文章有几个验证实验，最有说服力的我觉得是下面两幅图
<center><img src="/images/paper2020_3/ham_iou.png" width="300"></center>
<center><img src="/images/paper2020_3/ham_anchor.png" width="300"></center>
首先解释一下high quality anchors的意思是指最后通过回归结果与GT 的IoU大于0.5的anchor。那么图一的意思就是anchor小于IoU多少时，能够产生的正确检测框比例。从图中可以看到那些IoU小于0.35的anchor检测框们能够在最后产生占到89%的正确检测框。当然这张图是inference的时候，而在train的时候，这个比例也是占到了65%（训练结束时差不多70%）。可以看到实际上我们很多的正确检测框来自于我们被设置为负标签的样本。那么这么做会不会又影响呢？当然会咯，那么我们就可以看第二张图，这张图画的是真的难看懂...😒 首先解释几个词语：CPBB是corrected predicted bounding box，横坐标是每张脸（GT）match到的anchor的数目，纵坐标是被match到对应anchor数目的人脸（GT）有多少个。所以红色可以解释为，被match到对应anchor数目的脸有多少张，蓝色可以解释为被match到对应CPBB数目的人脸又多少张，绿色就是CPBB NMS之后的。那么可以发现每个人脸被assign到的anchor越多，结果也就越好，对应的蓝色就越多。然后重点来了，在NMS之后，被match到对应CPBB的人脸数目都显著下降了！为啥？如果我们NMS掉的是同样的框，那么绿色应该不会比蓝色低多少才对。真相只有一个，那就是其实我们NMS掉了之前找出来的框，但是由于分太低被remove掉了，所以就出现了大量漏检。所以说我们assign负标签给这些本来正确的样本还是有影响的哇！

其实这个点发现很有意思，解决方法也不难，就是在train的过程中，动态计算每个gt周围的anchor数目label，给对应的high quality anchor重新assign label。做法的话懒得写了，其实也是根据对应阈值对每个GT找top-k个positive anchor（前提是有），这个positive anchor的定义是从最后回归的结果定义的，而非通过一开始的IoU predefined。这样就能够避免之前说到的high quality anchor被分类为negative samples的问题啦～

不过说到这里，这篇文章一直说的是one stage，那么two stage是否存在这个问题呢？那RPN是不是也应该来一波这个说不定能够比Cascade RPN涨更多呢哈哈哈哈

## Revisiting the Sibling Head in Object Detector(TSD)

## &&

## Rethinking Classification and Localization for Object Detection(Double Head)

这两篇说的都是一件事情 但是做法略微有些区别 有些殊途同归的味道

首先出发点都是 回归和分类是两件事情 这两件事情需要解耦 为什么呢 因为分类注重的是 图片中的某些具有discriminative的feature，或者说全局的信息，而回归更注重边界信息，即我需要回归到哪个位置。

好啦，出发点说完了，Double Head的做法可以概括为 分类two fc, 回归4conv。为什么这么说呢，还是从出发点来的，分类需要全局信息，所以用fc，它flatten了；回归需要局部信息，用conv这种局部的op。

TSD的做法就比较麻烦一点，首先它有一个shared proposal，就是正常的proposal，然后这个proposal会通过不同的操作提供一个regression用的proposal和一个classfication 用的proposal。这里引入两个概念，pixel-wise offset和proposal-wise offset。pixel wise offset有点像deformable，即proposal上每个点都会有一个offset，这个offset加上原来的坐标是这个点最后取到的特征。而proposal wise offset其实就是所有点都共享一个offset，相当给定proposal xyxy去align feature的时候 把xyxy都加一个offset。然后TSD对regression head用proposal wise的offset生成新proposal，对classification head用pixel wise的offset生成proposal，然后对应head过FC就完事啦～

当然除此之外，两者都提到如果保留原有的sibling head也会给结果带来一定程度上的提高。不过两个用的方式也存在区别，Double Head是把这两个head上的做ensemble，比较憨批...TSD就比较优雅，提出了Progressive constraint，有点像是consistency的东西，可能为了防止offset回归的太过火了叭，所以从loss function上限制了offset，即保证TSD Head与原有head的IoU或者cls score不能差过一定margin，不然就惩罚这个margin...不过涨点也是有的，所以也不是为了凑数搞出来的...



