图马尔科夫神经网络 GMNN (Graph Markov Neural Networks)

本仓库是改论文的pytorch实现。

## 介绍

对于半监督对象分类任务，GMNN集成了**统计关系学习方法**(如关系马尔科夫网络和马尔科夫逻辑网络)和**图神经网络**(如图卷积网络和图注意网络)。

GMNN使用条件随机场定义所有对象标签的联合分布，使用的**伪搜索算法**对框架进行优化，它通过E-step和M-step实现。

- 在E-step中，我们**推断**未标记对象的标签

- 在M-step中，**学习**参数，最大限度地提高伪概率。

为了便于训练该模型，我们在GMNN中引入了两个图神经网络，即GNNp和GNNq。

- GNNq用来提高推理能力，通过学习对象的有效表示**特征传播**。

- GNNp对局部标签依赖关系进行建模。

GMNN还可以应用于许多其他应用，如无监督节点表示学习和链接分类。在本报告中，我们提供了**半监督对象分类**和**无监督节点表示学习**的代码。

## 数据
对于半监督分类任务，我们提供了Cora、Citeseer和Pubmed数据集。

对于无监督节点表示学习，我们提供了Cora和Citeseer数据集。 

- 均可cpu运行。运行结果见.out

数据集由[Yang et al.， 2016](https://arxiv.org/abs/1603.08861)构建，我们使用来自Thomas N. Kipf的[code](https://github.com/tkipf/gcn)将数据集预处理成我们的格式。用户还可以按照提供的数据集的格式使用自己的数据集。

## 使用
半监督对象分类的代码在目录`semisupervised`


要运行这些代码，请转到文件夹`semisupervised/codes`，执行`python run_cora.py`

然后程序将运行100次并打印结果。



---

# GMNN
This is an implementation of the [GMNN (Graph Markov Neural Networks)](https://arxiv.org/abs/1905.06214) model.

Table of Contents
=================
<!--ts-->
* [Introduction](#introduction)
* [Illustration](#illustration)
     * [Semi-supervised Object Classification](#Semi-supervised-Object-Classification)
     * [Two Graph Neural Networks](#Two-Graph-Neural-Networks)
     * [Optimization](#optimization)
* [Data](#data)
* [Usage](#usage)
* [Further Improvement](#further-improvement)
* [Acknowledgement](#acknowledgement)
* [Citation](#citation)
<!--te-->

## Introduction
GMNN integrates **statistical relational learning methods** (e.g., relational Markov networks and Markov logic networks) and **graph neural networks** (e.g., graph convolutional networks and graph attention networks) for semi-supervised object classification. GMNN uses a conditional random field to define the joint distribution of all the object labels conditioned on object features, and the framework can be optimized with a **pseudolikelihood variational EM algorithm**, which alternates between an E-step and M-step. In the E-step, we **infer** the labels of unlabeled objects, and in the M-step, we **learn** the parameters to maximize the pseudolikelihood.

To benefit training such a model, we introduce two graph neural networks in GMNN, i.e., GNNp and GNNq. GNNq is used to improve inference by learning effective object representations through **feature propagation**. GNNp is used to model local label dependency through local **label propagation**. The variational EM algorithm for optimizing GMNN is similar to the **co-training** framework. In the E-step, GNNp annotates unlabeled objects for updating GNNq, and in the M-step, GNNq annotates unlabeled objects for optimizing GNNp.

GMNN can also be applied to many other applications, such as unsupervised node representation learning and link classification. In this repo, we provide codes for both **semi-supervised object classification** and **unsupervised node representation learning**.

## Illustration
### Semi-supervised Object Classification
We focus on the problem of semi-supervised object classification. Given some labeled objects in a graph, we aim at classifying the unlabeled objects.
<p align="left"><img width="50%" src="figures/problem.png"/></p>

### Two Graph Neural Networks
GMNN uses two graph neural networks, one for learning object representations through feature propagation to improve inference, and the other one for modeling local label dependency through label propagation.
<p align="left"><img width="50%" src="figures/component.png"/></p>

### Optimization
Both GNNs are optimized with the variational EM algorithm, which is similar to the co-training framework.

#### E-Step
<p align="left"><img width="50%" src="figures/e-step.png"/></p>

#### M-Step
<p align="left"><img width="50%" src="figures/m-step.png"/></p>

## Data
For semi-supervised object classification, we provide the Cora, Citeseer and Pubmed datasets. For unsupervised node representation learning, we provide the Cora and Citeseer datasets. The datasets are constructed by [Yang et al., 2016](https://arxiv.org/abs/1603.08861), and we preprocess the datasets into our format by using the [codes](https://github.com/tkipf/gcn) from Thomas N. Kipf. Users can also use their own datasets by following the format of the provided datasets.

## Usage
The codes for semi-supervised object classification can be found in the folder ```semisupervised```. The implementation corresponds to the variant ```GMNN W/o Attr. in p``` in the Table 2 of the original paper. To run the codes, go to the folder ```semisupervised/codes``` and execute ```python run_cora.py```. Then the program will print the results over 100 runs with seeds 1~100.

The mean accuracy and standard deviation are summarized in the following tables:

| Dataset | Cora | Citeseer | Pubmed |
| --------  |----------|----------|----------| 
| GMNN | 83.4 (0.8) | 73.0 (0.8) | 81.3 (0.5) |

The codes for unsupervised node representation learning are in the folder ```unsupervised```. The implementation corresponds to the variant ```GMNN With q and p``` in the Table 3 of the original paper.  To run the codes, go to the folder ```unsupervised/codes``` and execute ```python run_cora.py```. Then the program will print the results over 50 runs.

The mean accuracy and standard deviation are summarized in the following tables:

| Dataset | Cora | Citeseer |
| --------  |----------|----------|
| GMNN | 82.6 (0.5) | 71.4 (0.5) |

Note that the numbers are slightly different from those in the paper, since we make some changes to the codes before release. In addition, the above experiment was conducted with ```PyTorch 0.4.1```, and the results might be slightly different if different versions of PyTorch are used.

## Further Improvement
The results reported in the previous section are not carefully tuned, and there is still a lot of room for further improvement. For example, by slightly tuning the model, the results on semi-supervised object classification can easily reach ```83.675 (Cora)```,  ```73.576 (Citeseer)```, ```81.922 (Pubmed)```, as reported in the appendix of the paper. Some potential ways for further improving the results include:

1. Train the model for longer iterations.

2. Use more complicated architectures for GNNp and GNNq.

3. Use different learning rate and number of training epochs for GNNp and GNNq.

4. Draw more samples to approximate the expectation terms in objective functions.

5. Integrate GNNp and GNNq for final prediction.

6. Adjust the annealing temperature when using GNNp to annotate unlabeled objects.

7. Use more effective strategies for early stopping in training.

8. Tune the weight of the unsupervised objective function for training GNNq.

## Acknowledgement
Some codes of the project are from the following repo: [pygcn](https://github.com/tkipf/pygcn).

## Citation
Please consider citing the following paper if you find our codes helpful. Thank you!
```
@inproceedings{qu2019gmnn,
title={GMNN: Graph Markov Neural Networks},
author={Qu, Meng and Bengio, Yoshua and Tang, Jian},
booktitle={International Conference on Machine Learning},
pages={5241--5250},
year={2019}
}
```


