# 开源神经网络库

1. 简介

    在本教程中，我们将回顾一些开源[神经网络库](https://www.baeldung.com/cs/genetic-algorithms-vs-neural-networks)。

    神经网络库通常用于在计算机程序中实现神经网络。多年来，这些库中的许多库都得到了开发和增强，使神经网络的处理功能更容易实现和利用。

2. TensorFlow

    [TensorFlow](https://www.baeldung.com/tensorflow-java#overview) 由谷歌机器学习研究小组 Google Brain 团队于 2015 年首次开发。它旨在促进构建机器学习和神经网络模型的研究。因此，TensorFlow 提供了简单的模型构建和强大的机器学习模型部署环境。

    它还拥有一个初学者和专家都能使用的平台，可在 Windows、Linux、Android 和 macOS 平台上使用。

    使用 TensorFlow 的公司包括谷歌、DeepMind、AirBnB、可口可乐和英特尔。

3. CNTK

    [CNTK](https://www.baeldung.com/spark-mlib-machine-learning#3-cntk) 又称微软认知工具包（Microsoft Cognitive Toolkit）。CNTK 于 2016 年首次发布，现已弃用，是一个用于训练[深度学习](https://www.baeldung.com/deeplearning4j#what-is-deep-learning)神经网络的开源库。

    它允许用户创建和组合常用的神经网络，如卷积神经网络（CNN）和循环神经网络（RNN）。

    CNTK 支持 Linux、Windows 和 macOS 平台。

4. PyTorch

    PyTorch 发布于 2016 年，由 Facebook 的人工智能研究实验室开发。PyTorch 构建在 Torch 之上，Torch 是一个机器学习和科学计算库。根据 PyTorch 开发人员的说法，这是一个机器学习框架，可以加速从研究原型到生产部署的过程。

    PyTorch 具有神经网络模型分布式训练、云支持、强大的生态系统等特点，支持 Windows、Linux 和 macOS 平台。斯坦福大学和 Udacity 等机构都在使用它。

5. Theano

    [Theano](https://www.baeldung.com/spark-mlib-machine-learning#2-theano) 是蒙特利尔学习算法研究所（MILA）于 2007 年发布的一个开源库。Theano 支持评估机器学习常用的数学表达式。它提供一个 Python 接口，可与 Keras 配合使用。

    Theano 还支持 Linux、Windows 和 macOS 等操作系统。

6. Caffe

    [Caffe](https://caffe.berkeleyvision.org/) 是伯克利人工智能研究小组于 2013 年开发的深度学习框架。它旨在为构建神经网络提供速度和模块化。不过，Caffe 主要支持用于图像分类和图像分割的神经网络。

    目前可在 Linux、macOS 和 Windows 操作系统上运行。

7. Keras

    [Keras](https://www.baeldung.com/spark-mlib-machine-learning#1-tensorflowkeras) 由谷歌工程师 Francois Chollet 于 2015 年设计。它为用 Python 构建神经网络提供了友好的用户界面。Keras 具有激活函数和层等模块，只需几步即可实现神经网络，便于快速实验。

    此外，该库还可以在 TensorFlow 和 CNTK 等其他库之上运行。由于其简单而强大的界面，NASA 和 CERN 等机构都在使用它。

8. 深度学习 4J

    [DeepLearning4J](https://www.baeldung.com/deeplearning4j#introduction) 是为 Java 和 Scala 编写的深度学习库，最初于 2014 年发布。它具有分布式计算训练环境，可以加速性能。DeepLearning4J 允许用户灵活地组成和组合神经网络模型。

    DeepLearning4J 目前支持 Windows、Linux 和 macOS 平台。此外，它还能与 TensorFlow 和 Keras 等其他神经网络库配合使用。

9. 总结

    现在，让我们来总结一下上述框架的优缺点：

    ![由 QuickLaTeX.com 渲染](pic/quicklatex.com-99b322c8a64d47ac14b6975c4b5a7e7c_l3.svg)

10. 结论

    在本文中，我们确定并回顾了一些常用的神经网络开源库。
