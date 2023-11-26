# 如何计算线性回归中的正规化参数

[机器学习](https://www.baeldung.com/cs/category/ai/ml)

[回归](https://www.baeldung.com/cs/tag/regression) [训练](https://www.baeldung.com/cs/tag/training)

1. 简介

    在本教程中，我们将介绍被称为线性回归的经典机器算法。首先，我们将讨论回归一词和所有不同类型的回归。然后，我们将详细研究线性回归，以及为什么需要正则化。之后，我们将给出相应的数学公式和符号，并对其进行清晰的解释。

    毕竟，线性回归是最流行、最常用的统计和机器学习算法之一。

2. 回归

    一般来说，回归是我们用来衡量两个或多个变量之间关系的一系列统计算法。其主要逻辑是，我们希望找到输入变量（特征）（我们称之为自变量）与结果或因变量之间的联系。通过利用输入和输出变量之间的这种关系，回归被广泛用作预测或预报的机器学习模型。

    例如，我们可以使用 R 平方来衡量两个变量之间的关系，它告诉我们变量与拟合回归线的接近程度。我们还可以使用多项式回归根据历史数据预测未来价格，或使用逻辑回归建立二元分类的概率模型。

    1. 回归类型

        回归有很多种类型，这里我们只提及其中的几种。根据输入变量的数量，我们可以将回归分为两类：

        - 简单回归 - 仅使用一个变量作为输入变量
        - 多元回归 - 使用两个或多个变量作为输入变量

        同样，根据因变量或输出变量的数量，我们可以将回归分为以下几类：

        - 单变量回归 - 只有一个因变量
        - 多元回归 - 有两个或更多因变量

        最常用的三种回归模型是：

        - 线性回归 - 假设输入和输出变量之间存在线性关系，并使用线性函数来解释这种关系
        - 多项式回归 - 与线性回归类似，但使用多项式来近似变量之间的关系
        - 逻辑回归 - 使用逻辑函数或 sigmoid 为二元分类问题的概率建模

        最后，值得一提的是，"回归"一词在另一种语境中也被广泛使用，它是一类机器学习算法。通常，我们可以将大多数机器学习问题分为两类：

        - 分类问题 - 目标是预测一个预定义的标签或类别。例如，预测特定句子的情感是否积极，或预测图像上的手写数字。
        - 回归问题 - 目标是量化。例如，预测特斯拉股票明天的价格，或利用一些天气数据预测准确的温度。

3. 线性回归

    前面我们提到了一些最常用的回归技术，在本节中，我们将介绍可能是最流行的一种。线性回归是一种模拟一个因变量与一个或多个连续或离散解释变量之间关系的方法。这种技术有助于通过使用另一个变量来预测一个变量的未来值。通过使用两个变量过去的数据，我们可以预测一个变量对另一个变量的影响程度。

    我们用公式定义线性回归：

    (1) \[\begin{equation*} \hat{y} = \theta_{0} + \theta_{1}x_{1} + \theta_{2}x_{2} + ... + \theta_{n}x_{n}, \end{equation*}\]

    这里，$\hat{y}$ 是预测值，n 是特征个数，$x_{i}$ 是第 i 个特征值，$\theta_{j}$ 是第 j 个模型参数或权重。此外，$\theta_{0}$ 被称为偏差项。

    类似地，我们可以用向量化的形式来写上面的等式：

    (2) \[\begin{equation*} \hat{y} = \boldsymbol{\theta}^{T} \cdot \boldsymbol{x} \end{equation*}\]

    这里，$\boldsymbol{\theta}^{T}$ 是转置模型的权重向量，$\boldsymbol{\theta} = (\theta_{0}, \theta_{1}, ..., \theta_{n})$，$\boldsymbol{x} = (x_{0}, x_{1}, ..., x_{n})$ 是特征向量，其中 $x_{0} = 1$。

    接下来，为了训练模型，我们首先需要测量模型与训练数据的拟合程度。为此，我们通常使用均方误差 (MSE) 成本函数。我们用公式定义它：

    (3) \[\begin{equation*} MSE = \frac{1}{m} \sum_{i=1}^{m} (\boldsymbol{\theta}^{T} \cdot \boldsymbol{x}^{(i)} - y^{(i)})^{2} \end{equation*}\]

    这里，m 表示样本数，$y^{(i)}$ 是第 i 个样本的实际值。我们用 MSE 来衡量估计值与实际值之间的平均平方差。

    要找到使成本函数最小的 $\boldsymbol{\theta}$ 值，有三种方法：

    - 闭式求解
    - 梯度下降法
    - SVD 和 Moore-Penrose 伪逆变换法

    1. 闭式求解

        所谓闭式求解，是指使用正则方程只需一步就能找到最优值。因此，我们可以将正则方程定义为

        (4) \[\begin{equation*} \hat{\boldsymbol{\theta}} = (\boldsymbol{X}^{T} \cdot \boldsymbol{X})^{-1} \cdot \boldsymbol{X}^{T} \cdot \boldsymbol{y} \end{equation*}\]

        这里，$\hat{\boldsymbol{\theta}}$ 是 MSE 的最优解，$\boldsymbol{X}$ 是包含所有 $\boldsymbol{x}^{(i)}$ 特征向量的矩阵，$\boldsymbol{y}$ 是包含所有 $y^{(i)}$ 的目标值向量。

        由于我们需要计算$(\boldsymbol{X}^{T} \cdot \boldsymbol{X})$矩阵的逆矩阵，该矩阵是 n x n 矩阵，其中 n 是特征的个数，因此复杂度约为 $O(n^{3})$。因此，这种方法会随着特征数量的增加而变得非常缓慢。相比之下，复杂度与训练测试中的样本数呈线性关系。

        总之，当我们有大量特征或内存中无法容纳太多训练样本时，这种方法并不是最好的方法。

    2. 梯度下降法

        梯度下降是机器学习中的一种技术，我们通常在优化中使用它。我们使用这种算法，通过迭代找到相对于当前位置斜率最陡的方向，从而找到函数的最小值。在这种情况下，我们使用梯度下降法计算相对于当前位置的最陡斜率。

        具体来说，如果我们的 MSE 成本函数 $J(\boldsymbol{\theta})$ 的形式为：

        (5) \[\begin{equation*} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} ({\theta}^{T} \cdot x^{(i)} - y^{(i)})^{2} \end{equation*}\]

        那么梯度为：

        (6) \[\begin{equation*} \frac{\partial J(\theta)}{\partial \theta} = \frac{2}{m} \sum_{i=1}^{m}[ ({\theta}^{T} \cdot x^{(i)} - y^{(i)})x^{(i)}] \end{equation*}\]

        之后，我们用梯度乘以学习率 $\alpha$ 来更新权重：

        (7) \[\begin{equation*} \theta := \theta - \alpha \cdot \frac{\partial J(\theta)}{\partial \theta}. \end{equation*}\]

        在我们的案例中，这种方法会导致全局最小值，因为我们的代价函数是一个凸函数，因此它只有一个局部最小值，同时也是全局最小值。梯度下降法之所以盛行，是因为它广泛应用于包括神经网络在内的许多机器学习技术中。尽管如此，它也有一些缺点。

        由于梯度下降法是一种迭代方法，因此需要一定时间才能达到最优解。此外，收敛性还取决于学习率 $\alpha$。

        总之，梯度下降法或该方法的改进版在机器学习中得到了广泛应用，但在我们的案例中，还有一种更快的方法，我们将在下文中介绍。

    3. SVD 和摩尔-彭罗斯伪求逆法

        正则方程前一步：

        (8) \[\begin{equation*} \boldsymbol{\theta} = (\boldsymbol{X}^{T} \cdot \boldsymbol{X})^{-1} \cdot \boldsymbol{X}^{T} \cdot \boldsymbol{y} \end{equation*}\]

        我们得到这个等式：

        (9) \[\begin{equation*} \boldsymbol{X}^{T} \cdot \boldsymbol{X}\boldsymbol{\theta} = \boldsymbol{X}^{T} \cdot \boldsymbol{y}. \end{equation*}\]

        由于样本数量多于特征数量，这个等式几乎没有精确解，因此我们使用欧氏距离尽可能接近 \boldsymbol{theta}向量：

        (10) \[\begin{equation*} \min_{\theta} = ||\boldsymbol{X}\hat{\boldsymbol{\theta}} - \boldsymbol{y}||^{2}_{2}. \end{equation*}\]

        我们把这个问题称为普通最小二乘法（OLS）。有很多方法可以解决这个问题，其中之一就是使用摩尔-彭罗斯伪逆定理。具体来说，该定理指出，对于线性方程组 $Ax = b$，具有最小 $L2$ 准则 $||Ax - b||^{2}_{2}$ 的解是 $A^{\dagger}b$，其中 $A^{\dagger}$ 是摩尔-彭罗斯伪逆定理(Moore-Penrose pseudoinverse)。

        接下来，我们通过奇异值分解（SVD）计算 $A^{\dagger}$ 的值，或者在我们的例子中计算 $X^{\dagger}$：

        (11) \[\begin{equation*} X = V\Sigma U^{T} \end{equation*}\]

        这里，X 的格式为 n \times m，其中 n 是特征数，m 是样本数，$V_{n \times n}$ 是正交矩阵，$\Sigma_{n \times m}$ 是对角矩阵，$U_{m \times m}$ 是正交矩阵。因此，我们可以计算 Moore-Penrose 伪逆：

        (12) \[\begin{equation*} X^{\dagger} = V\Sigma^{\dagger} U^{T} \end{equation*}\]

        这里，$\Sigma^{\dagger}$是通过取非零元素的倒数值，并将得到的矩阵转置，从而从$\Sigma$中构建出来的。最后，我们使用公式计算线性回归权重：

        (13) \[\begin{equation*} \theta = V\Sigma^{\dagger} U^{T}y. \end{equation*}\]

        Scikit-learn 软件包默认将此方法用于线性回归优化，因为它速度快且数值稳定。

4. 线性回归中的正则化

    正则化是机器学习中一种试图实现模型泛化的技术。这意味着我们的模型不仅能很好地处理训练数据或测试数据，还能很好地处理未来的数据。总之，为了实现这一点，正则化将权重向零收缩，以阻止复杂模型的出现。因此，这可以避免过度拟合，并减少模型的方差。

    线性回归中有三种主要的正则化技术：

    - 拉索回归 Lasso Regression
    - 岭回归 Ridge Regression
    - 弹性网 Elastic Net Regression

    1. 拉索回归

        Lasso 回归或 L1 正则化是一种增加代价函数的惩罚技术，惩罚值等于线性回归非截距权重的绝对值之和。形式上，我们将 L1 正则化定义为

        (14) \[\begin{equation*} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} ({\theta}^{T} \cdot x^{(i)} - y^{(i)})^{2} + \lambda\sum_{j=1}^{n}|\theta_{j}| \end{equation*}\]

        这里，$\lambda$ 是正则化参数。基本上，$\lambda$ 控制着正则化的程度。尤其是，$\lambda$ 越大，权重就越小。为了找到最佳的 $\lambda$，我们可以从 $\lambda = 0$ 开始，在每次迭代时测量交叉验证误差，并以固定值增加 $\lambda$。

    2. 岭回归

        与 Lasso 回归类似，岭回归或 L2 正则化也在代价函数中加入了惩罚。唯一不同的是，惩罚是使用线性回归中非截距权重的平方值计算的。因此，我们将 L2 正则化定义为

        (15) \[\begin{equation*} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} ({\theta}^{T} \cdot x^{(i)} - y^{(i)})^{2} + \lambda\sum_{j=1}^{n}\theta_{j}^{2}. \end{equation*}\]

        与 L1 正则化一样，我们可以用同样的方法估计 $\lambda$ 参数。Lasso 和 Ridge 的唯一区别是 Ridge 收敛更快，而 Lasso 更常用于特征选择。

    3. 弹性网回归

        总之，这种方法是前两种方法的结合；它结合了 Ridge 回归和 Lasso 回归的惩罚。因此，我们用以下公式定义[弹性网](https://en.wikipedia.org/wiki/Elastic_net_regularization)正则化：

        (16) \[\begin{equation*} J(\theta) = \frac{1}{m} \sum_{i=1}^{m} ({\theta}^{T} \cdot x^{(i)} - y^{(i)})^{2} + \lambda_{1}\sum_{j=1}^{n}|\theta_{j}| + \lambda_{2}\sum_{j=1}^{n}\theta_{j}^{2}. \end{equation*}\]

        与前两种方法不同，这里我们有两个正则化参数，即 \lambda_{1} 和 \lambda_{2}。因此，我们必须找到这两个参数。

5. 结论

    在本文中，我们详细探讨了回归一词、回归类型和线性回归。然后，我们解释了三种最常用的正则化技术，以及找到正则化参数的方法。最后，我们学习了线性回归和正则化方法背后的数学知识。

[How to Calculate the Regularization Parameter in Linear Regression](https://www.baeldung.com/cs/regularization-parameter-linear-regression)
