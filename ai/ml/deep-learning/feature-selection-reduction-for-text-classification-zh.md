# 文本分类的特征选择与缩减

[深度学习](https://www.baeldung.com/cs/category/ai/deep-learning) [机器学习](https://www.baeldung.com/cs/category/ai/ml)

[训练](https://www.baeldung.com/cs/tag/training)

1. 简介

    在本教程中，我们将讨论文本[分类](https://www.baeldung.com/cs/ml-classification-vs-clustering#classification)中[特征](https://www.baeldung.com/cs/feature-vs-label#features)选择和特征缩减的主要方法。

2. 维度的诅咒与选择的祝福

    所有机器学习都受到一个诅咒的影响：[维度诅咒](https://www.baeldung.com/cs/correlation-classification-algorithms#the-curse-of-dimensionality)。如今，记录新信息的成本在历史上首次接近于零。因此，源于现实世界的数据集，无论是否与文本相关，往往都会包含比我们认为有信息量的特征更多的特征。

    因此，我们希望减少数据集中的特征数量，只选择那些最重要的特征。这一点对于文本来说尤为重要：一个典型的文本语料库可能有数千个独特的单词，而在任何特定文本中出现的单词却寥寥无几。

    因此，我们希望只从文本中选择能最大程度提高[分类准确率](https://www.baeldung.com/cs/ml-loss-accuracy#accuracy)的特征，而忽略其他特征。有三种简单的技术可以帮助我们：卡方检验、信息增益以及使用 n-grams 代替 uni-grams。

3. 卡方分布

    卡方检验是在分类任务中选择特征的基础技术之一。它使用的假设很少：因此，我们很容易记住并实现它。

    假设我们正在执行一项分类任务。我们有 N 个观察结果，要将它们归入 k 个独立的类别。每个观测值都只属于一个类，因此这里不存在[多类分类](https://www.baeldung.com/cs/multi-class-f1-score)。

    我们把 $\boldsymbol{x_i}$ 称为属于每个类别的观测值的数量 $\boldsymbol{i}$：

    \[\sum_{i=1}^k x_i = N\]

    现在我们可以计算某个观测值属于 k 个可用类别中第 i 个类别的概率 $p_i$。为此，我们用给定类别 $\boldsymbol{i}$ 中的观测值数量除以观测值总数。这样，对于每个类别 i，$p_i = x_i / N$。

    因此，我们也可以写出这个公式：

    \[\frac{\sum_{i=1}^k x_i}{N} = \sum_{i=1}^k p_i = 1\]

    因为 $\sum_{i=1}^k p_i = 1$，所以我们可以把 p 视为概率分布，它描述了任何一般观察结果落入每个 $\boldsymbol{k}$ 类别的可能性。因此，如果 $p_i$ 是属于第 i 个类别的概率，我们可以称 $m_i = N \times p_i$ 为与第 i 个类别相关的观测值的预期数量。

4. 奇平方(Chi-Squared)检验

    我们可以注意到，在我们迄今为止开发的结构中，$x_i$ 和 $m_i$ 对于 k 个类别中的每一个都具有相同的值，因此两者似乎都是多余的。但是，如果观测值的数量 N 足够大，我们就可以假设种群满足这一规则：

    \[\chi^2 = \sum_{i=1}^k \frac{(x_i - m_i)^2}{m_i} = N \times \sum_{i=1}^k \frac {(m_i / N - p_i)^2}{p_i}\]

    等式的右边表示除以 N 再乘以 N，这样就又出现了概率分布。用这种方法推理的好处是，我们可以使用预期计数 $\boldsymbol{m_i}$，而不是测量观测值 $\boldsymbol{x_i}$，因为后者不在右边。

    然后，我们可以将现实世界的分布与 N 接近无穷大的 $\chi^2$ 抽象分布进行比较，并计算两者之间的皮尔逊 p 值。这就是皮尔逊卡方检验的定义。

5. 信息增益

    另一种在数据集中选择特征的方法需要我们从熵的角度来思考。我们可以问自己"如果我们知道其中一个特征的假设值，分布中的熵会减少多少？"

    这表明，我们可以将获得的信息计算为两个熵的差值。一个是分布的熵，另一个是同一分布在已知其中一个值的条件下的熵。

    我们把 $H(Y)$ 称为随机变量 Y 的熵，把 $H(Y|X)$ 称为给定 X 假设值的 Y 的条件熵，我们可以将信息增益 $IG(Y,X)$ 定义为：

    \[IG(Y,X) = H(Y) - H(Y|X)\]

    如果我们需要复习计算随机变量的熵和条件熵，可以参考本网站的其他文章。信息增益就是这两者之间的差值。

6. N 符及其频率

    我们讨论的最后一种方法是最简单的，无论是在智力上还是在计算上。当我们处理文本时，我们可以创建其中包含的单词的频率分布。这些单词构成了这些文本的单词组(uni-grams)，也是文本分类的标准特征。

    但是，有些文本中的单词成对分析比单独分析更有意义。成对的连续词语被称为 "双格"（bi-grams），我们可以用它们作为对文本进行分类的特征。一般来说，由 $\boldsymbol{n}$ 个单词组成的 [n-gram](https://www.baeldung.com/cs/text-sequence-to-vector#1-bag-of-n-grams) 应该比单独的 $\boldsymbol{n}$ 个单词信息量更大，但代价是内存消耗增加。

7. 结论

    本文研究了文本分类中最常用的特征选择和缩减技术。
