# 机器学习中的信息增益

[机器学习](https://www.baeldung.com/cs/category/ai/ml) [数据科学](https://www.baeldung.com/cs/category/ai/data-science)

[熵](https://www.baeldung.com/cs/tag/entropy)

1. 简介

    在本教程中，我们将介绍信息增益。我们将用熵来解释信息增益，信息增益是信息论中的一个概念，在包括机器学习在内的许多科学和工程领域都有应用。然后，我们将展示如何使用它来拟合[决策树](https://www.baeldung.com/cs/decision-tree-vs-naive-bayes)。

2. 信息

    直观地说，信息就是能增加我们对系统、过程或现象的了解的任何东西。但什么是知识呢？

    抛开哲学细节不谈，我们可以说它是不确定性的镜像。我们对某件事的把握越大，我们的知识就越丰富。反之亦然，这也是信息论的核心：我们越不确定，我们的知识就越少。因此，信息就是任何能减少我们不确定性的东西。

    下一个问题是如何量化不确定性和信息。让我们举例说明。

    1. 举例说明： 不确定性和信息

        让我们考虑一个装有 1 个红球和 2 个白球的不透明盒子。任何人从中随机抽取一个球，都有 1/3 的机会抽到红球，2/3 的机会抽到白球。虽然我们不能确定颜色，但我们可以用下面的随机变量来描述结果及其可能性：

        (1)
        \[\begin{equation*}  \begin{pmatrix} \text{red} & \text{white} \\ \frac{1}{3} & \frac{2}{3} \end{pmatrix} \end{equation*}\]

        现在，如果我们发现有人抽到了红球，那么我们能得到的颜色就没有不确定性了：

        (2)
        \[\begin{equation*}  \begin{pmatrix} \text{red} & \text{white} \\ 0 & 1 \end{pmatrix} \end{equation*}\]

        我们通过得知有人画了红球而获得的信息量，就是概率模型（2）和（1）之间的差值。更具体地说，就是不确定性的差异，我们用熵来衡量。

    2. 熵

        在信息论中，[熵](https://www.baeldung.com/cs/cs-entropy-definition)是概率 $p_1, p_2, \ldots, p_n$ 的函数 H。它满足我们所期望的不确定性度量应该满足的三个条件：

        H 在 $p_1, p_2, \ldots, p_n$ 中是连续的。这个条件可以确保在任何一个 $\boldsymbol{p_i}$ 发生微小变化时，不确定性都不会有很大的波动。

        如果概率都相等$\boldsymbol{p_1 = p_2 = \ldots = p_n = \frac{1}{n}}$，那么不确定性应该随着 $\boldsymbol{n}$的增长而增加。在数学上，$H \left( \frac{1}{n}, \frac{1}{n}, \ldots, \frac{1}{n} \right)$ 应该与 n 单调。

        最后，将 $\boldsymbol{H(p_1, p_2, \ldots,p_n)}$分解为 $\boldsymbol{p_1, p_2, \ldots,p_n}$的子组的不确定性的（加权）总和的任何方法都应该得到相同的结果。

        [香农](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)证明，满足这些条件的唯一函数是

        (3)
        \[\begin{equation*} H ( p_1, p_2, \ldots, p_n) = - K\sum_{i=1}^{n} p_i \log(p_i) \end{equation*}\]

        在这里，K 是一个常数，所有的对数都有相同的基数。通常，我们取二进制对数。所以

        (4)
        \[\begin{equation*} H (p_1, p_2, \ldots, p_n) = - \sum_{i=1}^{n} p_i \log_2 p_i \end{equation*}\]

    3. 信息量就是熵的差异

        既然我们必须计算熵，那么我们就可以测量信息量。我们将其定义为不确定性的变化。

        形式上，假设 S 是我们在学习信息 z 之前的信念系统的随机变量模型。让我们用 $S \mid z$ 表示我们更新后的信念状态（z 之后），并扩展符号，以便 H 除了概率之外还能作用于 S 和 $S \mid z$。

        这样，z 的信息量 AI(z) 就是：

        (5)
        \[\begin{equation*}  AI(z) = H(S) - H(S \mid z) \end{equation*}\]

    4. 举例说明： 计算信息量

        让我们来计算一下，当我们发现有人从一个最初装有一个红球和两个白球的盒子中取出了红球时，我们所获得的信息量是多少。

        信息之前的熵为

        \[\begin{aligned} H \left( \frac{1}{3}, \frac{2}{3} \right) &= - \frac{1}{3} \log_2 \frac{1}{3} - \frac{2}{3} \log_2 \frac{2}{3} \\ & \approx - \frac{1}{3} \cdot (-1.585) - \frac{2}{3} (-0.585) &= 0.918 \end{aligned}\]

        之后的熵为

        \[H \left(0, 1 \right) = - \underbrace{0 \log_2 0}_{=0} - 1 \log_2 1 = 0+0 = 0 \\\]

        所以，信息量是 0.918-0=0.918。在这种情况下，我们得到了最大的信息量，因为信息消除了我们的不确定性。

        如果有人告诉我们他抽到的是白球而不是红球呢？概率将变为

        (6)
        \[\begin{equation*} \begin{pmatrix} \text{red} & \text{white} \\ \frac{1}{2} & \frac{1}{2} \end{pmatrix} \end{equation*}\]

        所以，我们的新熵是

        \[H \left( \frac{1}{2}, \frac{1}{2} \right) = - \frac{1}{2} \log_2 \frac{1}{2} - \frac{1}{2} \log_2 \frac{1}{2} = -\frac{1}{2} (-1) - \frac{1}{2} (-1) = \frac{1}{2} + \frac{1}{2} = 1\]

        现在的数量是 0.918-1=-0.082，一个负数。等等，我们不是说信息应该减少不确定性吗？理想的情况应该是这样。我们只接收能减少不确定性的信息。然而，并不是所有的信息都是这样的。有些信息会让我们更加不确定，这样的信息量就是负值。

        这就是为什么我们把信息量定义为信息前后的熵差，而不是相反。这样，正数代表知识的增加（即不确定性的减少），这样更直观。

        由此我们可以看出，信息是任何影响我们关于某个过程或现象的概率模型的东西。如果信息降低了不确定性，那么它的贡献就是正数，反之则是负数。

3. 信息增益

    既然知道了如何计算熵和信息量，我们就可以定义信息增益了。我们将在决策树的背景下进行定义。

    1. 决策树

        在[决策树](https://www.baeldung.com/cs/machine-learning-intro#classification)的每个内部节点上，我们都要检查对象的一个特征。根据其值，我们会访问该节点的一个子树。

        我们重复这一步骤，直到找到叶子节点。叶子节点不检查任何特征，每个节点都包含一个训练数据子集。一旦找到叶子节点，我们就会将对象分配到叶子节点数据子集中的[多数](https://www.baeldung.com/cs/array-majority-element)类。

    2. 信息增益

        要建立决策树，我们需要决定在哪个节点检查哪个特征。例如，假设我们有两个未使用的特征：a 和 b，都是二进制的。我们还有五个对象，其中两个是正面的：

        \[S : \qquad \begin{bmatrix} a & b & class \\ \hline 0 & 0 & positive \\ 0 & 1 & positive \\ 1 & 0 & negative \\ 1 & 1 & positive \\ 0 & 0 & negative \end{bmatrix}\]

        我们应该测试哪个功能来添加新节点？信息增益可以帮助我们做出决定。它是我们通过检测特征获得的预期信息量。直观地说，预期信息量最大的特征是最佳选择。因为平均而言，它能最大程度地减少我们的不确定性。

    3. 首先，我们计算熵

        在我们的例子中，添加新节点前的熵为

        \[H(S) = - \frac{2}{5} \log_2 \frac{2}{5} - \frac{3}{5} \log_2 \frac{3}{5} = 0.971\]

        这就是我们对随机对象类别的不确定性的度量（假设之前的检查让它到达了树中的这一点）。如果我们选择 a 作为新节点的测试特征，我们将得到两个子树，分别覆盖 S 的两个子集：

        \[S \mid a = 0: \quad \begin{bmatrix} 2 \text{ positive objects} \\ 1 \text{ negative object} \end{bmatrix} \qquad \qquad S \mid a = 1: \quad \begin{bmatrix} 1 \text{ positive object} \\ 1 \text{ negative object} \end{bmatrix}\]

        它们的熵是

        \[\begin{aligned} H(S \mid a = 0) &= H \left( \frac{2}{3}, \frac{1}{3} \right) &&= 0.918 \\ H(S \mid a = 1) & = H \left( \frac{1}{2}, \frac{1}{2}\right) &&= 1 \end{aligned}\]

    4. 然后，我们计算增益

        信息增益 IG(a) 是我们通过检查特征 a 所获得的预期信息量：

        \[\begin{aligned} IG(a) &= P(a=0)\cdot AI(a=0) + P(a=1) \cdot AI(a=1) \\ &= P(a=0) \left( H(S) - H(S \mid a = 0) \right) + P(a=1) \cdot \left( H(S) - H(S \mid a = 1) \right) \\ &= \left( P(a = 0) + P(a = 1) \right) H(S) -  P(a=0) H(S \mid a = 0) - P(a=1) H(S \mid a = 1) \\ &= H(S) -  P(a=0) H(S \mid a = 0) - P(a=1) H(S \mid a = 1) \\ &= 0.971 - 0.6 \cdot 0.918 - 0.4 \cdot 1 \\ &= 0.0202 \end{aligned}\]

        我们将 P(a=0) 和 P(a=1) 分别定义为 S 中 a=0 和 a=1 的频率。

        对 b 进行同样的计算，可以得出其增益为

        \[\begin{aligned} IG(b) &= H(S) -  P(b=0) H(S \mid b = 0) - P(b=1) H(S \mid b = 1) \\ &= 0.971 - 0.6 \cdot 0.918 - 0.4 \cdot 0 = 0.4202 \end{aligned}\]

        由于 $IG(b) > IG(a)$，我们选择 b 来创建一个新节点。

        正如我们所看到的，$\boldsymbol{IG}$倾向于最大化子集纯度的拆分，这样每个子集的多数类分数就会尽可能高。

    5. 计算公式

        IG(a) 的计算向我们展示了一般情况下的增益公式。让 S 表示通过创建新节点进行拆分的数据集。假设属性 a 可以取 m 个值：$a_1, a_2, \ldots,a_m$，并且 $p_i$ 是 S 中 $a=a_i$ 的对象的分数。那么，$\boldsymbol{a}$ 的信息增益为：

        (7)
        \[\begin{equation*} IG(a) = H(S) - \sum_{i=1}^{m}p_i H(S | a=a_i) \end{equation*}\]

        即使单个信息的贡献为负，增益也不会为负。

4. 结论

    在本文中，我们展示了如何计算信息增益。此外，我们还讨论了信息论中的熵及其与不确定性的联系。

[Information Gain in Machine Learning](https://www.baeldung.com/cs/ml-information-gain#id2306625574)
