# 如何选择皮尔逊相关性和斯皮尔曼相关性？

[数据科学](https://www.baeldung.com/cs/category/ai/data-science) [数学与逻辑](https://www.baeldung.com/cs/category/core-concepts/math-logic)

[概率与统计](https://www.baeldung.com/cs/tag/probability-and-statistics)

1. 简介

    在本教程中，我们将回顾两个概念：[Pearson](https://www.baeldung.com/cs/correlation-coefficient#pearsons-correlation-coefficient)相关和[Spearman](https://www.baeldung.com/cs/correlation-coefficient#spearmans-rank-correlation)相关。我们还将讨论如何在两者之间做出选择。

2. 相关性

    在深入研究 Pearson 和 Spearman 相关性的复杂性之前，我们先来定义[相关性](https://www.baeldung.com/cs/correlation-coefficient-vs-regression-model#correlation)。相关性是指变量之间的关系。它通过量化变量之间的关联程度来描述变量之间的关系。变量之间的相关性可以是线性的，即一个变量的移动会带动另一个变量的移动。或者，它也可以是非线性的，即一个变量的变化并不对应另一个变量的变化。

    例如，假设我们观察到一个人在冬天花费的钱随着年龄的增长而增加。我们可以假设这两个变量之间存在相关性。这里的相关性是指花费的钱随着年龄的增长而增加。

3. 皮尔逊相关性

    皮尔逊相关测量变量之间线性相关的程度和方向。它的计算方法是两个变量的协方差与它们的标准差的乘积之比。假设我们有 X 和 Y 两个变量，那么皮尔逊相关性就可以用下面的公式计算出来：

    \[\rho_{X,Y} = \frac{\operatorname{cov}(X,Y)}{\sigma_X \sigma_Y} \]

    这里，$\operatorname{cov}(X,Y)$ 是 X 和 Y 之间的协方差。$\sigma_X$ 和 $\sigma_Y$ 分别是 X 和 Y 的均值。 两个变量之间的 Pearson 相关性值介于 [-1, 1] 之间，其中 1 表示强正相关。这意味着当 X 增加时，Y 也会增加。相反，-1 表示负相关；这意味着当 X 减少时，Y 也会增加。同样，相关性接近 0 意味着不存在相关性。

4. 斯皮尔曼相关性

    斯皮尔曼相关性通过考虑变量的排序来衡量变量之间单调相关的强度和方向。单调相关是指两个变量始终朝同一方向变化。它的计算方法与皮尔逊相关性类似，但考虑了变量的顺序等级：

    \[r_s = \rho_{\text{R}(X),\text{R}(Y)} = \frac{\text{cov}(\text{R}(X),\text{R}(Y))}{\sigma_{\text{R}(X)} \sigma_{\text{R}(Y)}}\]

    这里，$\text{R}(X)$ 和 $\text{R}(X)$ 是变量 X 和 Y 的等级。同样，两个变量之间的斯皮尔曼相关性介于 [-1, 1] 之间，其中 -1 表示负单调相关。相反，1 表示强烈的正相关，而 0 则表示没有单调关系。

5. 举例说明

    假设我们有 X 和 Y 两个变量，X 是参与者的年龄，Y 是花费的金额：

    | X  | Y   |
    |----|-----|
    | 26 | 500 |
    | 56 | 780 |
    | 78 | 200 |
    | 18 | 20  |
    | 50 | 300 |

    计算这两个变量的皮尔逊相关性可以得出：

    \[\rho_{X,Y} = 0.20\]

    随后，使用数据点的排序作为等级进行斯皮尔曼相关性计算，将得到：

    \[ r_s = \rho_{\text{R}(X),\text{R}(Y)} = 0.73 \]

    这一计算结果表明，X 和 Y 之间不存在线性相关关系，但这两个变量之间存在正的单调关系。

6. 你应该选择哪一个？

    选择皮尔逊相关还是斯皮尔曼相关取决于数据的特征和手头的任务。例如，选择：

    - 线性关系选择皮尔逊相关，反之选择斯皮尔曼相关；
    - 有排序的数据选择斯皮尔曼相关，否则选择皮尔逊相关；

7. 结论

    本文概述了 Pearson 和 Spearman 相关性。

    Pearson 相关性量化线性关系，而 Spearman 相关性测量变量之间的单调相关程度。选择哪一种取决于数据的特征或分析的目标。
