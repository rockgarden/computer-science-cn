# 红黑树木的应用

[数据结构](README-zh.md) [搜索](https://www.baeldung.com/cs/category/algorithms/searching) [树](https://www.baeldung.com/cs/category/graph-theory/trees)

[红黑树](https://www.baeldung.com/cs/tag/red-black-trees)

1. 引言

    红黑树（RB）是一种平衡的二叉搜索树。在本教程中，我们将研究它的一些最重要的应用。

2. 使用 RB 树的动机

    在之前的教程中，我们研究了二叉搜索树在 O (h) 时间内对[动态集合的基本操作](https://www.baeldung.com/cs/balanced-bst-search-complexity)。

    如果树的高度较小，这些操作的速度就会很快，但如果树的高度较大，其性能就会与使用链表时的性能相当。RB 树代表了一种可能的平衡树方案，在最坏的情况下，它可以在 $O (\ln n)$ 的时间内完成基本操作，其中 n 是树的元素数。

3. RB 树的特性

    RB 树是一种二进制搜索树，它除了包含标准二进制树的键和指针外，还包含一个称为颜色的二进制字段，颜色可以是红色或黑色。通过为任意路径上的节点着色的精确规则，我们可以得到 RB 树中没有一条路径的颜色是其他路径的两倍以上，从而得到一棵近似平衡的树。

    1. 着色

        不存在节点的每个子节点都是 NULL，并具有以下属性：

        - 每个节点为红色或黑色
        - 每个 NULL 节点为黑色
        - 如果一个节点是红色的，那么两个子节点都是黑色的
        - 从一个节点到另一个节点的每条简单路径都包含相同数量的黑色节点

    2. 高度

        从不计在内的节点 x 到其子节点的黑色节点数称为节点的 b-高度，即 $bh (x)$。RB 树的 b 高度就是其根的 b 高度。

        RB 树的有趣之处在于，对于一棵有 n 个节点的树，最大高度是 $2 \ln (n + 1)$，这意味着对经典二叉树的改进。

        在 RB 树中，INSERT 和 DELETE 操作需要 $O (\ln n)$ 时间。由于它们修改了树的结构，因此生成的树有可能违反我们上面列出的属性。在这种情况下，有必要改变一些节点的颜色并修改指针的结构，这种机制被称为旋转。

4. 应用

    与其他算法相比，RB 树保证了 INSERT、DELETE 和 SEARCH 操作的最佳计算时间。因此，从计算时间的角度来看，RB 树可用于敏感的应用，如实时应用。

    不过，由于 RB 树的特性，我们也可以将其作为众多应用程序底层数据结构的基本构件。

    1. AVL 树

        [AVL](https://www.baeldung.com/java-avl-trees) 树（Adelson-Velsky 和 Landis 树）是最早发明的自平衡二叉搜索树。在 AVL 树中，两个子树高度之差最多为 1。如果不满足这一条件，则需要重新平衡。

        AVL 树是另一种在平均和最坏情况下都支持 $O (\ln n)$ 复杂度时间的结构，用于 SEARCH、INSERT 和 DELETE。AVL 树可以用红黑色来表示。因此，它们是 RB 树的一个子集。AVL 树最坏情况下的高度是 RB 树最坏情况下高度的 0.720 倍，因此 AVL 树更加严格均衡。

    2. 探戈树

        [探戈树](https://www.wikiwand.com/en/Tango_tree)是一种为快速搜索而优化的二叉搜索树，其原始描述特别使用 RB 树作为其数据结构的一部分。

    3. 函数式编程

        RB 树在函数式编程中用于构建关联数组。

        在此应用中，RB 树与 [2-4 树](https://en.wikipedia.org/wiki/2–3–4_tree)结合使用，2-4 树是一种自平衡数据结构，每个有子节点的节点都有两个、三个或四个子节点。

        对于每一棵 2-4 树，都有相应的 RB 树，其数据元素的顺序相同。我们可以证明，在 2-4 树上进行 INSERT 和 DELETE 操作等同于在 RB 树上进行颜色翻转和旋转操作。

        这一结果可以推广到证明 RB 树可以等同于 2-3 树或 2-4 树，这一结果归功于 [Guibas 和 Sedgewick](https://ieeexplore.ieee.org/document/4567957) (1978)。

    4. Java

        除了上一段，我们还特别报告了一些关于在编程语言 C ++ 和 Java 中使用 RB 树的注意事项：

        - Java 中的 [TreeSet](https://www.baeldung.com/java-tree-set) 和 [TreeMap](https://www.baeldung.com/java-treemap) 使用 RB 树进行排序和排序
        [HashMap](https://www.baeldung.com/java-hashmap) 也使用 RB 树代替 [LinkedList](https://www.baeldung.com/cs/binary-trees-vs-linked-lists-vs-hash-tables) 来存储具有碰撞散列代码的不同元素。这使得搜索这种元素的时间复杂度从 $O (n)$ 降到了 $O (\ln n)$，其中 n 是具有碰撞散列码的元素的数量。

    5. 计算几何

        RB 树在计算几何中的应用有很多。这里我们举两个有趣的例子：

        - [Cgal](https://citeseerx.ist.psu.edu/pdf/d1dcb03c456a1146a1be95420db833c473d3181b) 库（计算几何算法库）中的多集合类模板--受标准模板库（STL）中多集合类模板的启发
        - [圆间包含层次的扫线算法](https://projecteuclid.org/download/pdf_1/euclid.jjiam/1150725475)--在最坏情况下，包含层次的工作时间为 $O (n \ln n)$。Deok-Soo Kim、Byunghoon Lee 和 Kokichi Sugihara 的算法使用扫线法和 RB 树进行高效计算。

    6. Linux 内核

        完全公平调度程序（CFS）是 Linux 内核 2.6.23 版本中的一个进程调度程序名称。它管理 CPU 的目的是最大限度地提高其平均利用率。CFS 将任务表示成一棵树，并找出下一个要运行的任务。

        CFS 使用虚拟运行时间（vruntime）将每个任务存储在 RB 树中。树中最左边的节点将是 vruntime 最少的节点。当 CFS 需要选择下一个要运行的任务时，它会选择最左边的节点。

        值得一提的 RB 树在 [Linux内核](https://www.baeldung.com/linux/kernel-versions-32-vs-64-bit) 中的另一个用途与内存管理有关。RB 树会跟踪进程的虚拟内存段，其中范围的起始地址是关键。

    7. 机器学习

        RB 树在机器学习和数据挖掘领域有着广阔的应用空间，可以提高传统算法的性能。

        例如，它们被用于[K-mean聚类](https://www.baeldung.com/cs/clustering-unknown-number)算法，以降低时间复杂性。

    8. 数据库引擎

        数据库引擎中的数据索引直接或间接使用 RB 树。

        例如，MySQL 使用 B+ 树，它可以看作是 B 树的一种。RB 树的结构类似于阶数为 4 的 B 树。

        B+ 树具有以下特点：

        - 非叶节点不存储数据。它们只存储索引（冗余）--这允许存储多个索引
        - 叶节点包含所有索引字段
        - 叶节点用指针连接，可提高性能

5. 结论

    在本教程中，我们简要回顾了 RB 树的一些重要应用。值得注意的是 RB 树与其他技术的关系，这使得在许多实现中提高排序和搜索操作的效率成为可能。

    我们认为，它们在 Linux 内核中的应用尤其有趣。免费提供的与 RB 树相关的部分代码对如何在复杂的实际问题中使用该算法很有启发。
