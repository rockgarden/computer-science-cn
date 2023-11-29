# 用于机器学习的 Python

[机器学习](https://www.baeldung.com/cs/category/ai/ml) [编程](https://www.baeldung.com/cs/category/core-concepts/programming)

[项目管理](https://www.baeldung.com/cs/tag/project-management)

1. 简介

    在本教程中，我们将了解为什么 [Python](https://www.baeldung.com/cs/max-int-java-c-python) 是[机器学习](https://www.baeldung.com/cs/analytical-inductive-learning-prior-knowledge)的首选编程语言。

2. 机器学习与编程

    我们可以用任何高级编程语言（Python、R、Matlab）、低级编程语言（C、C++）或它们的组合来实现机器学习算法。

    不过，我们通常选择高级语言，以便更快、更准确地编写更易于理解和维护的代码。因此，我们在机器学习的现代软件开发中使用多种语言的组合。

    使用高级语言可以为任何机器学习算法的编码提供简洁、紧凑和丰富的界面。大部分细节都由语言或库来处理，这样我们就可以专注于想法和逻辑。

    让我们看看是什么让 Python 脱颖而出。

3. Python 编程语言

    Python 的设计宗旨是提高质量、生产率、可移植性和集成性。

    由于语法清晰、设计连贯，Python 代码具有很高的可读性、简洁性和可扩展性。我们可以很快学会它。它的程序看起来就像算法，这让我们可以专注于想法，而不是算法逻辑附带的实现细节。

    Python 提高了开发速度。我们发现在 Python 中编写程序的速度非常快。这是因为解释器会处理大多数细节，如类型声明、存储、内存管理、中断处理和构建程序。这使得 Python 灵活而敏捷。

    Python 是开源的、可访问的，并拥有庞大的社区基础。它的标准库包含不同领域的各种包和模块。

    Python 还具有[可移植性](https://www.baeldung.com/cs/software-quality)。我们可以在 Windows 上编写 Python 代码，也可以在 Mac 和 Linux 上运行，无需修改。这是因为 Python 语言的核心部分及其库是平台中立的。对于平台不兼容的边缘情况，Python 提供了构建依赖关系的工具。

    Python 具有高度集成性。我们可以在任何级别上将 Python 代码与其他系统组件轻松混合。例如，我们可以在 Python 中调用快速的 C/C++。

    1. 示例

        让我们考虑一个[线性搜索](https://www.baeldung.com/cs/linear-search-faster)的 Python 函数：

        ```py
        def linear_search(input_list, search_item):
            for item in input_list:
                if item == search_item:
                    return True
            return False
        ```

        由于没有涉及数据类型和列表索引等细节，因此读起来就像一个算法。

        相比之下，同样算法在 C++ 中的实现涉及更多细节，因为我们需要处理内存管理、类型和数组索引：

        ```c++
        bool linear_search(int *input_list, int size, int search_item){ 
            for (int i=0; i < size; i++){
                if (input_list[i] == search_item){
                    return true;
                }
            }
            return false;
        }
        ```

        我们必须为其他数据类型编写另一个函数来进行线性搜索。由于 Python 采用[动态类型](https://www.baeldung.com/cs/statically-vs-dynamically-typed-languages)，因此不会出现这种情况。

4. 用于机器学习的 Python

    Python 有许多适用于各个领域的库。

    我们使用 [numpy](https://www.baeldung.com/cs/svm-multiclass-classification) 来进行科学计算，它使用一种架构高效的数据表示方式。pandas 库用于数据分析以及结构化和非结构化数据的处理。此外，我们还使用 matplotlib 绘制所有数字和图表。这些库本身并不涉及机器学习，但支持我们构建和测试 ML 模型。

    1. 关键机器学习框架

        让我们来看看四个关键的机器学习 Python 框架：[scikit-learn](https://scikit-learn.org/stable/)、[Keras](https://keras.io/)、[TensorFlow](https://www.tensorflow.org/) 和 [PyTorch](https://pytorch.org/)。

        scikit-learn 或 sklearn 是最常用的机器学习 Python 库。它具有健壮性、可扩展性和扩展性。在分类、[回归](normalization-vs-standardization-zh.md)、[聚类](https://www.baeldung.com/cs/dbscan-algorithm)和[降维](https://www.baeldung.com/cs/feature-selection-reduction-for-text-classification)任务方面，我们可以获得许多久经考验、有据可查的工具。

        TensorFlow 是一个开源的完整平台，我们可以用它来完成多种机器学习任务。谷歌开发了它，并于 2015 年将其公开。它为构建和训练深度学习中的自定义模型提供了丰富的抽象。TensorFlow 拥有一个由社区资源、库和工具组成的充满活力的生态系统，可用于构建和部署可快速扩展的机器学习应用程序。

        另一方面，Keras 是一个高级神经网络库，是 TensorFlow 的[封装](https://www.baeldung.com/java-wrapper-classes)。Keras 对用户更加友好，因为它向用户隐藏了 Tensorflow 的所有底层细节。因此，我们可以使用 Keras 来定义我们的模型，因为它更容易使用。如果我们需要 Keras 不提供的特定 TensorFlow 功能，我们可以直接使用 TensorFlow。

        PyTorch 是一个基于 Torch 的较新的[深度学习](https://www.baeldung.com/cs/end-to-end-deep-learning)框架。Facebook 的人工智能研究小组于 2017 年将其开源。与 TensorFlow 相比，我们发现它更简单、更易用。它还为我们提供了更大的灵活性和更好的内存管理。此外，它还使用动态计算图来训练模型。此外，PyTorch 速度更快，让我们的代码更易于管理和理解。

5. 替代语言

    本节将探讨我们可以用于机器学习的其他编程语言。

    1. C/C++

        C/C++ 的运行时间比同类语言更快，因此适用于实现那些要求低延迟而又不影响准确性的机器学习问题。C++ 还拥有丰富的机器学习库支持。

        另一方面，C++ 的缺点是，与 Python 相比，它在解决相同问题时编写的代码更少。这是因为我们必须在代码中直接指定大部分实现细节。

        我们可以通过 [GitHub](https://github.com/Baeldung/posts-resources/tree/main/cs-articles/python-and-machine-learning) 上一个简单的 C++ 线性回归模型来了解这些想法。它篇幅很长，而且非常底层。

    2. Matlab

        Matlab 提供了一个基于数值计算的环境，支持多种语言。它非常适合实现数学含量较高的机器学习算法（大量使用统计和微积分）。

        Matlab 提供最快的执行时间、最好的集成开发环境、丰富的库和其他语言集成。但是，由于它不够紧凑和灵活，因此学习难度很大。此外，它是授权软件，因此在行业中的应用有限。

        此外，由于 Matlab 以应用程序或模块的形式提供关键算法的底层 Matlab 代码，因此我们需要帮助查看这些代码。因此，调试 Matlab 代码需要大量工作。

        此外，Matlab 代码还不具备可移植性。

    3. R

        R 是一种用于统计编程和可视化的开源编程语言。它源于 Lisp，因此是一种解释型语言。

        R 适用于研究中的原型算法，但不太适合工业应用。此外，我们发现它的文档很复杂，因此学习曲线很陡峭。此外，R 语言在实现机器学习算法时需要更多的一致性，因为它的库使用不同的命名约定和设计风格。

    4. Node.js

        我们主要将 Node.js 用于服务器端应用程序。最近几年，开发者社区开始将 Node.js 用于机器学习。

        大多数主要的机器学习库（如 OpenCV、TensorFlow 和 NumPy）都有 Node.js 实现。这些库提供了易于使用的机器学习模型，因此我们可以直接在 Node.js 环境中训练和部署我们的解决方案。

6. 比较

    在本节中，我们将对所有候选的机器学习编程语言进行比较：

    | 标准  | Python | C/C++  | Matlab | R       | Node.js |
    |-----|--------|--------|--------|---------|---------|
    | 可用性 | 简单     | 复杂     | 复杂     | 复杂      | 简单      |
    | 许可证 | 开源     | 开放源代码  | 商业     | 开源      | 开源      |
    | 抽象化 | 高级     | 低级     | 低级     | 高级      | 高级      |
    | 库支持 | 非常高    | 有限     | 中级     | 有限      | 高级      |
    | 用户群 | 最高     | 低级     | 非常低    | 低级      | 中级      |
    | 适用性 | 所有应用程序 | 实时应用程序 | 科学应用程序 | 可视化应用程序 | 网络应用程序  |

    该表显示，Python 是最适合编码任何机器学习解决方案的语言。

7. 结论

    本文将 Python 作为机器学习问题的核心编程语言进行了研究。

    最后，我们发现 Python 在实现机器学习算法方面优于其他语言，因为它简单、灵活、支持丰富的库，而且用户社区非常活跃。

[Python for Machine Learning](https://www.baeldung.com/cs/python-ml-programming)
