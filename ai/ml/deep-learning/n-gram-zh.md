# 究竟什么是 N-Gram？

[人工智能](https://www.baeldung.com/cs/category/ai) [深度学习](https://www.baeldung.com/cs/category/ai/deep-learning)

[自然语言处理](https://www.baeldung.com/cs/tag/nlp)

1. 简介

    在本教程中，我们将解释自然语言处理（NLP）中的一个常用概念，即 n-gram。这是大多数 NLP 课程和讲座都会涉及的基本术语。除此之外，数据科学家、机器学习工程师和开发人员也经常在他们的 NLP 项目中使用 n-gram。

    因此，除了概念的定义，我们还将解释在哪些实际应用中可以使用 n-gram。

2. 自然语言处理（NLP）

    自然语言处理（NLP）是人工智能的一个分支，涉及计算机与人类（自然）语言之间的交互。自然语言处理的目标是使计算机系统能够像人类一样理解人类语言。NLP 已经应用了几十年，但由于计算机硬件和软件的进步，它越来越受欢迎。

    NLP 包括多种多样的分支学科，从语言意义研究到统计机器翻译、语音识别和问题解答。NLP 中最受欢迎的任务包括

    - [垃圾邮件过滤](https://towardsdatascience.com/how-to-identify-spam-using-natural-language-processing-nlp-af91f4170113)
    - [情感分析](https://www.baeldung.com/cs/sentiment-analysis-training-data#sentiment-analysis)
    - [机器翻译](https://www.analyticssteps.com/blogs/4-types-machine-translation-nlp)
    - 文本摘要
    - 语音识别
    - [聊天机器人](https://www.baeldung.com/cs/smart-chatbots)
    - [命名实体识别](https://www.baeldung.com/cs/finding-dates-addresses-in-emails#named-entity-recognition-ner)及其他
    因此，这些任务大不相同，我们使用许多不同类型的算法来解决 NLP 问题。

3. NLP 如何工作？

    一般来说，计算机并不擅长理解文本。首先，我们需要将文本转换成数字，然后以特定的方式进行数学运算，这样计算机才能对其进行处理。即使在这一点上，计算机也能识别单词及其顺序，但却无法理解其含义。

    有一些基于transformer的复杂系统[能够引导人类进行对话](https://blog.google/technology/ai/lamda/)，但它们需要大量的数据和时间来学习。此外，即使系统输出了有意义的回答，也并不意味着它理解了文本，而只是根据输入数据给出了统计上最有可能的答案。

    1. 开发 NLP 系统的各个阶段

        无论如何，NLP 系统需要以某种方式将文本处理成数字。此外，除了模型开发，开发 NLP 系统最重要的阶段是文本预处理。我们可以采用的一些常见步骤包括

        - 文本清理--将文本转换为小写字母，并通过去除 HTML 标记、符号或有时数字等非实质性部分进行清理。
        - 删除停滞词--剔除一些不能提供有用信息的常用词。
        - [词母化和词干化](https://www.baeldung.com/cs/stemming-vs-lemmatization)--通过遵循字典中的一些规则、去除常见的前缀和后缀等类似方法，将单词简化为其基本形式或词根形式。
        - 标记化(Tokenization) - 将清理后的文本分成更小的单元，如单词、字符或它们的某些组合。

        文本清理后，我们就可以将文本[转换为计算机可读格式](https://www.baeldung.com/cs/sentence-vectors-word2vec)。转换后的单词将直接被处理到 NLP 模型中。单词通常以向量的形式表示，这就是单词嵌入。常见的嵌入方法有：

        - 一热编码(One-hot encoding)--用一个除了一个等于 1 的分量外全部为 0 的向量来表示一个词。
        - [TF-IDF](https://www.baeldung.com/cs/text-sequence-to-vector#4-tf-idf-score-strategy) - 词频-反文档频率，是一种估算文档集合中单词重要性的方法。
        - [Word2vec](https://www.baeldung.com/cs/word2vec-word-embeddings#word2vec)--通过神经网络学习的词表示法，语义相似的词具有相似的向量表示法。

        我们可以从观察到的文本中将单词分成 2、3 或 N 个连续单词集，然后嵌入这些词集，我们称之为 n-gram，而不是嵌入单个单词。

4. 什么是 N-语法？

    用一句话来解释，N-gram 就是来自特定文本源的 N 个相邻单词或字母的序列。例如，如果我们考虑这样一个句子：

    > 棕色的狐狸跳过懒惰的狗。

    如果我们想找到由句子中的单词构成的所有 5-grams ，那么我们可以得到

    > 棕狐狸跳过懒狗
    > 棕色狐狸跳过
    > 棕色的狐狸跳过
    > 狐狸跳过懒狗
    > 跳过懒惰的狗

    计算包含 K 个单词的句子中的 n 个语法的公式是

    (1) \[\begin{align*} \text{N-Grams}_{K} = K - (N - 1). \end{align*}\]

    在实践中，我们通常使用小数 N 的 n-gram，如 1-gram（单gram）、2-gram（大gram）和 3-gram（三gram）。一般来说，n-gram 是一个非常简单的概念，但它在文本挖掘和 NLP 中的应用却非常广泛。

    n-grams 的一个特殊概括是 skip-gram。在跳格中，成分不需要从可观察到的文本中连续出现，但可能会留下空白。例如，2-skip-n-gram 会跳过文本中的每第二个单词，形成 N 个单词的序列。我们使用跳格来获得比 n 格更高水平的概括能力。一些研究人员将跳字作为分类模型的特征或语言建模的方法，以[降低复杂度](https://openresearchsoftware.metajnl.com/articles/10.5334/jors.105/)。

    总的来说，n-grams 有很多应用。下面我们将介绍其中一些。

5. N-gram 的应用

    使用 n-grams 可以开发基于单词出现次数的概率模型。例如，如果我们的目标是预测 "United"后面会出现什么词，那么最有可能的就是 "States"，因为在某些文本语料库中，以 "United"开头的词组很可能以 "States"结尾。根据文本语料库中所有以 "United"开头的词组，模型就能学习到下面这个词的概率。

    当然，要取得合理的结果，模型需要庞大的文本语料库。利用这种逻辑可以构建自动完成系统。Gmail 或 Google docs 中就有类似的系统。除此之外，n-grams 还可用于以下相关模型中：

    - 拼写校正
    - 文本摘要
    - 部分语音标记及其他
    此外，正如我们之前提到的，我们可以使用 n-grams，而不是将单个单词表示为一个向量。举例来说，如果我们的目标是构建一个情感分析模型，我们有以下两个句子的例子：

    > 不，这项服务很好。
    > 这项服务不好。

    在这两种情况下，如果使用单字符串，我们将得到相同的嵌入向量集。但是，如果使用大词，并且句子中含有否定词 "不"，那么模型很可能会对 "不，这"和 "不好"这两个大词的情感进行区分。

    最后，在一些应用中，n-grams 还可用于分析蛋白质序列和 DNA 序列。

6. 结论

    在本文中，我们简要介绍了 NLP 一词和 NLP 的工作原理。在 NLP 工作流程之后，我们提出了 n-grams 的概念，并通过几个例子对其进行了详细解释。最后，我们介绍了 n-grams 的一些应用。

[What Exactly Is an N-Gram?](https://www.baeldung.com/cs/n-gram)
