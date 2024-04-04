# \[检索]为什么 RAG 管道会失败？高级 RAG 模式 — 第 1 部分

The failures in RAG pipelines can be attributed to a cascade of challenges spanning the retrieval of data, the augmentation of this information, and the subsequent generation process. Beyond the technical intricacies, external factors like data biases, ever-evolving domains, and the dynamic nature of language further complicate the landscape. This article delves into the myriad reasons behind these failures, offering a holistic view of the hurdles faced by RAG implementations.

\
RAG 管道中的故障可归因于一系列挑战，包括数据检索、信息扩充以及后续生成过程。除了技术上的复杂性之外，数据偏差、不断发展的领域和语言的动态性质等外部因素也使情况进一步复杂化。本文深入探讨了这些失败背后的无数原因，全面了解了 RAG 实施面临的障碍。

In this first post of “Advanced RAG Pipelines” series, we will go over why RAG pipelines fail by dividing potential problems into a-Retrieval problems, b-Augmentation problems and c-Generation problems.\
在“高级 RAG Pipeline”系列的第一篇文章中，我们将通过将潜在问题分为 a-Retrieval 问题、b -Augmentation 问题和 c 生成问题来讨论 RAG Pipeline失效的原因。

## a) Retrieval problems 检索问题 <a href="#id-7c4f" id="id-7c4f"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:1400/1*Zj2vPYU6i31yTifI_Op4TA.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

Usually, if a RAG system is not performing well, it is because the retrieval step is having a hard time finding the right context to use for generation.\
通常，如果 RAG 系统性能不佳，那是因为检索步骤很难找到用于生成的正确上下文。

When using vector search to return similarity-based results, discrepancies can arise for several reasons:\
使用矢量搜索返回基于相似性的结果时，可能会出现差异，原因如下：

1\. **Semantic Ambiguity**: Vector representations, such as word embeddings, may not capture nuanced differences between concepts. For instance, the word “apple” might refer to the fruit or the tech company. Embeddings may conflate these meanings, leading to non-relevant results.\
1\. 语义歧义：向量表示（例如词嵌入）可能无法捕捉到概念之间的细微差别。例如，“苹果”一词可能是指水果或科技公司。嵌入可能会混淆这些含义，从而导致不相关的结果。

2\. **Magnitude vs. Direction**: Cosine similarity, a common measure, focuses on the direction of vectors and not their magnitude. This might result in matches that are semantically distant but directionally similar.\
2\. 幅度与方向：余弦相似度是一种常见的度量，侧重于向量的方向而不是它们的大小。这可能会导致语义上相距遥远但方向相似的匹配。

3\. **Granularity Mismatch**: Your query vector may represent a specific concept, but if your dataset has only broader topics, you may retrieve broader results than desired.\
3\. 粒度不匹配：查询向量可能表示特定概念，但如果数据集只有更广泛的主题，则可能会检索到比预期更广泛的结果。

4\. **Vector Space Density**: **In high-dimensional spaces, the difference in distances between closely related and unrelated items might be very small. This can lead to seemingly unrelated results being considered relevant.**\
4\. 向量空间密度：在高维空间中，密切相关和不相关的项目之间的距离差异可能非常小。这可能导致看似不相关的结果被认为是相关的。

5\. **Global vs. Local Similarities**: Most vector search mechanisms identify global similarities. Sometimes, you might be interested in local or contextual similarities which are not captured.\
5\. 全局相似性与局部相似性：大多数向量搜索机制都识别全局相似性。有时，您可能对未捕获的局部或上下文相似性感兴趣。

6\. **Sparse Retrieval Challenges**: Retrieval mechanisms might struggle to identify the right passages in vast datasets, especially _if the information required is spread thinly across multiple documents_.\
6\. 稀疏检索挑战：检索机制可能难以在庞大的数据集中识别正确的段落，特别是当所需的信息分散在多个文档中时。

e.g. When querying about a niche topic like “Graph Machine Learning use-cases in cloud security”, if the retrieval system fetches generic articles on “cloud security” without specifics on Graph ML, the subsequent generation will miss the mark.\
例如，当查询像“云安全中的图形机器学习用例”这样的小众主题时，如果检索系统获取有关“云安全”的通用文章，而没有关于图形ML的细节，那么后续生成将错过标记。

As generative AI and transformers become more popular, **dynamic embeddings** and **context-aware searches** might reduce some of these discrepancies. Additionally, hybrid models combining symbolic and sub-symbolic AI elements might offer better precision.\
随着生成式 AI 和 transformer 变得越来越流行，动态嵌入和上下文感知搜索可能会减少其中一些差异。此外，结合符号和子符号 AI 元素的混合模型可能会提供更好的精度。

## b) Augmentation Problems b） 增强问题 <a href="#id-27d9" id="id-27d9"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:1400/1*fn499Zb3k75Pc4CL0boIdg.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

**1.Integration of Context**: The challenge here is smoothly integrating the context of retrieved passages with the current generation task. If not done well, the output might appear disjointed or lack coherence.\
1.上下文的整合：这里的挑战是将检索到的段落的上下文与当前生成任务顺利集成。如果做得不好，输出可能会显得脱节或缺乏连贯性。

Example: If a retrieved passage provides in-depth information about “Python’s history” and the generation task is to elaborate on “Python’s applications”, the output might overemphasize the history at the expense of applications.\
示例：如果检索到的段落提供了有关“Python 的历史”的深入信息，并且生成任务是详细说明“Python 的应用程序”，则输出可能会以牺牲应用程序为代价过强调历史记录。

**2. Redundancy and Repetition:** If multiple retrieved passages contain similar information, the generation step might produce repetitive content.\
2\. 冗余和重复：如果检索到的多个段落包含相似的信息，则生成步骤可能会产生重复的内容。

Example: If three retrieved articles all mention “PyTorch’s dynamic computation graph”, the generated content might redundantly emphasize this point multiple times.\
示例：如果检索到的三篇文章都提到了“PyTorch 的动态计算图”，则生成的内容可能会多次冗余地强调这一点。

**3. Ranking and Priority:** Deciding the importance or relevance of multiple retrieved passages for the generation task can be challenging. The augmentation process must weigh the value of each passage appropriately.\
3\. 排名和优先级：确定多个检索到的段落对生成任务的重要性或相关性可能具有挑战性。增强过程必须适当权衡每个段落的价值。

Example: For a query on “cloud security best practices”, if a retrieved passage about “two-factor authentication” is ranked lower than a less crucial point, the final output might misrepresent the importance of two-factor authentication.\
示例：对于有关“云安全最佳实践”的查询，如果检索到的有关“双因素身份验证”的段落排名低于不太关键的点，则最终输出可能会歪曲双因素身份验证的重要性。

**4. Mismatched Styles or Tones:** Retrieved content might come from sources with diverse writing styles or tones. The augmentation process needs to harmonize these differences to ensure a consistent output.\
4\. 风格或语气不匹配：检索到的内容可能来自具有不同写作风格或语气的来源。增强过程需要协调这些差异，以确保一致的输出。

Example: If one retrieved passage is written in a casual tone while another is more formal, the final generation might oscillate between these styles, leading to a less cohesive response.\
示例：如果检索到的一段话是用随意的语气写的，而另一段则更正式，那么最后一代人可能会在这些风格之间摇摆不定，从而导致反应不那么连贯。

**5. Over-reliance on Retrieved Content:** The generation model might lean too heavily on the augmented information, leading to outputs that parrot the retrieved content rather than adding value or providing synthesis.\
5\. 过度依赖检索内容：生成模型可能过于依赖增强信息，导致输出鹦鹉学舌地检索到的内容，而不是增加价值或提供综合。

Example: If the retrieved passages offer multiple perspectives on “Graph Machine Learning Technologies”, but the generated output only reiterates these views without synthesising or providing additional insight, the augmentation hasn’t added substantial value.\
示例：如果检索到的段落提供了关于“图形机器学习技术”的多个观点，但生成的输出只是重申了这些观点，而没有综合或提供额外的见解，则增强没有增加实质性价值。

## c) Generation problems c） 生成问题 <a href="#id-0e5c" id="id-0e5c"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:1400/1*LdorRsM5qnHLf5cLfqotcA.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

### LLM related problems LLM相关问题 <a href="#id-976a" id="id-976a"></a>

* Hallucinations — LLM making up non-factual data\
  幻觉——LLM编造非事实数据
* Misinformation — factuality problems, training data includes incorrect information.\
  错误信息——事实性问题，训练数据包含不正确的信息。
* Limited context lengths — You will be limited with the context length of the LLM of choice. Usually the larger the context length is the more context you will be able to submit along with the LLM.\
  有限的上下文长度 — 您将受到所选上下文长度LLM的限制。通常，上下文长度越大，您可以提交的上下文就越多。LLM

(For a more comprehensive take on inherent problems with LLM’s refer to my earlier post -Taming the Wild — Enhancing LLM Reliability- [here](https://medium.com/@343544/how-to-improve-llm-reliability-30a14219d918).)\
（有关LLM固有问题的更全面的看法，请参阅我之前的帖子 - 驯服野外 - 增强LLM可靠性）。

Below is a list of further generation problems which may undermine the performance of RAG pipelines…\
以下是可能破坏 RAG 管道性能的进一步生成问题列表......

**1.Coherence and Consistency:** Ensuring the generated output is logically coherent and maintains a consistent narrative, especially when integrating retrieved information, can be challenging.\
1.连贯性和一致性：确保生成的输出在逻辑上是连贯的，并保持一致的叙述，尤其是在整合检索到的信息时，可能具有挑战性。

Example: The output might start discussing “Python’s efficiency in machine learning” and abruptly switch to “Python’s use in web development” without a clear transition.\
示例：输出可能会开始讨论“Python 在机器学习中的效率”，然后突然切换到“Python 在 Web 开发中的使用”，而没有明确的过渡。

**2. Verbose or Redundant Outputs:** The generation model might produce unnecessarily lengthy responses or repeat certain points.\
2\. 冗长或冗余输出：生成模型可能会产生不必要的冗长响应或重复某些点。

Example: In elaborating on “advantages of PyTorch”, the model might mention “dynamic computation graph” multiple times in different phrasings.\
示例：在阐述“PyTorch 的优势”时，模型可能会以不同的措辞多次提到“动态计算图”。

**3. Over-generalization:** The model might provide generic answers instead of specific, detailed responses tailored to the query.\
3\. 过度泛化：模型可能会提供通用答案，而不是针对查询量身定制的具体、详细的响应。

Example: A query about “differences between PyTorch and TensorFlow” might receive a broad response about the importance of deep learning frameworks without addressing the specific differences.\
示例：关于“PyTorch 和 TensorFlow 之间的差异”的查询可能会收到有关深度学习框架重要性的广泛响应，而没有解决具体差异。

**4. Lack of Depth or Insight:** Even with relevant retrieved information, the generated response might not delve deep enough or provide insightful synthesis.\
4\. 缺乏深度或洞察力：即使检索到相关的信息，生成的响应也可能不够深入或提供有见地的综合。

Example: When asked about “potential applications of Graph Machine Learning in cloud technologies”, the model might list general applications without elaborating or providing unique insights.\
示例：当被问及“图形机器学习在云技术中的潜在应用”时，该模型可能会列出一般应用程序，而不会详细说明或提供独特的见解。

**5. Error Propagation from Retrieval:** Mistakes or biases in the retrieved data can be carried forward and amplified in the generation.\
5\. 检索误差传播：检索到的数据中的错误或偏差可以在生成过程中结转和放大。

Example: If a retrieved passage inaccurately claims “next.js is a backend framework”, the generated content might expand on this incorrect premise.\
示例：如果检索到的段落错误地声称“next.js是一个后端框架”，则生成的内容可能会在此不正确的前提上扩展。

**6. Stylistic Inconsistencies:** The generated content might not maintain a consistent style, especially when trying to blend information from diverse retrieved sources.\
6\. 风格不一致：生成的内容可能无法保持一致的风格，尤其是在尝试混合来自不同检索来源的信息时。

Example: Mixing formal technical explanations with casual anecdotal content within the same response.\
示例：在同一回复中将正式的技术解释与随意的轶事内容混合在一起。

**7. Failure to Address Contradictions:** If retrieved passages contain contradictory information, the generation model might struggle to reconcile these differences or might even reproduce the contradictions in the output.\
7\. 未能解决矛盾：如果检索到的段落包含矛盾的信息，生成模型可能难以调和这些差异，甚至可能在输出中重现矛盾。

Example: If one retrieved source says “PyTorch is primarily for research” and another says “PyTorch is widely used in production”, the generated response might confusingly state both without clarification.\
示例：如果检索到的一个源说“PyTorch 主要用于研究”，而另一个说“PyTorch 在生产中广泛使用”，则生成的响应可能会混淆地说明两者，而无需澄清。

**8. Context Ignorance:** The generated response might sometimes miss or misinterpret the broader context or intent behind a query.\
8\. 上下文无知：生成的响应有时可能会遗漏或误解查询背后的更广泛的上下文或意图。

Example: In response to “Tell me a fun fact about machine learning,” the model might produce a highly technical point rather than something light and interesting for a general audience.\
示例：为了回答“告诉我一个关于机器学习的有趣事实”，该模型可能会产生一个高度技术性的观点，而不是对普通受众来说轻松有趣的东西。

These generation problems highlight the complexities of producing accurate, relevant, and high-quality content, even when augmented with retrieved data. It underscores the need for iterative refinement, feedback loops, and possibly even domain-specific tuning to optimize the generation process, especially in specialized fields like cloud technologies and machine learning.\
这些生成问题凸显了生成准确、相关和高质量内容的复杂性，即使使用检索到的数据进行扩充也是如此。它强调了迭代优化、反馈循环甚至可能特定于领域的调整的必要性，以优化生成过程，尤其是在云技术和机器学习等专业领域。

Next we will look at how we can solve these problems…\
接下来，我们将看看如何解决这些问题......

\*\*References\*\*: \*\*引用\*\*：\
\- Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111–3119).\
\- Mikolov，T.，Sutskever，I.，Chen，K.，Corrado，GS和Dean，J.（2013）。单词和短语及其组合性的分布式表示。在神经信息处理系统进展中（第 3111-3119 页）。\
\- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.\
\- Devlin，J.，Chang，MW，Lee，K.和Toutanova，K.（2018）。Bert：用于语言理解的深度双向转换器的预训练。arXiv 预印本 arXiv：1810.04805。
