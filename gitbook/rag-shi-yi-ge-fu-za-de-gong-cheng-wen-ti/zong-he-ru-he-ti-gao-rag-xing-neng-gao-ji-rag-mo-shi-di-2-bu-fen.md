---
description: 从多个维度详细描述了如何提高 RAG 性能的文章，非常值得一读。
---

# \[综合]如何提高 RAG 性能 — 高级 RAG 模式 — 第 2 部分

<figure><img src="https://miro.medium.com/v2/resize:fit:700/1*uV7i0mZUK_OSWJ5YPpTG5Q.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

In the realm of experimental Large Language Models (LLMs), creating a captivating LLM Minimum Viable Product (MVP) is relatively straightforward, but achieving production-level performance can be a formidable task, especially when it comes to building a high-performing Retrieval-Augmented Generation (RAG) pipeline for in-context learning. This post, part of the “Advanced RAG Patterns” series, delves into strategies and provides in-depth insights to enhance the performance of your RAG application. Subsequent posts will focus on implementing these strategies.\
在实验性大型语言模型领域（LLMs），创建引人入胜的LLM最小可行产品 （MVP） 相对简单，但实现生产级性能可能是一项艰巨的任务，尤其是在构建用于上下文学习的高性能检索增强生成 （RAG） 管道时。本文是“高级 RAG 模式”系列的一部分，深入探讨了提高 RAG 应用程序性能的策略并提供深入的见解。后续文章将重点介绍这些战略的实施。

_Understanding the Challenges_\
_了解挑战_

Before we explore the strategies, let’s dissect the critical challenges that contribute to suboptimal RAG system performance, classifying them into three distinct categories:\
在探讨这些策略之前，让我们剖析导致 RAG 系统性能欠佳的关键挑战，将它们分为三个不同的类别：

(For common issues with retrieval, augmentation and generation that will result in suboptimal RAG performance please refer to my earlier post Why do RAG pipelines fail? Advanced RAG Patterns — Part1)\
（有关检索、增强和生成的常见问题，这些问题将导致 RAG 性能欠佳，请参阅我之前的帖子 为什么 RAG 管道会失败？高级 RAG 模式 — 第 1 部分）。

{% content-ref url="jian-suo-wei-shi-mo-rag-guan-dao-hui-shi-bai-gao-ji-rag-mo-shi-di-1-bu-fen.md" %}
[jian-suo-wei-shi-mo-rag-guan-dao-hui-shi-bai-gao-ji-rag-mo-shi-di-1-bu-fen.md](jian-suo-wei-shi-mo-rag-guan-dao-hui-shi-bai-gao-ji-rag-mo-shi-di-1-bu-fen.md)
{% endcontent-ref %}

1\. Retrieval Problems 检索问题：

* _Semantic Ambiguity:_ Ambiguities in query interpretation.\
  语义歧义：查询解释中的歧义。
* _Vector Similarity Issues:_ Challenges with vector similarity measures like cosine similarity.\
  向量相似性问题：向量相似度度量（如余弦相似度）的挑战。
* _Granularity Mismatches:_ Mismatches in the level of granularity between query and retrieved content.\
  粒度不匹配：查询内容和检索内容之间的粒度级别不匹配。
* _Vector Space Density:_ Irregularities in vector space distribution.\
  向量空间密度：向量空间分布的不规则性。
* _Sparse Retrieval Challenges:_ Difficulty in retrieving relevant content due to sparse data.\
  稀疏检索挑战：由于数据稀疏，难以检索相关内容。

2\. Augmentation Problems 增强问题：

* _Mismatched Context:_ Content integration issues.\
  上下文不匹配：内容集成问题。
* _Redundancy:_ Repeated information.\
  冗余：重复的信息。
* _Improper Ranking:_ Incorrect ranking of retrieved content.\
  不正确的排名：检索到的内容的排名不正确。
* _Stylistic Inconsistencies:_ Inconsistencies in writing style.\
  文体不一致：写作风格不一致。
* _Over-reliance on Retrieved Content:_ Heavy reliance on retrieved content, sometimes at the expense of original generation.\
  过度依赖检索的内容：严重依赖检索到的内容，有时以牺牲原始生成为代价。

3\. Generation Problems:  生成问题：

* _Logical Inconsistencies:_ Contradictions or illogical statements.\
  逻辑不一致：矛盾或不合逻辑的陈述。
* _Verbosity:_ Excessive verbosity in generated content.\
  详细程度：生成的内容过于详细。
* _Over-generalization:_ Providing overly generalized information.\
  过度概括：提供过度概括的信息。
* _Lack of Depth:_ Superficial content.\
  缺乏深度：肤浅的内容。
* _Error Propagation:_ Errors originating from retrieved data.\
  错误传播：源自检索数据的错误。
* _Stylistic Issues:_ Stylistic inconsistencies.\
  文体问题：文体不一致。
* _Failure to Reconcile Contradictions:_ Inability to resolve conflicting information.In this post we will focus more on the methods to fix RAG pipelines…\
  无法调和矛盾：无法解决相互矛盾的信息。在这篇文章中，我们将更多地关注修复 RAG 管道的方法......

_Strategies for Enhanced Performance_\
_提高性能的策略_ <a href="#f2f3" id="f2f3"></a>
----------------------------------------

### **1. Data 数据** <a href="#b5d7" id="b5d7"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:700/1*LdorRsM5qnHLf5cLfqotcA.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

For a high-performing RAG system, the data needs to be clean, consistent, and context-rich. Text should be standardized to remove special characters and irrelevant information, thereby enhancing retriever efficiency. Entities and terms should be disambiguated for consistency, while duplicate or redundant information should be eliminated to streamline retriever focus. Factuality is key; each piece of data should be validated for accuracy when possible. Implementing domain-specific annotations can add another layer of context, and incorporating a user feedback loop for continuous updates ensures the system adapts to real-world interactions. Time-sensitive topics require a mechanism to refresh outdated documents. Overall, the emphasis should be on clarity, context, and correctness to make the system efficient and reliable. Here is a list of best practices…\
对于高性能的 RAG 系统，数据需要干净、一致且上下文丰富。应标准化文本以删除特殊字符和不相关的信息，从而提高检索器的效率。为了保持一致性，应消除实体和术语的歧义，同时应消除重复或冗余信息，以简化检索器的焦点。真实性是关键;在可能的情况下，应验证每条数据的准确性。实现特定于域的注释可以添加另一层上下文，并结合用户反馈循环进行持续更新可确保系统适应现实世界的交互。对时间敏感的主题需要一种机制来刷新过时的文档。总体而言，重点应放在清晰度、上下文和正确性上，以使系统高效可靠。以下是最佳实践列表...

1. Text Cleaning: Standardize text format, remove special characters, and irrelevant information. This improves retriever efficiency and avoids garbage-in-garbage-out.\
   文本清理：规范文本格式，删除特殊字符和不相关信息。这提高了检索器的效率，避免了垃圾进垃圾出的情况。
2. Entity Resolution: Disambiguate entities and terms for consistent referencing. For example, standardize “ML,” “Machine Learning,” and “machine learning” to a common term.\
   实体解析：消除实体和术语的歧义，以实现一致的引用。例如，将“ML”、“机器学习”和“机器学习”标准化为通用术语。
3. Data Deduplication: Remove duplicate documents or redundant information to enhance retriever focus and efficiency.\
   重复数据删除：删除重复文档或冗余信息，以提高检索器的专注度和效率。
4. Document Segmentation: Break down long documents into manageable chunks, or conversely, combine small snippets into coherent documents to optimize retriever performance.\
   文档分段：将长文档分解为可管理的块，或者相反，将小片段组合成连贯的文档，以优化检索器性能。
5. Domain-Specific Annotations: Annotate documents with domain-specific tags or metadata. For instance, given your cloud tech focus, you could tag cloud-related technologies like “AWS,” “Azure,” etc.\
   特定于域的注释：使用特定于域的标记或元数据对文档进行注释。例如，鉴于您的云技术重点，您可以标记与云相关的技术，例如“AWS”、“Azure”等。
6. Data Augmentation: Use synonyms, paraphrasing, or even translation to/from other languages to increase the diversity of your corpus.\
   数据增强：使用同义词、释义，甚至与其他语言的相互翻译，以增加语料库的多样性。
7. Hierarchy & Relationships: Identify parent-child or sibling relationships between documents to improve contextual understanding.\
   层次结构和关系：识别文档之间的父子关系或兄弟姐妹关系，以提高上下文理解。
8. User Feedback Loop: Continuously update your database with new Q\&A pairs based on real-world interactions, marking them for factual correctness.\
   用户反馈循环：根据真实世界的交互，使用新的问答对不断更新您的数据库，并标记它们的事实正确性。
9. Time-Sensitive Data: For topics that are frequently updated, implement a mechanism to invalidate or update outdated documents.\
   时效性数据：对于经常更新的主题，实施一种机制来使过时的文档失效或更新。

### **2. Embeddings 嵌入** <a href="#id-48b3" id="id-48b3"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:700/1*Zj2vPYU6i31yTifI_Op4TA.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

OpenAI’s embeddings are fixed-size and non-fine-tunable. With fixed OpenAI embeddings, the emphasis would indeed be on optimising other parts of your RAG pipeline — like the retrieval mechanism or the quality of your data corpus — to ensure that you’re making the most out of the embeddings you have.\
OpenAI 的嵌入是固定大小且不可微调的。对于固定的 OpenAI 嵌入，重点确实是优化 RAG 管道的其他部分——例如检索机制或数据语料库的质量——以确保您充分利用您拥有的嵌入。

If your embeddings model is fine-tunable you may take advantage of fine-tuning the embedding model, dynamic embeddings or\
如果你的嵌入模型是微调的，你可以利用微调嵌入模型、动态嵌入或

* **fine-tuning embeddings (with fine-tunable/trainable embeddings)**\
  **微调嵌入（具有微调/可训练的嵌入）**

Fine-tuning embeddings within RAG has a direct bearing on its efficacy. By adapting embeddings to domain specifics, the retrieval step becomes sharper, ensuring the content fetched is highly relevant to the query. This fine-tuned retrieval acts as a more accurate foundation for the subsequent generation step. Especially in specialized domains, or when dealing with evolving or rare terms, these tailored embeddings are pivotal. In essence, for RAG, fine-tuning embeddings is akin to tuning the ears before letting the voice speak, ensuring what’s heard (retrieved) optimally influences what’s said (generated).\
RAG 中的微调嵌入对其功效有直接影响。通过根据域特定情况调整嵌入，检索步骤变得更加清晰，从而确保获取的内容与查询高度相关。这种微调的检索为后续生成步骤提供了更准确的基础。特别是在专业领域，或者在处理不断发展或罕见的术语时，这些量身定制的嵌入至关重要。从本质上讲，对于 RAG 来说，微调嵌入类似于在让声音说话之前调整耳朵，确保听到（检索）的内容以最佳方式影响所说的（生成）内容。

At the moment you cannot fine-tune ada-embedding-02. bge embedding models like bge-large-en; developed by the Beijing Academy of Artificial Intelligence (BAAI) are fine-tunable and high performant embedding models. You can use LLaMa Index to fine-tune bge embedding models \[[\*](https://gpt-index.readthedocs.io/en/stable/examples/finetuning/embeddings/finetune\_embedding.html)]. To create the training data to fine-tune bge models, you first create questions for your document chunks using an LLM like gpt-35-turbo. The question and the document chunk (the answer) become fine-tuning pairs for your fine-tuning.\
目前，您无法微调 ada-embedding-02。BGE 嵌入模型，如 BGE-large-EN;由北京人工智能研究院（BAAI）开发的是精细可调和高性能的嵌入模型。您可以使用 LLaMaIndex 微调 bge 嵌入模型 \[ \*]。要创建训练数据以微调 bge 模型，您首先使用LLM类似 gpt-35-turbo 的文档块创建问题。问题和文档块（答案）成为微调对，以便进行微调。

* **Dynamic embeddings (with fine-tunable/trainable embeddings)**\
  **动态嵌入（具有可微调/可训练的嵌入）**

Dynamic embeddings adjust according to the context in which a word appears, unlike static embeddings that represent each word with a single vector. For instance, in transformer models like BERT, the same word can have different embeddings depending on the surrounding words.\
动态嵌入会根据单词出现的上下文进行调整，这与使用单个向量表示每个单词的静态嵌入不同。例如，在像 BERT 这样的 transformer 模型中，同一个单词可以有不同的嵌入，具体取决于周围的单词。

There is also empirical evidence that OpenAI’s embeddings model text-embedding-ada-002 model gives unexpectedly high cosine similarity results when length of the text is shot e.g. <5 tokens. Ideally we should ensure the embeddings text will have as much context around it as possible so that embedding gives “healthy” results.\
还有经验证据表明，OpenAI 的嵌入模型 text-embedding-ada-002 模型在拍摄文本长度（例如 <5 个标记）时会给出出乎意料的高余弦相似度结果。理想情况下，我们应该确保嵌入文本周围有尽可能多的上下文，以便嵌入提供“健康”的结果。

OpenAI’s `embeddings-ada-02` model is based on the principles of large language models like GPT. It is more advanced than static embedding models and can capture some level of context. This means the embeddings it generates are influenced by the surrounding text to a certain degree. However, it’s important to note that while it captures context better than static models, it might not be as context-sensitive as the latest full-scale language models like GPT-4.\
OpenAI 的 `embeddings-ada-02` 模型基于 GPT 等大型语言模型的原理。它比静态嵌入模型更高级，可以捕获某种级别的上下文。这意味着它生成的嵌入在一定程度上受到周围文本的影响。然而，需要注意的是，虽然它比静态模型更好地捕捉上下文，但它可能不像 GPT-4 等最新的全尺寸语言模型那样对上下文敏感。

* **Refresh embeddings (with fine-tunable/trainable embeddings)**\
  **刷新嵌入（具有可微调/可训练的嵌入）**

The embeddings should also be periodically refreshed to capture evolving semantics in your corpus. The goal is to make them efficient for both retrieval and matching, ensuring a speedy and accurate RAG implementation.\
嵌入还应定期刷新，以捕获语料库中不断演变的语义。目标是使它们在检索和匹配方面都很高效，确保快速准确的 RAG 实施。

### **3. Retrieval  检索** <a href="#id-3f3e" id="id-3f3e"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:700/1*fn499Zb3k75Pc4CL0boIdg.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

To enhance retrieval efficiency in your RAG system, adopt a holistic strategy. Start by refining your chunking process, exploring various sizes to strike the right balance. Embed metadata for improved filtering capabilities and context enrichment. Embrace query routing across multiple indexes, catering to diverse query types. Consider Langchain’s multi-vector retrieval method, which employs smaller chunks, summary embeddings, and hypothetical questions to bolster retrieval accuracy. Address vector similarity issues with re-ranking, and experiment with hybrid search and recursive retrieval techniques for performance gains. Strategies like HyDE and iterative approaches such as “Read Retrieve Read” offer promising results. Lastly, fine-tune your vector search algorithm, optimizing the trade-off between accuracy and latency. This comprehensive approach ensures your RAG system excels in retrieving relevant and contextually rich information.\
为了提高 RAG 系统的检索效率，请采用整体策略。首先完善您的分块过程，探索各种尺寸以达到适当的平衡。嵌入元数据以改进过滤功能和上下文丰富。支持跨多个索引的查询路由，满足不同的查询类型。考虑 Langchain 的多向量检索方法，该方法采用较小的块、摘要嵌入和假设问题来提高检索准确性。通过重新排名解决向量相似性问题，并尝试使用混合搜索和递归检索技术来提高性能。像 HyDE 这样的策略和像“读取、检索、读取”这样的迭代方法提供了有希望的结果。最后，微调矢量搜索算法，优化准确性和延迟之间的权衡。这种全面的方法可确保您的 RAG 系统在检索相关且上下文丰富的信息方面表现出色。

* **Tune your chunking 调整分块**

Our aim is to collect as much relevant context and as little noise as possible. Chunk with small, medium & large size and use an evaluation framework like “LlamaIndex Response Evaluation” to decide on the optimal chunk size which uses GPT4 to evaluate faithfulness and relevancy to rate and compare seperate chunk sizes.\
我们的目标是收集尽可能多的相关上下文和尽可能少的噪音。具有小、中和大大小的块，并使用像“LlamaIndex Response Evaluation”这样的评估框架来决定最佳块大小，该框架使用 GPT4 来评估可靠性和相关性，以评估和比较单独的块大小。

When building a RAG system, always remember that chunk\_size is a pivotal parameter. Invest the time to meticulously evaluate and adjust your chunk size for unmatched results.\
在构建 RAG 系统时，请始终记住chunk\_size是一个关键参数。花时间仔细评估和调整您的区块大小，以获得无与伦比的结果。

LLaMA index has an automated evaluation capability for different chunking methods…(Evaluating the Ideal Chunk Size for a RAG System using LlamaIndex \[[link](broken-reference)]).\
LLaMA指数具有针对不同分块方法的自动评估功能...（使用 LlamaIndex \[链接]评估 RAG 系统的理想块大小）。

* **embed references (metadata) to your chunks —** such as date & use for filtering. Adding chapter, sub-chapter references might be helpful metadata to improve retrieval too.\
  将引用（元数据）嵌入到您的块中，例如用于过滤的日期和用途。添加章节、子章节引用也可能是有助于改进检索的元数据。
* **query routing over multiple indexes** — This works hands in hand with the previous approaches with metadata filtering and e.g. chunking. You may have different indexes and query them at the same time. If the query is a pointed query you may use your standard index or if it is a keyword search or filtering based on metadata such as a certain ‘date’ then you may use the relevant seperate index.\
  基于多个索引的查询路由 — 这与前面的元数据过滤和分块方法密切相关。您可能有不同的索引并同时查询它们。如果查询是指向查询，则可以使用标准索引，或者如果是基于元数据（例如某个“日期”）的关键字搜索或过滤，则可以使用相关的单独索引。

Langchain’s [_**multi-vector retrieval**_](https://python.langchain.com/docs/modules/data\_connection/retrievers/multi\_vector) is one such method. The methods to create multiple vectors per document include:\
Langchain的多向量检索就是这样一种方法。为每个文档创建多个向量的方法包括：

* Smaller chunks: split a document into smaller chunks, and embed those along with the longer chunks.\
  较小的块：将文档拆分为较小的块，并将这些块与较长的块一起嵌入。
* Add “summary embeddings” — create a summary for each document, embed that along with (or instead of) the document.\
  添加“摘要嵌入”——为每个文档创建一个摘要，将其与文档一起嵌入（或代替文档）。
* Hypothetical questions: create hypothetical questions that each document would be appropriate to answer, embed those along with (or instead of) the document.\
  假设性问题：创建每个文档都适合回答的假设性问题，将这些问题与文档一起嵌入（或代替文档）。
* **Re-ranking —** vector similiary search for embeddings might not interpret to semantic similarity . With rerenking your can address this discrepency.\
  重新排序 — 嵌入的向量模拟搜索可能无法解释为语义相似性。通过重新命名，您可以解决这种差异。
* **Explore hybrid search —** By intelligently blending techniques such as keyword-based search, semantic search, and vector search, you can harness the advantages of each approach. This approach allows your RAG system to adapt to varying query types and information needs, ensuring that it consistently retrieves the most relevant and contextually rich information. Hybrid search can be a powerful addition to your retrieval strategy, enhancing the overall performance of your RAG pipeline.\
  探索混合搜索 — 通过智能地混合基于关键字的搜索、语义搜索和矢量搜索等技术，您可以利用每种方法的优势。这种方法使 RAG 系统能够适应不同的查询类型和信息需求，确保它始终如一地检索最相关且上下文丰富的信息。混合搜索可以成为检索策略的有力补充，可增强 RAG 管道的整体性能。
* **Recursive retrieval & query engine —** Another powerful approach to optimize retrieval in your RAG system is to implement recursive retrieval and a sophisticated query engine. Recursive retrieval involves fetching smaller document chunks during initial retrieval to capture key semantic meaning. Later in the process, provide larger chunks with more contextual information to your Language Model (LM). This two-step retrieval method helps strike a balance between efficiency and context-rich responses.\
  递归检索和查询引擎 — 在RAG系统中优化检索的另一种强大方法是实现递归检索和复杂的查询引擎。递归检索涉及在初始检索期间获取较小的文档块以捕获关键语义含义。在此过程的后期，向语言模型 （LM） 提供更多上下文信息的较大块。这种两步检索方法有助于在效率和上下文丰富的响应之间取得平衡。

Complementing this strategy is a robust query engine. A well-designed query engine is essential for interpreting user queries effectively, especially when they involve nuanced or complex language. It enables your RAG system to iteratively evaluate the question for missing information, formulating a more comprehensive response once all relevant details are available.\
与此策略相辅相成的是强大的查询引擎。设计良好的查询引擎对于有效解释用户查询至关重要，尤其是当它们涉及细微或复杂的语言时。它使您的 RAG 系统能够迭代评估问题是否缺少信息，一旦所有相关细节可用，就会制定更全面的响应。

The combination of recursive retrieval and a smart query engine can significantly enhance the performance of your RAG system, ensuring that it retrieves not just relevant but contextually complete information for more accurate and informative answers.\
递归检索和智能查询引擎的结合可以显著提高 RAG 系统的性能，确保它不仅检索相关信息，而且检索上下文完整的信息，以获得更准确和信息丰富的答案。

* **HyDE**: [HyDE](http://boston.lti.cs.cmu.edu/luyug/HyDE/HyDE.pdf) is a strategy which takes a query, generates a hypothetical response, and then uses both for embedding look up. Researches have found this can dramatically improve performance.\
  HyDE：HyDE 是一种策略，它接受查询，生成假设响应，然后将两者用于嵌入查找。研究发现，这可以显着提高性能。
* “Read Retrieve Read” / ReAct , _iteratively evaluate the question for missing information and formulate a response once all information is available._\
  “Read Retrieve Read” / ReAct，迭代评估问题是否缺少信息，并在所有信息可用时制定响应。
* Parent Document Retriever , _fetch small chunks during retrieval to better capture semantic meaning, provide larger chunks with more context to your LLM_\
  父文档检索器，在检索过程中获取小块以更好地捕获语义含义，为更大的块提供更多上下文LLM
* **Vector Search —** Tune your vector search algorithm and parameters, _find the right balance between accuracy and latency._ When it comes to vector search in your RAG system, precision and speed are key. Start by fine-tuning the vector search algorithm and parameters, focusing on factors like the number of neighbors to search for and the distance metric used. The goal is to strike the right balance between accuracy and latency. Experiment with different configurations and benchmark their impact on retrieval efficiency.\
  矢量搜索 — 调整矢量搜索算法和参数，在准确性和延迟之间找到适当的平衡。当涉及到RAG系统中的矢量搜索时，精度和速度是关键。首先微调矢量搜索算法和参数，重点关注要搜索的邻居数量和使用的距离度量等因素。目标是在准确性和延迟之间取得适当的平衡。尝试不同的配置，并对它们对检索效率的影响进行基准测试。

Stay updated on the latest advancements in vector search algorithms and libraries, as new options frequently emerge. Additionally, consider implementing query batching to enhance search efficiency. By optimizing vector search, you ensure that your RAG system responds both accurately and swiftly to user queries, a critical factor for an efficient and responsive pipeline.\
随时了解矢量搜索算法和库的最新进展，因为新选项经常出现。此外，请考虑实现查询批处理以提高搜索效率。通过优化矢量搜索，您可以确保 RAG 系统准确、快速地响应用户查询，这是高效响应管道的关键因素。

### **4. Synthesis 合成** <a href="#id-09de" id="id-09de"></a>

‘Synthesis,’ explores advanced techniques to refine your RAG system. We delve into query transformations, the art of decomposing complex queries into manageable sub-queries, a proven strategy for enhancing Large Language Models’ (LLMs) effectiveness. Additionally, we address the critical aspect of engineering the base prompt, where prompt templating and conditioning play a pivotal role in tailoring your RAG system’s behavior to specific use-cases and contexts. Together, these strategies elevate the precision and efficiency of your RAG pipeline.\
“合成”探索先进的技术来完善您的 RAG 系统。我们深入研究了查询转换，这是将复杂查询分解为可管理的子查询的艺术，这是一种增强大型语言模型 （LLMs） 有效性的行之有效的策略。此外，我们还解决了设计基本提示的关键方面，其中提示模板和条件在根据特定用例和上下文定制 RAG 系统的行为方面起着关键作用。这些策略共同提升了 RAG 管道的精度和效率。

* **Query transformations** — Split complex questions into multiple questions (llamaindex). Sub-queries: LLMs tend to work better when they break down complex queries. You can build this into your RAG system such that a query is decomposed into multiple questions.\
  查询转换 — 将复杂问题拆分为多个问题 （llamaindex）。子查询：LLMs当它们分解复杂查询时，它们往往会更好地工作。您可以将其内置到 RAG 系统中，以便将查询分解为多个问题。
* **Engineer your base prompt**\
  **设计您的基本提示符**

Engineering the base prompt in a RAG system is crucial for guiding the model’s behavior. A two-fold approach: prompt templating and prompt conditioning.\
在 RAG 系统中设计基本提示对于指导模型的行为至关重要。双重方法：提示模板和提示条件反射。

1. Prompt Templating: Define a template that captures the essence of the query and context, keeping in mind the specific use-case. For instance, if you’re building a tech support bot, the template might look like: “Help the user resolve issue: {issue\_description}. Consider these documents: {document\_snippets}.”\
   提示模板：定义一个模板，该模板捕获查询和上下文的本质，同时牢记特定用例。例如，如果要构建技术支持机器人，则模板可能如下所示：“帮助用户解决问题：{issue\_description}。请考虑以下文档：{document\_snippets}。
2. Prompt Conditioning: You can also condition the model by adding a prefix that sets the context or instructs the model to answer in a certain way. Given your interest in machine learning, for example, you could prepend with “Using your understanding of machine learning and cloud technologies, answer the following:”\
   提示条件反射：您还可以通过添加前缀来调节模型，该前缀设置上下文或指示模型以某种方式回答。例如，鉴于你对机器学习感兴趣，你可以在前面加上“利用你对机器学习和云技术的理解，回答以下问题：”

Here’s a simplified Python example for creating such a prompt, assuming you’ve got your query and retrieved documents:\
下面是一个简化的 Python 示例，用于创建此类提示，假设您已经获得了查询并检索到的文档：

```
# Your question and retrieved documents
question = "What is the best ML algorithm for classification?"
retrieved_docs = ["Doc1: SVM is widely used...", "Doc2: Random Forest is robust..."]

# Template
template = "Help answer the following question based on these documents: {question}. Consider these documents: {docs}"

# Construct the full prompt
full_prompt = template.format(question=question, docs=" ".join(retrieved_docs))

# Now, this `full_prompt` can be fed into the RAG generator
```

As additional improvements consider fine-tuning the base model and use function calling…\
作为其他改进，请考虑微调基本模型并使用函数调用...

## Fine-tuning & RAG 微调 & RAG <a href="#id-1458" id="id-1458"></a>

<figure><img src="https://miro.medium.com/v2/resize:fit:700/1*dlbZpWvOE17vdE-cTMK_1Q.png" alt="" height="263" width="700"><figcaption></figcaption></figure>

Fine-tuning just the generator in a RAG setup (e.g. fine-tuning the base gpt model the context and the prompt is being sent to) aims to improve the language generation component without touching the retrieval part. Doing this can have several benefits:\
在 RAG 设置中仅微调生成器（例如，微调基本 gpt 模型、上下文和发送到的提示）旨在改进语言生成组件，而无需触及检索部分。这样做有几个好处：

1. Answer Quality: Directly improves how well the generator formulates answers.\
   答案质量：直接提高生成器制定答案的能力。
2. Contextual Understanding: Fine-tuning on domain-specific datasets can help the generator understand the context provided by the retriever more accurately.\
   上下文理解：对特定领域的数据集进行微调可以帮助生成器更准确地理解检索器提供的上下文。
3. Speed: It can make the generator more efficient, thus speeding up the entire RAG operation.\
   速度：可以使发电机效率更高，从而加快整个RAG的运行速度。

## Function calling & RAG 函数调用 & RAG <a href="#id-23a1" id="id-23a1"></a>

The function calling feature can significantly enhance Retrieval-Augmented Generation (RAG) pipelines by introducing structured, actionable output during the generation step. This allows for real-time API integrations for up-to-date answers, optimized query execution to reduce errors, and modular retrieval methods for improved relevance. It can also facilitate a feedback loop for dynamic document fetching and offer structured JSON output for multi-step reasoning or data aggregation. Overall, it makes the RAG system more dynamic, accurate, and responsive.\
函数调用功能可以通过在生成步骤中引入结构化的、可操作的输出来显著增强检索增强生成 （RAG） 管道。这允许实时 API 集成以获得最新答案，优化查询执行以减少错误，以及模块化检索方法以提高相关性。它还可以促进动态文档获取的反馈循环，并为多步骤推理或数据聚合提供结构化的 JSON 输出。总体而言，它使 RAG 系统更加动态、准确和响应迅速。

## _Conclusion 结论_ <a href="#id-2854" id="id-2854"></a>

In the world of RAG systems, optimizing performance is an ongoing journey. By carefully managing data, fine-tuning embeddings, enhancing retrieval strategies, and utilizing advanced synthesis techniques, you can push the boundaries of what your RAG application can achieve. Stay curious, innovative, and adaptive in this ever-evolving landscape.\
在 RAG 系统领域，优化性能是一个持续的旅程。通过仔细管理数据、微调嵌入、增强检索策略以及利用先进的综合技术，您可以突破 RAG 应用程序所能实现的界限。在这个不断变化的环境中保持好奇心、创新性和适应性。
