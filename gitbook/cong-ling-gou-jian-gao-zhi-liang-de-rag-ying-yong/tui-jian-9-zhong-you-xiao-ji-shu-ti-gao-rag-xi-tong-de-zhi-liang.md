---
description: >-
  这是来自同济的一篇论文。它涵盖了关于RAG框架及其局限性的所有你需要知道的内容。它还列出了一些现代技术，用以提升其在检索、增强和生成方面的性能。
  这些技术背后的终极目标是使这个框架准备好进行可扩展性和生产使用，特别是对于那些答案质量*非常*重要的用例和行业。
  我不会在这篇文章中讨论所有内容，但以下是我认为会使你的RAG更高效的关键思路。
---

# \[推荐]9种有效技术提高RAG系统的质量

<mark style="color:green;">感谢您关注我们对检索增强生成（RAG）系统的概述。我们希望这份指南能够揭示RAG的复杂工作原理，并展示其在不同环境下革新信息检索和响应生成的潜力。</mark>

<mark style="color:green;">我们已经建立了将RAG系统应用于生产环境的丰富经验。我们的专业知识涵盖了评估组织需求、部署定制化的、高性能的RAG解决方案。</mark>

<mark style="color:green;">如果您正在考虑将RAG系统融入公司运营中，并需要专业指导以确保成功，我们愿意提供帮助。我们将协助您在这个充满活力的技术领域中导航，以全新的方式利用检索增强生成（RAG）解锁您组织的集体知识潜力。</mark>

<mark style="color:green;">可以添加我的微信（备注RAG），咨询构建高质量申请加入我们的LLM+RAG高可用技术群！</mark>

<figure><img src="../.gitbook/assets/WechatIMG255.jpg" alt="" width="375"><figcaption></figcaption></figure>





阅读论文 [https://arxiv.org/pdf/2312.10997.pdf](https://arxiv.org/pdf/2312.10997.pdf)

### 1—🗃️ **Enhance the quality of indexed data 提高索引数据的质量（反复提及）**

As the data we index determines the quality of the RAG’s answers, the first task is to curate it as much as possible before ingesting it. _(**Garbage in, garbage out still applies here**)_\
You can do this by removing duplicate/redundant information, spotting irrelevant documents, and checking for fact accuracy (if possible).\
If the maintainability of the RAG matters, you also need to add mechanisms to refresh outdated documents.\
\
Cleaning the data is a step that is often neglected when building RAGs, as we usually tend to pour in all the documents we have without verifying their quality.

Here are some quick fixes that I suggest you go through:

* Remove text noise by cleaning special characters, weird encodings, unnecessary HTML tags… Remember that old NLP techniques using regex? You can reuse them.
* Spot document outliers that are irrelevant to the main topics and remove them. You can do this by implementing some topic extraction, dimensionality reduction techniques, and data visualization.
* Remove redundant documents by using similarity metrics

### 2—🛠️ **Optimize index structure 优化索引结构**

When constructing your RAG, the chunk size is a key parameter. It determines the length of the documents we retrieve from the vector store.\
**A small chunk size might result in documents that miss some crucial information while a large chunk size can introduce irrelevant noise**.

Coming up with the optimal chunk size is about finding the right balance.\
\
**How to do that efficiently? Trial and error.**

However, this doesn’t mean that you have to make some random guesses and perform qualitative assessments for every experience.

You can find that optimal chunk size by running evaluations on a test set and computing metrics. LlamaIndex has interesting features to do this. You can read more about that in their [blog](https://blog.llamaindex.ai/evaluating-the-ideal-chunk-size-for-a-rag-system-using-llamaindex-6207e5d3fec5).

### **3—🏷️ Add metadata 添加元数据**

Incorporating metadata with the indexed vectors helps better structure them while improving search relevance.

Here are some scenarios where metadata is useful:

* If you search for items and recency is a criterion, you can sort over a date metadata
* If you search over scientific papers and you know in advance that the information you’re looking for is always located in a specific section, say the experiment section for example, you can add the article section as metadata for each chunk and filter on it to match experiments only

Metadata is useful because it brings an additional layer of structured search on top vector search.

### 4—↔️ Align the input query with the documents **使输入查询与文档对齐**

LLMs and RAGs are powerful because they offer the flexibility of expressing your query in natural language, thus lowering the entry barrier for data exploration and more complex tasks. (Learn [here](https://thetechbuffet.substack.com/p/turn-complex-english-instructions-into-sql) how RAGs can help generate SQL code from plain English instructions).

LLM和RAG之所以强大，是因为它们提供了用自然语言表达查询的灵活性，从而降低了数据探索和更复杂任务的入门门槛。（在这里了解RAG如何帮助从简单的英文指令生成SQL代码）。

Sometimes, however, a misalignment appears between the input query the user formulates in the form of a few words or a short sentence, and the indexed documents, which are often written in the form of longer sentences or even paragraphs.

然而，有时用户以几个词或一个短句形式提出的输入查询与索引文档之间会出现不对齐的情况，而索引文档通常以较长的句子甚至段落的形式编写。&#x20;

Let’s go through an example to understand this.

Here’s a paragraph about the motor engine:

> _The motor engine stands as an engineering marvel, propelling countless vehicles and machinery with its intricate design and mechanical prowess. At its core, a motor engine converts fuel into mechanical energy through a precisely orchestrated series of combustion events. This process involves the synchronized movement of pistons, a crankshaft, and a complex network of valves, all carefully calibrated to optimize efficiency and power output. Modern motor engines come in various types, such as internal combustion engines and electric motors, each with its unique set of advantages and applications. The relentless pursuit of innovation continues to enhance motor engine technology, pushing the boundaries of performance, fuel efficiency, and environmental sustainability. Whether powering a car on the open road or driving industrial machinery, the motor engine remains a driving force behind the dynamic movement of our modern world._

If you formulate a simple query such as **“Can you tell how the motor engine works in a nutshell?**” and compute its cosine similarity with the paragraph you obtain the value of `0.72`

Not bad, but can we do better?

To do that, we will no longer index the paragraph by its embedding but with the embeddings of the questions it answers instead.

Let’s consider these three questions the paragraph answers.

1. What is the fundamental function of a motor engine?",
2. How does a motor engine convert fuel into mechanical energy?",
3. What are some key components involved in the operation of a motor engine, and how do they contribute to its efficiency?"

If we compute their similarity with the input query, we obtain these values, respectively.

1. **0.864**
2. 0.841
3. 0.845

These values are higher and indicate that the input query matches the questions more precisely.

Indexing the chunks with the questions they answer changes the problem slightly but helps address alignment problems and increases search relevance: we don’t optimize for the similarity with the documents but with underlying questions.

这些值更高，表明输入查询与问题的匹配更加精确。用它们回答的问题对块进行索引虽然略微改变了问题，但有助于解决对齐问题并增加搜索相关性：我们不是优化与文档的相似性，而是与底层问题的相似性。

### 5—🔍 Mixed retrieval **混合检索**

While vector search helps retrieve semantically relevant chunks for a given query, it occasionally lacks precision in matching specific keywords.

Depending on the use case, an exact match can be sometimes necessary.

Imagine searching a vector database of millions of e-commerce products, and to the query “Adidas ref XYZ sneakers white” the top results include white Adidas sneakers with none matching the exact XYZ reference.

This would be rather disappointing.

To tackle this issue, mixed retrieval is a solution. This strategy leverages the strengths of different retrieval technologies such as vector search and keyword search and combines them intelligently.

With this hybrid approach, you can still match relevant keywords while maintaining control over the query intent.

Check Pinecone’s starter [guide](https://www.pinecone.io/learn/hybrid-search-intro/) to learn more about hybrid search.

### 6—🔄 ReRank **重新排序**

When you query the vectorstore, the top K results are not necessarily ordered in the most relevant way. Granted they’re all relevant but the most relevant chunk among these relevant chunks can be number #5 or #7 instead of #1 or #2

Here’s where ReRank comes in.

The straightforward concept of re-ranking to relocate the most pertinent information toward the edges of the prompt has been successfully implemented in various frameworks, including LlamaIndex, LangChain, and HayStack.

For example, Diversity Ranker focuses on reordering based on document diversity, while LostInTheMiddleRanker alternates the placement of the best document at both the beginning and end of the context window.

### 7—🗜️ Prompt compression **提示压缩**

It’s been shown that noise in the retrieved contexts adversely affects the RAG performance, more precisely, the answer generated by the LLM.

Some suggested techniques apply a post-processing step after retrieval to compress irrelevant context, highlight important paragraphs, and reduce the overall context length.

Approaches like Selective Context \[[_Litman et. al_](https://arxiv.org/abs/2003.11288)] and LLMLingua \[[Aderson et. al](https://arxiv.org/abs/2310.05736)] use small LLMs to calculate prompt mutual information or perplexity, thus estimating element importance.

### 8—💡 HyDE

This method comes from this [paper](https://arxiv.org/abs/2212.10496) and stands for **Hy**pothetical **D**ocument **E**mbedding.

When presented with a query, HyDE instructs an LLM to generate a hypothetical answer.

This document, while capturing relevance patterns, is not real and may contain inaccuracies. Subsequently, an unsupervised contrastively learned encoder (e.g., Contriever) transforms the document into an embedding vector.

This vector serves to pinpoint a neighborhood in the corpus embedding space, enabling the retrieval of similar real documents based on vector similarity. In this second step, the generated document is anchored to the actual corpus, with the encoder's dense bottleneck effectively filtering out incorrect details.

Experiments demonstrate that HyDE consistently outperforms the state-of-the-art unsupervised dense retriever Contriever and exhibits robust performance comparable to fine-tuned retrievers across various tasks (such as web search, QA, and fact verification) and languages.

### 9—✒️ Query rewrite and expansion **查询重写和扩展**

When a user interacts with a RAG, his query is not necessarily well-formulated and doesn't fully express an intent that can be efficiently matched with documents in the vectorstore.

To solve this issue, we instruct an LLM to rewrite this query behind the scenes before sending it to the RAG.

This can be easily implemented by adding intermediate LLM calls but other sophisticated techniques (reviewed in this [paper](https://arxiv.org/abs/2305.03653)) exist.

***

## Appendix 教程**补充**

LlamaIndex offers a range of solutions that can address several of the 9 techniques for enhancing Retrieval Augmented Generation (RAG) systems. Here's how LlamaIndex can help:

1. **Data Quality and Indexing**: LlamaIndex provides tools for curating and cleaning data before indexing, which is crucial for maintaining high-quality RAG systems. The platform allows for the removal of duplicate, redundant, or irrelevant information, ensuring that the indexed data is as relevant and up-to-date as possible.
2. **Optimal Chunking Strategy**: The platform offers various node parsers like SentenceSplitter and CodeSplitter, which are essential for breaking down documents into manageable chunks. This facilitates better handling and processing of documents, allowing for more effective indexing and retrieval.
3. **Vector Store Optimization**: LlamaIndex supports the selection of an appropriate vector store, a critical component for storing and retrieving embeddings in the RAG pipeline. This ensures efficient management of vector data, which is key to the performance of RAG systems.
4. **Response Synthesis**: The platform provides various response synthesis methods, such as refine and tree\_summarize, allowing users to choose how the RAG pipeline synthesizes responses. This flexibility can enhance the relevance and accuracy of the generated answers.

LlamaIndex's comprehensive suite of tools and features makes it a valuable asset for building and optimizing RAG systems, particularly for applications where answer quality is paramount. The platform's focus on data quality, effective chunking, vector store optimization, and flexible response synthesis aligns well with the key techniques for boosting RAG systems.

For more detailed information on how LlamaIndex supports these aspects, you can visit their official documentation and guides:

* For an overview of evaluating RAG systems using LlamaIndex: [LlamaIndex Blog](https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93)
* For details on building a no-code RAG pipeline with LlamaIndex: [RAGArch by LlamaIndex](https://www.llamaindex.ai)

LlamaIndex为提升检索增强生成（RAG）系统提供了多种解决方案，可以应对前面提到的9种增强技术中的几种。以下是LlamaIndex如何帮助的一些方式：

1. **数据质量和索引**：LlamaIndex提供了在索引之前策划和清理数据的工具，这对于维护高质量的RAG系统至关重要。平台允许去除重复、冗余或不相关的信息，确保索引数据尽可能相关和最新。
2. **最佳分块策略**：该平台提供了各种节点解析器，如SentenceSplitter和CodeSplitter，这些对于将文档分解成易于管理的块至关重要。这有助于更好地处理和处理文档，允许更有效的索引和检索。
3. **向量存储优化**：LlamaIndex支持选择适当的向量存储，这是存储和检索RAG管道中嵌入的关键组件。这确保了向量数据的有效管理，这是RAG系统性能的关键。
4. **响应合成**：该平台提供了各种响应合成方法，如refine和tree\_summarize，允许用户选择RAG管道如何合成响应。这种灵活性可以增强生成答案的相关性和准确性。

LlamaIndex的全面工具和功能套件使其成为构建和优化RAG系统的宝贵资产，特别是对于答案质量至关重要的应用。平台对数据质量、有效分块、向量存储优化和灵活响应合成的关注与提升RAG系统的关键技术相吻合。

有关LlamaIndex如何支持这些方面的更多详细信息，您可以访问官方文档和指南：

* 有关使用LlamaIndex评估RAG系统的概览，请访问：[LlamaIndex博客](https://www.llamaindex.ai/blog/openai-cookbook-evaluating-rag-systems-fe393c61fb93)
* 有关使用LlamaIndex构建无代码RAG管道的详细信息，请访问：[LlamaIndex的RAGArch](https://www.llamaindex.ai)。

关于如何构建高质量RAG应用，可以添加我的微信（进入高质量RAG产品技术群）获取最新信息。

<figure><img src="../.gitbook/assets/WechatIMG255.jpg" alt=""><figcaption></figcaption></figure>
