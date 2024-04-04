---
description: Pipelienne
---

# \[索引相关] 通过这三种索引方法改善 RAG Pipeline

In this issue, we’ll explore 3 different ways you can index your data when building RAGs.

These methods are worth experimenting with as they can boost the **performance** and **accuracy**.

Let’s see how.



<figure><img src="../.gitbook/assets/f985c725-64d6-4791-93b5-8f4f52c24347_1075x934.jpg" alt=""><figcaption></figcaption></figure>

### Reminder on data indexing in typical RAGs

In the default implementation of a RAG system, documents are first split into chunks.

Then, each chunk is embedded and indexed into a vector database.

In the retrieval step, the input query is also embedded and the most similar chunks are extracted.

\
In this setup, the data (i.e. _the chunks_) we retrieve is the same as the data we index.

This is the most natural and intuitive implementation.

🚨 However, this doesn’t have to be always the case.

We can index the chunks differently to increase the performance of such systems



