# bge-m3 本地模型目录说明

本目录用于保存本地 bge-m3 embedding 模型文件。

该模型用于把问题和知识块编码为向量，供 Qdrant 向量检索使用。

正式运行索引构建前，请确认本目录中的模型文件完整可用，尤其是模型权重、分词器配置和 sentence-transformers 配置。
