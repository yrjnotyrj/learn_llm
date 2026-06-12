# learn_llm

大语言模型（LLM）学习记录。

---

## 项目内容

目前主要聚焦于 Transformer 架构的核心组件与常用函数实现：

### 1. 注意力机制 (Attention)
- **MHA (Multi-Head Attention)**：标准多头注意力
- **MQA (Multi-Query Attention)**：单键值多头注意力优化
- **GQA (Grouped-Query Attention)**：分组键值多头注意力
- **MLA (Multi-Head Latent Attention)**：带隐向量压缩的注意力机制

### 2. 前馈网络 (FFN)
- Transformer 标准 Feed-Forward Network 实现
- MLP、MOE

### 3. 常用函数
- Softmax
- SiLU
- 其他 Transformer 相关辅助函数
