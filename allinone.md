# LongLLMLingua 论文阅读与项目实现分析

## 检索结果

检索式：`LongLLMLingua Accelerating Enhancing LLMs Long Context Prompt Compression`；年份范围：2023–2026。

来源命中：Semantic Scholar=0、OpenAlex=10、arXiv=0、OpenReview=0、Crossref=10、DBLP=0；共 14 篇唯一论文，合并 6 条跨来源重复记录。由于当前检索脚本不支持 `--json`，无法取得全部候选论文摘要进行可靠的语义过滤，因此保留全部结果。

检索错误：
- arXiv：`429 Client Error`
- DBLP：`ConnectionResetError(54, 'Connection reset by peer')`
- OpenReview：`openreview not installed. pip install openreview-py`
- Semantic Scholar：`429 Client Error`

| # | Title | Year | Venue | Citations | Score | Sources |
|---|---|---:|---|---:|---:|---|
| [1](https://doi.org/10.18653/v1/2024.acl-long.91) | LongLLMLingua: Accelerating and Enhancing LLMs in Long Context Scenarios via Prompt Compression | 2024 | ACL Long Papers | 93 | 8 | OpenAlex, Crossref |
| [2](https://doi.org/10.2139/ssrn.6970438) | Large-Scale Evaluation of MaxEntRAG-Flow: Incremental Evidence Structures for Real-Time Context Compression and Joint Probabilistic Graph Retrieval in Long-Context LLMs | 未知 | SSRN | 0 | 6 | Crossref |
| [3](https://doi.org/10.48550/arxiv.2401.03462) | Long Context Compression with Activation Beacon | 2024 | arXiv | 2 | 5 | OpenAlex |
| [4](https://doi.org/10.21203/rs.3.rs-10458568/v1) | Graph-Aware Reinforcement Learning for Reusable Prompt Compression in Black-Box LLMs | 未知 | Research Square | 0 | 5 | Crossref |
| [5](https://doi.org/10.18653/v1/2023.emnlp-main.825) | LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models | 2023 | EMNLP | 132 | 4 | OpenAlex |
| [6](https://doi.org/10.1145/3677779.3677794) | Adapting LLMs for Efficient Context Processing through Soft Prompt Compression | 2024 | ACM | 42 | 4 | OpenAlex |
| [7](https://doi.org/10.18653/v1/2024.findings-acl.306) | Extending Context Window of Large Language Models via Semantic Compression | 2024 | Findings of ACL | 18 | 4 | OpenAlex |
| [8](https://doi.org/10.1145/3767695.3769499) | ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs | 2025 | ACM | 0 | 4 | OpenAlex |
| [9](https://doi.org/10.18653/v1/2025.naacl-long.524) | LCIRC: A Recurrent Compression Approach for Efficient Long-form Context and Query Dependent Modeling in LLMs | 2025 | NAACL | 0 | 4 | OpenAlex |
| [10](https://openalex.org/W6979262503) | QwenLong-CPRS: Towards ∞-LLMs with Dynamic Context Optimization | 2025 | arXiv | 0 | 4 | OpenAlex |
| [11](https://doi.org/10.32388/kpa6vi) | Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism | 未知 | 未知 | 0 | 4 | Crossref |
| [12](https://doi.org/10.18653/v1/2026.findings-acl.1454) | Embedding-based In-Context Prompt Training for Enhancing LLMs as Text Encoders | 2026 | Findings of ACL | 0 | 4 | Crossref |
| [13](https://doi.org/10.21203/rs.3.rs-10952127/v1) | Reasoning-Aware Error-Bounded KV-Cache Compression and Sparse Attention for Long-Context LLMs | 未知 | Research Square | 0 | 4 | Crossref |
| [14](https://doi.org/10.32388/85aljn) | [survey] Review of: “Enhancing Long Context Performance in LLMs Through Inner Loop Query Mechanism” | 2025 | Review | 0 | 4 | Crossref |

未在 API 结果中重复的模型知识补充：

| # | Title | Year | Venue | Notes |
|---|---|---:|---|---|
| [M1](https://scholar.google.com/scholar?q=Compressing+Context+to+Enhance+Inference+Efficiency+of+Large+Language+Models) | Compressing Context to Enhance Inference Efficiency of Large Language Models | 2023 | EMNLP | Selective Context，以信息熵删除冗余词元 |
| [M2](https://scholar.google.com/scholar?q=LLMLingua-2+Data+Distillation+for+Efficient+and+Faithful+Task-Agnostic+Prompt+Compression) | LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression | 2024 | Findings of ACL | 蒸馏式 token classification，速度更高但方法不同 |
| [M3](https://scholar.google.com/scholar?q=Adapting+Language+Models+to+Compress+Contexts) | Adapting Language Models to Compress Contexts | 2023 | EMNLP | AutoCompressor，软提示/摘要向量路线 |
| [M4](https://scholar.google.com/scholar?q=Learning+to+Compress+Prompts+with+Gist+Tokens) | Learning to Compress Prompts with Gist Tokens | 2023 | NeurIPS | Gist token 软压缩，需要训练 |
| [M5](https://scholar.google.com/scholar?q=In-Context+Autoencoder+for+Context+Compression+in+a+Large+Language+Model) | In-Context Autoencoder for Context Compression in a Large Language Model | 2024 | ICLR | ICAE，需训练的上下文编码路线 |
| [M6](https://scholar.google.com/scholar?q=RECOMP+Improving+Retrieval-Augmented+LMs+with+Compression+and+Selective+Augmentation) | RECOMP: Improving Retrieval-Augmented LMs with Compression and Selective Augmentation | 2024 | ICLR | RAG 场景中的抽取式与生成式压缩 |

## 检索综述

### Overview
结果覆盖 2023–2026 年的 prompt compression、上下文压缩、软提示压缩、KV-cache 压缩和长上下文检索。LongLLMLingua 与其前作 LLMLingua 是本任务最直接的两篇论文。

### Trends
- 2023 年主要是基于 perplexity/entropy 的无训练 token 删除，以及需要训练的软提示压缩。
- 2024 年开始强调 query-aware、语义压缩与 task-agnostic 蒸馏。
- 2025 年进一步融合动态优化、图检索、强化学习和 KV-cache 压缩。
- 方法由“统一删除低信息 token”转向“按问题、文档、任务动态分配预算”。

### Key themes
1. 无训练式 token pruning：用小型 causal LM 的 NLL/perplexity 判断 token 重要性（[1]、[5]、[M1]）。
2. Query-aware 压缩：由问题控制文档排序与 token 保留（[1]、[8]、[9]）。
3. 软提示与隐藏状态压缩：将长文本编码成少量连续表示（[3]、[6]、[M3]、[M4]、[M5]）。
4. 检索与压缩协同：先筛选文档，再进行细粒度压缩（[1]、[2]、[M6]）。
5. Prompt 与 KV-cache 联合优化：减少 prefill 输入，同时控制 decoding cache（[10]、[13]）。

### Keywords frequency

| Keyword | Count |
|---|---:|
| Compression | 11 |
| Long Context | 8 |
| LLM | 10 |
| Prompt | 6 |
| Efficient/Accelerating | 6 |

### Most cited by accepted paper

| Rank | Title | Year | Citations |
|---:|---|---:|---:|
| 1 | LLMLingua | 2023 | 132 |
| 2 | LongLLMLingua | 2024 | 93 |
| 3 | Adapting LLMs for Efficient Context Processing through Soft Prompt Compression | 2024 | 42 |
| 4 | Extending Context Window of Large Language Models via Semantic Compression | 2024 | 18 |
| 5 | Long Context Compression with Activation Beacon | 2024 | 2 |

### Most cited by first author

| Rank | Author | Papers in set | Total citations |
|---:|---|---:|---:|
| 1 | Huiqiang Jiang | 2 | 225 |
| 2 | Cangqing Wang | 1 | 42 |
| 3 | Weizhi Fei | 1 | 18 |
| 4 | Peitian Zhang | 1 | 2 |
| 5 | Xuancheng Li | 1 | 0 |

### Recommendations for reading
1. LLMLingua：理解 coarse-to-fine、budget controller 与 iterative token compression 的基础。
2. LongLLMLingua：理解 question-aware ranking、contrastive perplexity、动态预算和恢复机制。
3. Selective Context：作为不依赖问题的 entropy pruning 对照基线。
4. LLMLingua-2：若更重视部署速度，可比较蒸馏式 token classifier。
5. QwenLong-CPRS：用于后续探索 prompt 与 KV-cache 联合压缩。

## LongLLMLingua 核心方法

论文将输入表示为 instruction、多个 document 和 question。目标是在压缩后的 prompt 长度与目标模型输出偏差之间折中，并允许重新排列文档。

### 1. Question-aware coarse-grained compression
对每个文档计算问题在该文档条件下的平均 NLL：

`r_k = -(1/N_c) * Σ log p(question_i, restrict_i | document_k)`

其中附加限制句为：`We can get the answer to this question in the given documents.`。NLL 越低，文档与问题越相关。按相关性选择文档直到达到粗粒度 token budget。

### 2. Document reordering
依据相关性重新排列保留文档，将关键内容放到长上下文模型更敏感的位置，以缓解 lost-in-the-middle。官方实现中 `reorder_context="sort"` 实际保留相关性排序；`two_stage` 则交替放置到两端。

### 3. Question-aware fine-grained compression
对文档 token 计算 contrastive perplexity：

`s_i = NLL(x_i | x_<i) - NLL(x_i | question, x_<i)`

问题使某 token 的预测明显变容易时，`s_i` 较大，说明该 token 与问题更相关，应优先保留。论文将其解释为 conditional pointwise mutual information 的对应量。

### 4. Dynamic compression ratio
相关文档获得更高保留率，不相关文档获得更低保留率。官方线性调度以全局预算 `τ_doc` 为中心，按文档排名施加 `±δτ` 偏移；论文默认 `δτ=0.3`。

### 5. Iterative token compression
以固定窗口逐段计算 token loss、估计分位数阈值并删除低分 token；论文默认 `iterative_size=200`。这样避免一次性对超长上下文执行完整前向。

### 6. Subsequence recovery
生成后，将响应中来自压缩 prompt 的最长片段映射回原 prompt 的最短对应子序列，用原始实体字符串替换压缩后残缺的名称、数字和地点。

### 论文默认配置与结果
- 压缩模型：主要使用 LLaMA-2-7B-Chat；另有 GPT2-small 消融。
- `iterative_size=200`
- `τ_instruction=0.85`
- `τ_question=0.9`
- `δτ=0.3`
- coarse-to-fine granular coefficient `k=2`
- NaturalQuestions 4× 约束下，LongLLMLingua 明显优于 LLMLingua，并基本保持原 prompt 表现。
- LongBench 约 6× 压缩时，论文报告平均分 48.3，高于原 prompt 的 44.0。
- 端到端延迟最高约 2.6× 加速；成本降低取决于数据集，论文报告约 52.6%–94.0%。
- 消融表明 question-aware coarse filtering 是主要贡献；fine-grained、动态预算、reordering 和 recovery 均有增益。

## 在当前项目中的实现判断

### 关键结论
LongLLMLingua 是输入级 prompt 压缩，当前项目的 H2O/StreamingLLM/LookM 是推理期 KV-cache 压缩。两者作用阶段不同，可以单独评测，也可以叠加：

`原始 prompt → LongLLMLingua → chat template → prefill → KV-cache compression → decoding → recovery`

不建议直接沿用 `Qwen35.process()` 中现有的 `input_compressor.compress(input_ids)` 形态。官方 LongLLMLingua 需要结构化的 `context: List[str]` 和独立 `question`，并返回压缩文本；如果先套 chat template 再压 token IDs，会压坏 system/user/control tokens，也无法完成 question-aware scoring。

### 推荐模块结构
新增文本级压缩层，而不是修改模型内部 attention：

```text
compression/
  __init__.py
  base.py                 # PromptCompressor Protocol、CompressionResult
  longllmlingua.py        # 官方库适配器或本地实现
  needle_parser.py        # 将现有 needle prompt 拆成 instruction/context/question
```

建议结果结构：

```python
@dataclass
class CompressionResult:
    compressed_prompt: str
    original_prompt: str
    question: str
    origin_tokens: int
    compressed_tokens: int
    ratio: float
    latency_seconds: float
    metadata: dict
```

### 数据流改造
1. `main.py::parse_args()` 增加：
   - `--prompt_compression {none,longllmlingua}`
   - `--compression_rate`，默认先用 `0.5`
   - `--compressor_model_path`
   - `--compressor_device`
   - `--reorder_context {original,sort,two_stage}`
   - `--dynamic_context_compression_ratio`，默认 `0.3`
   - `--compression_chunk_tokens`
   - `--enable_recovery`
2. `main.py::build_model()` 构造一次 compressor，并注入模型，避免每个样本重复加载。
3. `main.py::inference()` 记录压缩 token 数、压缩耗时、prefill/生成耗时和压缩配置。
4. `Qwen35.process()` 在 `apply_chat_template()` 之前压缩纯文本，再进行现有生成流程。
5. 生成后可调用 `recover(original_prompt, compressed_prompt, response)`，同时保留 raw response 方便评测恢复收益。

### Needle prompt 解析
当前数据只有一个完整 `origin_prompt[-1]['prompt']` 字符串。它稳定包含：
- instruction 结束标记：`The document given to you by the user is \n`
- question 开始标记：`\n\nNow, the question is:`

MVP 可按这两个锚点拆出 instruction、document、question。为 coarse filtering 生效，应将 document 进一步按段落或 tokenizer 固定窗口切成 `List[str]`；单元素 context 不会触发官方 context-level filtering。

推荐 token-aware 分块：每块约 256–512 compressor tokens，优先在双换行、句号或换行处切分；保留块的原始次序与字符偏移，以支持 recovery 和诊断。

### 两种实现路线

#### 路线 A：先接官方 `llmlingua` 包
优点：最快建立可信基线，参数与论文一致，包含 `recover()`。缺点：默认 LLaMA-2-7B-Chat 占用大，官方实现内部仍使用 legacy tuple/list KV-cache 操作，可能与本项目 `transformers>=5.0.0` 不兼容。

调用核心参数：

```python
PromptCompressor(...).compress_prompt(
    context_chunks,
    instruction=instruction,
    question=question,
    rate=0.5,
    iterative_size=200,
    condition_in_question="after_condition",
    reorder_context="sort",
    dynamic_context_compression_ratio=0.3,
    condition_compare=True,
    context_budget="+100",
    rank_method="longllmlingua",
)
```

依赖需加入 `llmlingua`，其核心额外依赖为 `tiktoken`、`nltk`。建议先在独立兼容环境验证，避免直接破坏当前 Transformers 5.x/Qwen3.5 环境。

#### 路线 B：在项目内原生实现
使用一个独立小型 causal LM 完成文档和 token NLL 打分，但用当前 Transformers 5.x `Cache` API 重写 iterative compression。优点是可控、可与 Qwen3.5 现有设备管理兼容，也适合研究 prompt/KV 联合压缩；缺点是工作量和验证成本更高。

建议先用路线 A 得到数值基线，再实现路线 B，并逐阶段对齐：document ranking → budget allocation → token mask → compressed text → recovery。

### 模型选择
- 论文复现优先：LLaMA-2-7B-Chat，成本最高。
- 轻量冒烟测试：GPT2-small，但论文消融显示效果下降。
- 工程推荐：先用 0.5B–2B causal LM 做本地替代实验；必须重新验证中英文和 Needle 指标，不应假设模型可直接互换。
- 不建议复用当前生成 Qwen3.5 实例：压缩阶段会污染/受制于当前 KV 模式，并让同一大模型承担两遍长上下文前向，降低速度收益。

### 与现有 KV-cache 压缩的关系
应把 prompt compression 与 `kv_mode` 设计为两个正交参数。不要因为启用 LongLLMLingua 就跳过 H2O/StreamingLLM。当前 `_process_chunked()` 中 `kv_mode='origin'` 也会调用 `_compress_kv_cache()`，所以 `origin` 在强制 chunking 时并非真正无 KV 压缩；做论文对照实验前应增加明确的 `kv_compression=none` 路径，否则 baseline 会被污染。

### Qwen3-VL 范围
第一阶段只支持 Qwen3.5 文本评测。Qwen3-VL 的图片占位 token、视觉特征与文本 token 对齐，不能让文本压缩器删除 image/control tokens。后续只压缩多模态消息中的文本 context，并在 `processor.apply_chat_template()` 前完成。

## 验证方案

### 单元测试
- prompt parser：正确拆分中英文 instruction/context/question，锚点缺失时显式报错或回退。
- chunker：文本无丢失、偏移可逆、每块不超预算。
- budget allocator：总预算、强制保留、边界率 0/1、单文档和空文档。
- ranker：相关 needle 块排名高于随机 haystack 块。
- token pruning：question 与数字/实体可强制保留，输出 token 数接近目标。
- recovery：`Wilhelmrad`、年份和多 token 实体可恢复，普通生成内容不被误替换。

### 端到端矩阵
至少比较：
1. 原始 prompt + 无 KV 压缩
2. LongLLMLingua only
3. KV compression only
4. LongLLMLingua + KV compression

维度：
- compression rate：`0.5 / 0.25 / 0.1`
- context length：`2k / 16k / 32k / 48k / 100k`，稳定后再到 `300k / 1M`
- needle depth：`0 / 52 / 100`
- language：`en / zh`

指标：needle score、origin/compressed tokens、压缩耗时、prefill 耗时、生成耗时、端到端耗时、峰值显存、recovery 前后分数。

## 建议实施顺序
1. 先修正真正无 KV 压缩的 baseline，并补齐时间与 token 指标。
2. 增加 Needle 结构化 parser 和 token-aware chunker。
3. 用官方 `llmlingua` 包实现 adapter，仅接 Qwen3.5。
4. 跑 2k/16k 小规模 smoke test，验证相关块排序与答案保留。
5. 跑四组消融矩阵，确认 prompt compression 与 KV compression 的独立收益和叠加收益。
6. 若官方实现与 Transformers 5.x 冲突，再实现本地 Cache API 兼容版本。
7. 最后扩展 Qwen3-VL 文本部分与中英文场景。

## 参考资料
- 论文：[ACL Anthology](https://aclanthology.org/2024.acl-long.91/)
- arXiv：[2310.06839](https://arxiv.org/abs/2310.06839)
- 官方实现：[microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)
- 核心源码：[prompt_compressor.py](https://github.com/microsoft/LLMLingua/blob/main/llmlingua/prompt_compressor.py)
