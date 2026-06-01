## 26-6-1：增加 HyperGraph branch

本次实验在 Skeleton-Based Action Segmentation 模型的空间图卷积模块中，以最小侵入方式新增 Adaptive HyperGraph branch。

- 修改 `libs/models/SP.py`，新增 `AdaptiveHyperGraphBranch`，输入输出均为 `[B, C, T, V]`，支持 `mask: [B, 1, T]` 的 mask-aware temporal average pooling。
- 在 `MultiScale_GraphConv` 中保留原始 MultiScale GraphConv 和 CTRGCN，并默认启用 HyperGraph branch。
- 融合方式为 `out = CTRGCN_output + x_base + hyper_alpha * x_hyper`。
- 新增超图参数：`hyper_k`、`hyper_hidden`、`hyper_dropout`、`hyper_alpha_init`，没有添加 `use_hypergraph` 开关。
- 修改 `libs/models/StrongDeST.py` 和 `train.py`，将超图参数从配置传入模型，并将已有 `mask` 传给空间模块。
- 修改 `libs/config.py`，为超图参数提供默认值；只在 `config/TCG-15/config.yaml` 中显式加入这些参数，其他数据集配置继续使用默认值。
- 新增 `tools/smoke_test_hypergraph_branch.py`，使用随机张量测试 `TCG-15`、`LARA`、`PKU-subject` 三种节点设置的 forward、finite 检查和 backward。

验证结果：

- `python -m py_compile libs/models/SP.py libs/models/StrongDeST.py libs/config.py train.py` 通过。
- `python tools/smoke_test_hypergraph_branch.py` 通过。
- smoke test 输出覆盖 `[B, 64, T, V]`，其中 `V` 分别为 `17 / 19 / 25`。

计算量风险主要来自 `torch.cdist` 和 `[B, V, V]` 超图传播矩阵。当前节点数较小，理论上可接受。下一步建议在 `TCG-15` 上跑 5 epoch 小实验，对比 `acc`、`edit`、`F1` 和 `boundary F1`。
