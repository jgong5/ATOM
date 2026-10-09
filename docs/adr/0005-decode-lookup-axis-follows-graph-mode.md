---
status: accepted
---

# Decode 查表轴由测量时的 graph 模式决定

冻结测量表的 793 行 Decode 数据全部在 CUDAGraph replay 下采集，观测到的映射为 `B=1 → rung=2`、`B=2…78 → rung=B`、`B=80 → rung=80`。校准机器捕获了近乎稠密的档位，使该表看起来像 `cost(B)`，实际是 `cost(rung)`。生产配置的 ladder 稀疏，B=37 会重放 rung=48 的图并付该档位的耗时；按 B 查表或在档位之间插值都会系统性低估，且 ladder 越稀疏偏差越大，方向恒定。

因此 Decode 的查表轴由 graph 模式决定：full graph replay 用 `cost(rung)`，以 `BatchView.capture_rung` 为坐标，网格建在测量表已有但此前未被消费的 `graph_bs` 列上；eager 用 `cost(B)`，以请求数为坐标。装配层按目标 ladder 把 B 映射成 rung，复用 `ForwardMode.decide` 的 DP1 规则。Prefill 全为 eager，不受影响。

轴声明在 grid 文件上，而不是运行时从目标配置推断。轴是测量时的性质：replay 下测出的表，它的每一行就是某个档位的耗时，目标即使跑 eager 也无法从中查到 `cost(B)`。grid 同时声明 `requires_graph_mode`，由 A1 校验目标的 graph 模式与之一致。若改为运行时按目标模式分派，eager 目标配 replay 表会静默按 B 读出错误的数。

这使目标 graph ladder 成为 replay 配方的硬依赖，对应 [ADR-0004](0004-graded-calibration-condition-checks.md) 的 A2 断言项；未断言时拒绝该配方，不退回按 B 查表。按档位查表同时降低测量成本：只需在 ladder 的各档位上测量，而非覆盖每个整数 batch size。
