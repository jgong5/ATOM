---
status: accepted
---

# 几何与硬件常量由执行配置注入，配方版本只在特征函数改变时升级

查表与残差融合前向估时的特征计算中嵌有几何与硬件常量：DCP 分块宽度来自 `max(1, 16 // positive) * 1024`，chunk 轮数来自固定的 `16384`，波次量化的分母取自 `{128, 256, 304}` 三个候选值。这些字面量分别对应 block128／DCP8、attention chunk budget 以及 MI355X（256）与 MI300X（304）的 CU 数。沿用这种写法时，换一个并行度或换一代硬件就必须改实现，与 [ADR-0002](0002-data-driven-forward-calibration.md) 的输入化要求冲突。

因此把这些常量改为注入：`kv_cache_block_size`、`decode_context_parallel_size`、`dcp_config.interleave_size`、`attn_prefill_chunk_size` 取自 EngineCore 已解析的目标快照，CU 数取自 MachineSpec 新增的 `device.compute.compute_units`。波次分母拆成注入的 CU 数乘以一个量纲为一的拟合标量 `occupancy`，由校准包提供；它表达 kernel 的占用度，随参数重拟合而变，跟着 kernel 走而不是跟着硬件走。

注入不能免掉重新拟合。权重是在某一组几何值上学出来的，换成另一组几何后函数虽然相同，但树的分裂点与线性系数全部失效。注入省掉的是改代码，不是重测与重拟。因此配方版本只在特征函数本身被改写时升级：换表、换网格、换 TP／DCP、换硬件都只要求新参数与新测量表，而新增特征、改变特征定义或改变注入项集合才要求新版本。配方以具名版本加实现模块的 `code_digest` 标识，用旧摘要拟合的参数包在加载时拒绝。

真实 CU 数应当替代三候选搜索。保留三个候选等于把未知的占用度藏在模型里，并可能让权重落在另一代卡的 CU 数上。改为单一注入值会使该族特征从 28 个降到 8 个，属于配方版本变更；由于换目标本来就要重拟合，这一变更不额外增加成本。
