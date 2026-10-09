---
status: accepted
---

# 按完整批次估时，由预测 forward 的 runner 计价

首版采用查表与残差融合前向估时，以 Kimi-K3、8×MI355X、TP8/DCP8 为首个接入目标，并保留更换校准包的边界。`TableResidualBackend` 返回一个完整批次的 forward 成本；一个逻辑 forward 只计价一次。

选择这一边界是因为现有实测标签已包含 forward 内部通信和实际重叠，且取各 rank 的最大 wall time；把它拆成可相加的算子或通信项缺少独立测量依据。这样保留整批次测量口径、调度快照和 token 返回协议，首版表达一次阻塞 forward 的原子完成，不提供算子级、流水线或跨 step 的精细重叠模型。

计价发生在预测 forward 的 runner 上。`install_cost_backend()` 把后端装到 `NonAllocatingRunner` 实例上，`forward()` 用它为本 rank 的批次定价；没有装后端时按名拒绝该 step，因为一个未计价的 step 会表现为耗时为零的合法运行。成本经 `ScheduledBatchOutput.predicted_s` 随 reply 回到 engine，由 `clock.charge(reply)` 推进逻辑时钟——判据是 reply 是否带这个字段，不是调用的名字，因此真实 forward 留 `None` 时同一段代码是 no-op。空闲点由 `clock.idle()` 接 NER。数据并行下每个 rank 为自己的批次定价后对 DP 组做一次 `all_reduce(MAX)`，该集合通信同时替代真实 forward 会做而预测 forward 不做的跨 rank 通信；流水线 stage 明确拒绝计价，因为它只跑自己那部分层而后端定的是整个 step 的价。

首版验收计算功能、输入驱动的配置切换以及最小时钟闭环，不以 GPU 预测精度或冻结预测复现为条件。历史执行元数据缺失时保留未验证状态；已知的配置／配方不兼容仍拒绝。

运行期发生 `CostRefused` 时终止本次仿真，保留批次、校准包和首个拒绝原因，由运行控制层清理 engine、worker 及所有时钟参与者。原生 scheduler 在选批时已经改变请求和 KV 状态，首版不引入批次回滚或跳过失败批次继续执行的机制。被拒绝的批次不执行 forward／postprocess，不推进其 forward 计算时间；故障收尾与正常结束分开处理，不能伪造完成时间。

方法名称与实验轮次分开，具体校准包使用目标配置与不可变制品摘要标识；旧资产名称仅用于来源追溯。
