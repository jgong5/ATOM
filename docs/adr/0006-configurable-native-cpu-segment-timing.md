---
status: accepted
---

# 原生 CPU 区段的计时方式可配置，默认使用拟合配方

`scheduler.schedule()` 与 `scheduler.postprocess()` 是两段真实执行的主机侧工作（后者与 worker 中含 GPU 采样的 `ModelRunner.postprocess()` 是不同调用）。直接以宿主墙钟推进仿真时钟会使同一输入两次运行得到不同时间线，污染因果验收；宿主到目标的比例因子 1.0 是未经检验的断言；且这段时间推进了时钟却不进入任何 `CostTerm`，使运行记录出现没有来源的时间。

因此将计时方式做成三档配置：`off` 不计这两段；`measured` 实测宿主墙钟并同时落盘样本，兼作采集模式；`fitted` 读校准包中的 CPU 配方预测，为默认值。三档共用同一组 EngineCore 接点，换的只是数据来源而非接线，不增加对原生 EngineCore 的侵入。一个区段只取一种成本来源，不混用。

两档都产生 `scheduler.schedule` 与 `scheduler.postprocess` 两个 `CostTerm`，使 `ProvenanceMix` 覆盖全部仿真时间、不同模式的运行记录可横向比较。`fitted` 记为 `FITTED`；`measured` 记为 `MEASURED` 并在 detail 中强制标明系宿主实测、宿主非目标——Compass 的 `MEASURED` 通常指目标设备实测，这条标注区分二者。选用 `measured` 时运行记录打不可复现标记，因果验收限定在 `off` 或 `fitted` 下执行。

这细化而非取代 [ADR-0001](0001-table-residual-forward-clock-boundary.md)：forward 仍按整批次计价一次，新增的两项是它之外、此前完全未计价的区段。`measured` 模式在仿真中采集的样本受占位 forward 与 Compass 自身记账影响，可作为首个配方的起点，不足以作为最终依据，该限制须随样本记录。
