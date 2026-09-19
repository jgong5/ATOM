> **This is the original seed, not a design document.** It is the task as first written:
> sketchy, partly superseded, and non-normative. Several statements in it were revised
> during the design interview and a few were withdrawn. Where it disagrees with a design
> topic (`01`-`12`), **the design topic wins**. Kept verbatim for provenance.
>
> Milestones and sequencing will be owned by the execution plan (`15`), not by this file.

---

Please brainstorm the design according to the following development task. Your goal is to come up with a design and an execution plan for implementation. You should not to follow the instructions as is but should explore design options, bring suggestions, verify assumptions, identify potential issues, close open issues, clarify anything unclear and make a plan for efficient implementation (e.g. multi-agents working on parallel sub-tasks). When explore design options for key design points, try your best to survey/study/prove them before giving options and suggestions.

Start with high-level architecture, control/data flow design, then to the component/module details. Use multi-agents where applicable to speed up the brainstorming and design processes.

For each design point, dump problem statements, explored design options with pros/cons, preferred option with reasons, open issues and anything you believe important to a dedicate markdown file - you give a proper file name.

The conversation can be with English or Mandarin or mixed but make sure the generated documents and code all use English only.

### Project context
Project named "ATOM Compass", a performance simulator for ATOM. It answers: **for this model, this parallelism, and this workload — what latency and throughput, and does it fit in
memory?**

The central design choice: **Compass replaces only the forward pass.**, reusing ATOM's
real scheduler, real block manager, real admission logic and LLM modeling as much as possible.
A simulated run therefore makes the same scheduling decisions as a real one; it
only substitutes a predicted duration for the work. Compass does not model
serving *decisions*, only the time they consume.

Following are modelled:

* **time** — what a step costs, and what serving adds around it.
* **memory** — what a configuration consumes, and so whether it fits and how
  many KV blocks it gets.
* KV cache pool, simulation of the real counterpart from ATOM including caching abstraction and transfer.

Memory is in scope because it decides which configurations exist at all.
Predicting the speed of a configuration that cannot start is answering the wrong
question.

### Goal
The PoC goal is to develop and validate **ATOMCompass as a tool that predicts LLM serving performance and memory usage.** 无GPU仿真不是强制要求，但是仿真执行不在GPU运算，也不在GPU上分配内存。仿真GPU算力，显存大小，带宽，互联可配置，相关信息不通过真实GPU运行时获取。

The required results are:

| Requirement                  | Acceptance target                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------------- |
| Throughput prediction        | Error ≤10%                                                                                   |
| Time per output token (TPOT) | Error ≤10%                                                                                   |
| Time to first token (TTFT)   | Error ≤10%                                                                                   |
| Each non-KV memory term      | Error ≤10%                                                                                   |
| KV capacity/block count      | Error ≤5%                                                                                    |
| Generalization               | Demonstrate prediction beyond the configurations used for calibration                        |
| Simulation speed             | Aim for ≥5× speedup; **lower priority and negotiable, bottom line is faster than real runs** |

**The final proof must come from paired simulation and real execution of cc-traces proper (refer to below for details of cc-traces.**

The workload coverage must include:

- **Prefix caching enabled**, matching the cc-traces default.
- **1, 4, 16, 64, 256 clients**. These count root sessions; subagents can make the number of concurrent requests exceed the client count.
- Faithful agentic behavior: branching, joins, delays, completion-driven recycling, cancellation, cache reuse and the resulting batch shapes.

### Milestones
1. Complete serving modeling with fake models and establish foundations
	1. The fake models can simulate all a real model can do, like prefill, decode, KV cache needs, parallel strategies: TP, DP, PP, EP
	2. Establish discrete event simulation foundation
	3. test harness ready
	4. PD aggregation and disaggregation with various parallel strategies with cc_traces harness
2. **Qwen3.8-27B on MI308X-class hardware**, PD aggregation, covering TP1**
	1. Sub-goal: 先搭建一个使用假模型级别
3. **Qwen3.8-27B on MI308X-class hardware**, PD aggregation, covering TP2 and TP4**.
4. Qwen3.8-27B same hardware and TP configs but with PD disaggregation on two nodes.
5. Kimi-K3 same hardware, TP8, PD aggregation.
6. Kimi-K3 same hardware, TP8, PD disaggregation.
7. Kimi-K3 with DP, PP, EP support.

### Design principle
- 复用ATOM的api server和调度模块，只替换模型层以及模型层依赖的模块（如KV cache管理层，通信层等），复用部分可适当修改、重构以支持离散时间和其他仿真逻辑需要的功能。
- Prioritize simplicity, only add necessary design/implementation, not more
- Prefer clean abstractions and refactoring for that than adhoc code changes.
- Start with small, cut tasks into smaller easily verifiable stages.

### 关键技术点（设计需要重点考虑）
1. 整合
	1. 离散时间和真实时间轴整合
		1. 进程线程模型：是否需要使用单进程，单线程模拟，还是保持ATOM本身的多进程、多线程模型不变？全局虚拟时钟如何设计，在进程内部全局，还是分布式全局？
		2. 进程线程通信、同步：离散时间管理是否需要完全兼容python多进程、多线程通信，还是提供有限的支持，还是只支持单进程、单线程？
		3. 业务指标(metrics)
		4. tokenizer的耗时
	2. 并行策略支持：单节点部署模拟还是多节点部署？多卡建模用多进程还是单进程模拟？
	3. 模型捕获
		1. 无GPU抓取，模型IR定义，适配动态shape
		2. 图定义缓存，什么时候需要抓取，如何匹配？动态shape需要模糊
		3. 算子抓取：torch算子；Triton算子；FlyDSL算子
	4. GPU依赖去除
		1. KV缓存及前缀缓存建模
		2. 模型加载
	5. command line args：需要指定哪些关键参数？
2. 建模
	1. 服务化建模
	2. 模型单步建模（粗粒度建模），需要考虑cuda graph padding？
	3. 算子建模（triton, flydsl, aiter, torch），kernel launch开销，多流并行如何建模？
	4. 线下带GPU op benchmark，性能拟合，覆盖计算和通信算子，需注意单独op benchmark的性能和端到端跑同样op性能之间的差距。如何得到所有需要benchmark的算子？是否可以复用模型捕获逻辑？
	5. 显存占用建模：模型权重，KV cache大小（依赖KV cache建模），激活大小（liveness分析），算子scratchpad占用。
3. 打流
	1. 离散时间感知客户端
		1. 服务端接收请求保序：客户端提前告诉服务端总请求数，一次发送后在服务端排序，再处理。使用特殊的base url还是复用原先的url，在原end point上修改？
		2. 客户端需要给定请求时间戳
		3. 客户端需要给出每一个请求的输出长度
	2. 定长序列打流支持
	3. cc-traces harness支持

4. 设计问题讨论：
	1. 我们真的需要完全无GPU执行吗？或许不需要。作为ATOM的一部分，如果ATOM本身就依赖GPU，我们只要保证不使用真实GPU加载模型和执行就好了。
	2. model runner不需要多模式设计，每次都需要simulation，具体simulation的算法由建模后端给出。额外给定一个测量（measure）标志，做了在真实GPU上的测量实跑，并把测量结果缓存起来供simulation使用。模型抓取（trace）按需在第一次自动执行，并缓存起来供后续simulation使用。
5. 开放问题：
	1. 模型定义可能会随着输入shape不同而不同，比如TBO这个功能不是在所有shape下面都打开的。

### 开发分支
ATOM
branch: feature/atomcompass_new

### HW resources
`hjbog-srdc-<id>.amd.com` with id ranges from 18 to 22, and 39.
Access via `~/my_ssh/id_ed25519`
Configured environments on 18 and 39. 
containers configured on 18:
 xiaobizh_n18     │ rocm/atom-dev:latest
 xiaobizh_n18_cpu │ rocm/atom-dev:latest

### Implementation reference
Following is a partial implementation. Don't treat it as golden but it contains learnings from previous experiments, problem framing and developed components.
https://github.com/jgong5/ATOM/pull/2

### cc-traces
For max 1M context:
https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
For max 256k context:
[https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k "https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k")
Each client in cc-traces can spawn multiple sub-agents so that one client can make multiple LLM requests to the server.
Refer to the following code on how to construct the test harness:
https://github.com/SemiAnalysisAI/agentx-harness
