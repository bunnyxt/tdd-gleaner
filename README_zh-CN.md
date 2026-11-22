<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> [English](./README.md) | 简体中文

# tdd-gleaner

**拾穗者：汇集社区创作的虚拟歌手作品。**

tdd-gleaner 是[天钿 Daily](https://tdd.bunnyxt.com)（TDD）的视频发现、收录与复查项目：fetcher 发现候选视频，checker 判断内容是否符合收录范围，收录流程管理追踪资格。

## 当前阶段

项目处于评测阶段，目前有收录规则、评测计划、虚构示例，用于构建评测数据集的离线实验资料库，以及用于比较模型的评测工具。评测工具只用虚构响应测试过，尚未进行真实模型评测；fetcher 与收录服务尚未实现。

第一个里程碑是人工试标 50 条样本，并用相同的标题、简介和标签比较 Jev 与 OpenAI Decisions，记录分类错误、费用、延迟和失败情况，随后扩充评测数据集。

## 实验资料库

`gleaner` 命令（Python 3.9+，仅用标准库）导入候选 aid，获取标题、简介和标签并生成不可变的输入快照，以 CSV 导出和导入人工标注，并生成可校验的数据包。实验资料保存在显式指定、位于版本控制之外的数据目录中。

```bash
python -m gleaner --data-dir /path/outside/the/repo init
python -m unittest discover -s tests -t .
```

完整流程与使用虚构数据的离线演示见[实验资料库说明（英文）](docs/experiment-store.md)。

## 评测工具

同一命令可生成带哈希、开发集与验收集划分固定的数据集清单；把每个快照发给 Jev 和 OpenAI Decisions，一次请求同时问两个问题；保存每次尝试的请求、响应、用量、耗时和错误；达到预设的调用次数或费用上限即停止；并在本地计算阈值和报告。见[评测工具说明（英文）](docs/evaluation-harness.md)。

## 文档

- [收录与标注规则](docs/criteria.md)
- [数据集与评测计划](docs/evaluation.md)
- [实验资料库（英文）](docs/experiment-store.md)
- [评测工具（英文）](docs/evaluation-harness.md)
- [虚构样本格式](examples/sample.json)与[虚构资料获取样例](examples/synthetic/)

## 系统边界

- 发现来源可扩展。同一视频被重复发现时默认不重新推理，但每个请求都必须可追溯。
- checker 同时用于首次收录和复查。内容标签、追踪资格与采集频率相互独立。
- 管理员设置的持久强制收录或排除优先于自动判断。
- 新收录视频的持续采集，以频率控制和按追踪状态过滤为前提。

接口、请求 metadata、存储和部署方式尚待设计。评测样本格式不是最终的服务接口。
