<img src="https://gw.alipayobjects.com/zos/antfincdn/R8sN%24GNdh6/language.svg" width="18"> [English](./README.md) | 简体中文

# tdd-gleaner

**拾穗者：汇集社区创作的虚拟歌手作品。**

tdd-gleaner 是[天钿 Daily](https://tdd.bunnyxt.com)（TDD）的视频发现、收录与复查项目：fetcher 发现候选视频，checker 判断内容是否符合收录范围，收录流程管理追踪资格。

## 当前阶段

项目处于评测阶段，目前只有收录规则、评测计划和一个虚构示例；fetcher、模型接入与收录服务尚未实现。

第一个里程碑是人工试标 50 条样本，并用相同的标题、简介和标签比较 Jev 与 OpenAI Decisions，记录分类错误、费用、延迟和失败情况，随后扩充评测数据集。

## 文档

- [收录与标注规则](docs/criteria.md)
- [数据集与评测计划](docs/evaluation.md)
- [虚构样本格式](examples/sample.json)

## 系统边界

- 发现来源可扩展。同一视频被重复发现时默认不重新推理，但每个请求都必须可追溯。
- checker 同时用于首次收录和复查。内容标签、追踪资格与采集频率相互独立。
- 管理员设置的持久强制收录或排除优先于自动判断。
- 新收录视频的持续采集，以频率控制和按追踪状态过滤为前提。

接口、请求 metadata、存储和部署方式尚待设计。评测样本格式不是最终的服务接口。
