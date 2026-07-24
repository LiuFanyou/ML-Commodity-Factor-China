# 中国商品期货 Alpha 因子科学筛选系统

本仓库即本次统一试验根目录。仅从 `data_final_raw/` 重建研究面板，
不读取其他历史数据目录。因子定义见
`中国商品期货Alpha因子库V2.1_分类版.md`，执行口径见 `最终协议/`。

## 目录

```text
.
├── data_final_raw/                     # 唯一原始输入
├── 最终协议/                           # 冻结实验协议
├── 中国商品期货Alpha因子库V2.1_分类版.md
├── config/project.json                 # 冻结配置
├── src/unified_factor_system/          # 数据、标签、因子、筛选与交接代码
├── scripts/
│   ├── run_pipeline.py                 # 一键完整运行
│   ├── verify_handoff.py               # 独立交接包校验
│   └── generate_figures.py             # 学术图表生成
├── tests/                              # 防泄露与接口测试
├── outputs/
│   ├── intermediate/                   # 可复算中间面板、因子、标签和特征
│   └── screening/                      # 逐因子真实数据统计
├── reports/                            # 报告与 figures/
├── features/                           # 最终特征注册表和字典
├── training_handoff/                   # 可直接发给建模队友
├── sealed_evaluation/                  # 2025 独立评价数据
└── submission/                         # 实验提交包
```

## 完整运行

```bash
PYTHONPATH=src python scripts/run_pipeline.py
```

程序拒绝覆盖已有输出。需要重跑时，先归档或明确删除
`{outputs,reports,features,training_handoff,sealed_evaluation,submission}`。

## 验证训练交接包

```bash
PYTHONPATH=src python scripts/verify_handoff.py
```

## 交给建模队友

只发送：

```text
training_handoff/
```

队友可以直接读取 `ml_dataset_development.csv.gz`；全部输入字段以
`ml_feature_registry.csv` 的 `feature_name` 列为准。训练必须使用
`fold_definitions.csv`，禁止随机切分。

## 结论边界

- 开发期：2015-01-05 至 2024-12-31；独立评价期：2025。
- 2025 不进入开发训练包。
- 统一 5bps 是研究成本情景，不等于绝对真实交易成本。
- 因子筛选结果是统计证据，不是盈利承诺。
