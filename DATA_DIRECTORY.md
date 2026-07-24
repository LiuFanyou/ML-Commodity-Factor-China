# 数据目录结构

```text
中国期货量化因子挖掘/
├── data_raw/                         # 不修改的原始数据
│   ├── README.md
│   ├── tushare_theme7/                 # 2015—2025合约行情、元数据、交易规则
│   ├── legacy_2020_2025/               # 原有行情、主力、库存、仓单、会员成本
│   └── supplement_2015_2025/           # 后续补充的仓单/库存/可用量
├── data_research/                    # 可重建的研究数据
│   └── core30_v1/
│       ├── README.md
│       ├── DATA_DICTIONARY.md
│       ├── QUALITY_REPORT.md
│       ├── MANIFEST.json
│       ├── FILE_CHECKSUMS.json
│       ├── core30_universe.csv
│       └── tables/
│           ├── market_contract_daily.csv.gz
│           ├── contract_rules_daily.csv.gz
│           ├── contract_metadata.csv.gz
│           ├── main_selection_daily.csv.gz
│           ├── research_returns_daily.csv.gz
│           ├── warehouse_receipts_daily_new.csv.gz
│           ├── warehouse_inventory_periodic_new.csv.gz
│           ├── legacy_inventory_weekly.csv.gz
│           ├── legacy_warehouse_receipts_weekly.csv.gz
│           └── member_cost_daily.csv.gz
└── data_pipeline/                    # 数据重建代码
    └── build_core30_v1.py
```

## 使用约定

- `data_raw/` 是证据层，不直接改动。
- `data_research/core30_v1/` 是后续因子研究的唯一默认数据入口。
- 旧数据和新补充数据同时参与数据建设，但不强行合并不同口径。
- 原比较结果和因子报告属于研究记录，不是数据层，目前保留。
