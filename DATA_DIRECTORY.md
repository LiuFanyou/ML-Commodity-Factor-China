# 当前数据边界

## 唯一研究输入

```text
data_final_raw/
├── futures_contract_daily_30varieties_2015_2025.csv
├── futures_contract_daily_30varieties_2015_2025字段说明.md
└── tushare_theme7/*/fut_daily_raw.csv
```

本试验不读取任何其他原始数据目录。

## 可复现性

- 有 `data_final_raw/`、`最终协议/`、57 因子文档和 `src/` 源码，
  可以从头重建全部筛选、2025 评价和交接包。
- 只有 `training_handoff/`，可以直接训练模型，但不能重算因子。
- 原始数据是本地大文件，默认不提交 Git。

## 数据限制

- 连续主力通过 T 日行情字段映射真实合约，不实施 T−1 持仓量重选。
- 基本面只有观察日期时统一延迟一个交易日可用。
- 现货源截至 2023 年，超过 7 天的旧值不会继续进入基差。
- 2021 年前真实手续费覆盖不足。
- 没有完整盘口、精确涨跌停规则、逐笔冲击和容量数据。
