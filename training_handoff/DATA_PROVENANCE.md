# 数据血缘

- 原始合约行情：`data_final_raw/futures_contract_daily_30varieties_2015_2025.csv`。
- 主力来源：TuShare 年度 `fut_daily_raw.csv` 中的精确连续代码。
- 当前文件夹包含完成计算的开发期特征和标签；训练模型不需要原始数据。
- 如需重算因子、改变主力映射、标签或点时规则，必须取得原始数据及完整源码。
