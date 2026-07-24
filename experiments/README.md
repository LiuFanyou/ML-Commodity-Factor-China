# 正则化树模型商品期货 OOS 实验

本目录只包含实验代码，不包含行情、特征、标签、模型检查点或回测输出。

## 数据约定

将训练交接包解压到仓库根目录，使下列文件可用：

```text
training_handoff/training_handoff/ml_dataset_development.csv.gz
training_handoff/training_handoff/ml_feature_registry.csv
training_handoff/training_handoff/fold_definitions.csv
futures_contract_daily_30varieties_2015_2025.zip
```

主实验使用全部注册特征、`future_return_5d` 标签、官方 2019--2024 扩展窗口和 5 日净化间隔。信号在 T 日形成、T+1 实际合约开盘执行、持有/调仓各 5 日，前后 20% 等权多空；成本输出为单边 0/3/5/10 bps。

## 运行

```bash
pip install -r experiments/requirements.txt
python experiments/run_oos_ml_experiment.py
python experiments/run_selected_factor_experiment.py
python experiments/run_complex_tree_experiment.py
```

## Visualization

Generate aggregate-only comparison figures after the three experiments finish:

```bash
python experiments/plot_results.py --baseline delivery/all_features --selected delivery/selected_features --complex delivery/complex_trees --output figures
```

The plotting script reads model outputs only. It does not package raw market data.

- `run_oos_ml_experiment.py`：全 92 个注册字段的正则化 LightGBM、XGBoost、随机森林基准。
- `run_selected_factor_experiment.py`：每个训练折中先按登记相关性簇去重，再按训练期平均绝对横截面 Rank IC 选 Top-30。
- `run_complex_tree_experiment.py`：使用更多树、更深树的扩展，但仍保留采样、最小节点、L1/L2 或剪枝约束。

脚本将生成预测、持仓、日收益、成本敏感性、特征重要性和配置文件。所有模型都固定相同的时间切分、标签和交易口径。
