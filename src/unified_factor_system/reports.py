from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _format(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "NA"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def write_data_audit_report(
    path: Path,
    audit: dict[str, object],
    factor_registry: pd.DataFrame,
) -> None:
    unavailable = factor_registry.loc[~factor_registry["available"].astype(bool)]
    lines = [
        "# data_final_raw 数据审计报告",
        "",
        "## 审计结论",
        "",
        "- 本轮只使用 `data_final_raw`，未读取 `data3` 或旧版 research 数据生成因子或标签。",
        "- `trade_date × product` 主键通过唯一性检查。",
        "- 按用户修订，读取 TuShare 精确连续代码并映射真实合约，不实施 T−1 持仓量重选主力。",
        "- 完整期限结构来自全部实际合约，并与主力映射规则相互独立。",
        "- 基本面统一延迟一个交易日后才可见；现货最大陈旧期7天。",
        "- 2025 数据仅输出到独立评价目录，不进入开发训练交接包。",
        "",
        "## 数据概况",
        "",
        "| 项目 | 数值 |",
        "|---|---:|",
    ]
    for key, value in audit.items():
        lines.append(f"| `{key}` | {_format(value, 6)} |")
    lines.extend(
        [
            "",
            "## 因子数据可用性",
            "",
            f"- 预注册原始因子：{len(factor_registry)} 个。",
            f"- 通过覆盖率与非零方差硬门：{int(factor_registry['available'].sum())} 个。",
            f"- 数据不足：{len(unavailable)} 个。",
            "",
        ]
    )
    if unavailable.empty:
        lines.append("所有预注册因子均可计算。")
    else:
        lines.extend(
            [
                "| 因子 | 覆盖率 | 原因 |",
                "|---|---:|---|",
            ]
        )
        for row in unavailable.itertuples():
            reason = (
                row.unavailable_reason
                if isinstance(row.unavailable_reason, str) and row.unavailable_reason
                else "开发期覆盖率低于硬门或无有效横截面变化"
            )
            lines.append(
                f"| `{row.factor_id}` | {_format(row.development_coverage)} | "
                f"{reason} |"
            )
    lines.extend(
        [
            "",
            "## 已知数据限制",
            "",
            "- 连续主力到真实合约依赖同日行情字段映射；极少数非完全匹配行通过冻结的确定性规则处理。",
            "- `suspected_price_limit_lock` 是 OHLC 锁价代理，不等价于交易所逐日精确涨跌停价。",
            "- 文件没有盘口深度、买卖价差和逐笔冲击数据，不能核验真实盘口冲击与容量。",
            "- 2021年前真实手续费覆盖较弱，统一5bps只用于可比研究成本，不能声称为完整实盘净收益。",
            "- 现货源截至2023年，2024—2025过期现货不会被继续当作有效基差。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_screening_report(
    path: Path,
    classified: pd.DataFrame,
    incremental: pd.DataFrame,
) -> None:
    table = classified.merge(
        incremental[
            ["factor_id", "mean_oos_delta_mse", "mean_oos_delta_rank_ic"]
        ],
        on="factor_id",
        how="left",
    )
    class_counts = table["screening_class"].value_counts()
    fdr_rank_count = int(table["rank_ic_fdr_q"].le(0.10).fillna(False).sum())
    incremental_positive = int(
        table["mean_oos_delta_rank_ic"].gt(0).fillna(False).sum()
    )
    auxiliary = table.loc[
        table["screening_class"].eq("B 辅助预测因子"),
        "factor_id",
    ].tolist()
    icir_leaders = (
        table.dropna(subset=["rank_ic_ir"])
        .sort_values(["rank_ic_ir", "rank_ic"], ascending=False)
        .head(10)
    )
    lines = [
        "# 中国商品期货 Alpha 因子筛选报告",
        "",
        "## 方法与结论边界",
        "",
        "统计结果来自 2019—2024 六个扩展窗口的逐年样本外区间，2025 未参与筛选。所有方向均按预注册经济方向定向；"
        "混合方向变量保持原方向。特征是否进入机器学习只由预注册范围与数据质量硬门决定，"
        "不按样本外收益追逐式删选。",
        "",
        "分类含义：A 核心预测、B 辅助预测、C 条件、D 风险、E 当前证据不足、F 数据不足。",
        "",
        "## IC 与 ICIR 口径",
        "",
        "- `IC` 是逐日横截面 Pearson 相关系数的时间均值；`Rank IC` 是逐日横截面 Spearman 相关系数的时间均值。",
        "- `ICIR = Mean(IC_t) / Std(IC_t)`；`Rank ICIR = Mean(RankIC_t) / Std(RankIC_t)`。",
        "- 本报告中的 ICIR 和 Rank ICIR 均为**未年化日度口径**，避免把弱日度信号乘以 `sqrt(252)` 后造成直观夸大。",
        "- Rank IC 的 t 统计量和原始 p 值使用最多5阶 Newey–West/HAC 修正；FDR q 值对所有可评价 Rank IC 检验统一进行 Benjamini–Hochberg 修正。",
        "- ICIR衡量相关系数序列的稳定程度，不等于策略夏普，也不能替代FDR、成本后收益和跨年份稳定性。",
        "",
        "## 总体筛选结论",
        "",
        f"- 预注册因子：{len(table)}；进入机器学习接口：{int(table['enters_ml'].sum())}。",
        f"- A类：{int(class_counts.get('A 核心预测因子', 0))}；"
        f"B类：{int(class_counts.get('B 辅助预测因子', 0))}；"
        f"C类：{int(class_counts.get('C 条件变量', 0))}；"
        f"D类：{int(class_counts.get('D 风险变量', 0))}；"
        f"E类：{int(class_counts.get('E 暂不使用因子', 0))}；"
        f"F类：{int(class_counts.get('F 数据不足无法判断', 0))}。",
        f"- Rank IC在10% FDR下通过的因子：{fdr_rank_count}。"
        "若该值为0，意味着当前开发样本尚没有足够证据把任何预测因子称为核心Alpha。",
        f"- 固定Ridge增量Rank IC为正的候选：{incremental_positive}；该结果仅用于互补性诊断。",
        f"- B类候选：{', '.join(f'`{name}`' for name in auxiliary) if auxiliary else '无'}。",
        "- E类仍可进入全量机器学习接口，因为这里的分类是单因子证据等级，不等同于联合条件信息为零。",
        "",
        "## 逐因子结果",
        "",
        "| 因子 | 类别 | 数据来源 | 覆盖率 | IC日期数 | IC | IC标准差 | ICIR | Rank IC | Rank IC标准差 | Rank ICIR | Rank IC胜率 | HAC t | 原始p | FDR q | 正向年份率 | 正向品种率 | 正向板块率 | 分组单调性 | Q5-Q1 | 延迟1日Rank IC | 最大相关 | 5bps夏普 | 年化收益 | 组合FDR q | 增量Rank IC | 分类 | 进入ML | 经济原理 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    icir_summary = [
        "## ICIR 稳定性摘要",
        "",
        "下表按未年化 Rank ICIR 排序。ICIR较高表示逐日横截面预测方向相对稳定，"
        "但是否属于核心Alpha仍需同时满足FDR、成本后收益、跨年份稳定性和数据覆盖要求。",
        "",
        "| 因子 | ICIR | Rank ICIR | Rank IC | Rank IC胜率 | FDR q | 分类 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in icir_leaders.itertuples(index=False):
        icir_summary.append(
            f"| `{row.factor_id}` | {_format(row.ic_ir)} | {_format(row.rank_ic_ir)} | "
            f"{_format(row.rank_ic)} | {_format(row.rank_ic_hit_rate)} | "
            f"{_format(row.rank_ic_fdr_q)} | {row.screening_class} |"
        )
    icir_summary.append("")
    detail_index = lines.index("## 逐因子结果")
    lines[detail_index:detail_index] = icir_summary
    for row in table.itertuples(index=False):
        lines.append(
            f"| `{row.factor_id}` | {row.category} | {row.data_source} | {_format(row.coverage)} | "
            f"{int(row.ic_observations) if pd.notna(row.ic_observations) else 'NA'} | "
            f"{_format(row.ic_mean)} | {_format(row.ic_std)} | {_format(row.ic_ir)} | "
            f"{_format(row.rank_ic)} | {_format(row.rank_ic_std)} | {_format(row.rank_ic_ir)} | "
            f"{_format(row.rank_ic_hit_rate)} | {_format(row.rank_ic_t_stat)} | "
            f"{_format(row.rank_ic_p_value)} | {_format(row.rank_ic_fdr_q)} | "
            f"{_format(row.positive_year_rate)} | "
            f"{_format(row.positive_product_rate)} | {_format(row.positive_sector_rate)} | "
            f"{_format(row.monotonicity)} | {_format(row.group_return_spread)} | "
            f"{_format(row.delayed_1d_rank_ic)} | {_format(row.max_abs_feature_correlation)} | "
            f"{_format(row.sharpe)} | {_format(row.annual_return)} | {_format(row.portfolio_fdr_q)} | "
            f"{_format(row.mean_oos_delta_rank_ic, 6)} | {row.screening_class} | "
            f"{'是' if row.enters_ml else '否'} | {row.economic_meaning} |"
        )
    lines.extend(
        [
            "",
            "## 相关性与增量价值的处理",
            "",
            "- 高相关变量不按历史标签择优删除；交接注册表提供相关簇及代表变量。",
            "- Rank IC与组合均对本轮可评价因子实施Benjamini–Hochberg错误发现率修正。",
            "- `parameter_sensitivity_metrics.csv` 固定比较趋势、Carry变化、基差动量、反转、波动率、持仓增长和曲线动量的邻近窗口；只作稳健性诊断。",
            "- 线性模型建议使用 Ridge/Lasso 或在训练折内做降维；树模型可直接读取全部字段。",
            "- 增量列采用固定 Ridge，在每个扩展折中比较预注册基线（TSMOM、Carry、Amihud、Turnover）与“再加入该特征”；仅作诊断，不反向改变特征集合。",
            "- 单因子弱但属于不同信息维度的变量仍可作为条件、风险或组合信息进入模型。",
            "",
            "## 执行限制",
            "",
            "报告中的因子组合按 T 收盘信号、固定真实合约 T+1 开盘至 T+6 开盘收益、每五日调仓、前后20%、"
            "组内等权、单边5bps估算。源数据缺少精确涨跌停价、盘口与完整容量模型，"
            "因此结果是统一研究回测，不应表述为已核验实盘净收益。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_family_report(path: Path, classified: pd.DataFrame) -> None:
    summary = (
        classified.groupby("category", as_index=False)
        .agg(
            factor_count=("factor_id", "count"),
            available_count=("available", "sum"),
            mean_rank_ic=("rank_ic", "mean"),
            median_rank_ic=("rank_ic", "median"),
            mean_net_sharpe=("sharpe", "mean"),
            mean_coverage=("coverage", "mean"),
        )
        .sort_values("category")
    )
    lines = [
        "# 因子家族报告",
        "",
        "| 家族 | 因子数 | 可计算 | 平均Rank IC | 中位Rank IC | 平均5bps夏普 | 平均覆盖率 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.category} | {row.factor_count} | {int(row.available_count)} | "
            f"{_format(row.mean_rank_ic)} | {_format(row.median_rank_ic)} | "
            f"{_format(row.mean_net_sharpe)} | {_format(row.mean_coverage)} |"
        )
    lines.extend(
        [
            "",
            "家族统计用于判断信息来源的整体质量，不等同于承诺未来收益。动量、期限结构和流动性是统一协议主线；"
            "波动率、偏度、持仓变化、交互和质量标记属于扩展输入。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_feature_report(path: Path, registry: pd.DataFrame) -> None:
    lines = [
        "# 机器学习输入端报告",
        "",
        f"最终输入字段共 **{len(registry)}** 个。`ml_feature_registry.csv` 的 `feature_name` 列是唯一默认模型输入清单。",
        "",
        "## 加工规则",
        "",
        "- 品种级连续变量：同一交易日横截面中位数/MAD 标准化，截断后将缺失值置为横截面中性值 0。",
        "- 市场级条件变量：只使用历史值形成滚动均值和标准差。",
        "- 高于阈值的原始缺失率：附加独立缺失标记。",
        "- 二元质量变量：保留 0/1。",
        "- 交互项：只采用协议允许、经济含义预注册的少量乘积项。",
        "- 不使用标签决定最终输入字段；相关聚类只提供诊断和代表字段，不物理删除。",
        "- 产业链残差、板块残差和交互项按预注册经济公式实现；不实施全样本目标驱动的统一正交化。",
        "",
        "## 全部模型输入",
        "",
        "| feature_name | 来源 | 变换 | 类别 | 用途 | 相关簇 | 代表变量 | 经济含义 |",
        "|---|---|---|---|---|---:|---|---|",
    ]
    for row in registry.itertuples(index=False):
        lines.append(
            f"| `{row.feature_name}` | `{row.source_factor}` | {row.transformation} | "
            f"{row.category} | {row.usage_type} | {row.correlation_cluster} | "
            f"`{row.cluster_representative}` | {row.economic_meaning} |"
        )
    lines.extend(
        [
            "",
            "线性回归不建议直接使用普通 OLS 解释所有高度相关系数；首选 Ridge，或仅在各训练折内使用相关簇代表变量。"
            "LightGBM/XGBoost 可使用注册表中全部特征，但仍应固定滚动折与早停规则。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_sealed_report(
    path: Path,
    development: pd.DataFrame,
    sealed: pd.DataFrame,
) -> None:
    comparison = development[
        [
            "factor_id",
            "factor_name",
            "screening_class",
            "rank_ic",
            "annual_return",
            "sharpe",
        ]
    ].merge(
        sealed[
            [
                "factor_id",
                "rank_ic",
                "annual_return",
                "sharpe",
                "ic_observations",
            ]
        ],
        on="factor_id",
        how="left",
        suffixes=("_dev", "_2025"),
    )
    lines = [
        "# 2025 封存测试报告",
        "",
        "本报告在2019—2024开发筛选、特征集合、方向、组合映射和成本情景全部冻结后生成。"
        "2025不参与因子选择、参数选择或相关簇定义。",
        "",
        "| 因子 | 开发分类 | 开发Rank IC | 2025 Rank IC | 开发年化 | 2025年化 | 开发夏普 | 2025夏普 | 2025 IC日期数 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison.itertuples(index=False):
        lines.append(
            f"| `{row.factor_id}` | {row.screening_class} | "
            f"{_format(row.rank_ic_dev)} | {_format(row.rank_ic_2025)} | "
            f"{_format(row.annual_return_dev)} | {_format(row.annual_return_2025)} | "
            f"{_format(row.sharpe_dev)} | {_format(row.sharpe_2025)} | "
            f"{_format(row.ic_observations)} |"
        )
    usable = comparison.loc[comparison["rank_ic_2025"].notna()]
    same_direction = (
        np.sign(usable["rank_ic_dev"]) == np.sign(usable["rank_ic_2025"])
    )
    lines.extend(
        [
            "",
            "## 汇总结论",
            "",
            f"- 有2025可评价Rank IC的因子：{len(usable)}。",
            f"- 开发期与2025 Rank IC同方向比例：{_format(float(same_direction.mean()) if len(same_direction) else np.nan)}。",
            "- 该结果是一次封存检验，不用于回头改变已交付的特征注册表。",
            "- 缺少盘口冲击、完整涨跌停规则和容量数据，5bps净收益仍是研究口径。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
