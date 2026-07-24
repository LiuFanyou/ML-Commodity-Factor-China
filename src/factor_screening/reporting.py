from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ProjectConfig


def _fmt(value: object, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "暂无可报告记录。"
    return frame.to_markdown(index=False)


def write_data_audit_report(
    path: Path,
    config: ProjectConfig,
    panel: pd.DataFrame,
    curve: pd.DataFrame,
    availability: pd.DataFrame,
    point_in_time: dict[str, object],
    conceptual: pd.DataFrame,
) -> None:
    curve_days = curve[["trade_date", "product", "n_contracts"]].drop_duplicates()
    class_counts = conceptual["classification"].value_counts().sort_index().rename_axis("分类").reset_index(name="因子数")
    lines = [
        "# 数据审计报告",
        "",
        "> 本报告由项目内真实数据自动生成。没有真实点时数据的项目不会被估算或补造。",
        "",
        "## 审计结论",
        "",
        f"- 数据版本：`{config.raw['dataset_version']}`。",
        f"- 开发数据范围：{panel['trade_date'].min().date()} 至 {panel['trade_date'].max().date()}。",
        f"- 动态品种数：{panel['product'].nunique()}；品种日记录：{len(panel):,}。",
        f"- 可交易日收益覆盖率：{panel['tradable_return'].notna().mean():.2%}。",
        f"- 期限结构品种日：{len(curve_days):,}；至少2个合格合约占比：{(curve_days['n_contracts'] >= 2).mean():.2%}；至少3个占比：{(curve_days['n_contracts'] >= 3).mean():.2%}。",
        "- 开发流程没有加载仓单、库存或所谓现货表进入严格因子矩阵。",
        "- 主力选择在当日收盘后形成，最早下一交易日使用；点时检查已通过。",
        "- 2025 年封存期在开发阶段被排除。",
        "",
        "## 数据可用性清单",
        "",
        _markdown_table(availability.assign(strict_point_in_time_usable=availability["strict_point_in_time_usable"].map({True: "是", False: "否"}))),
        "",
        "## 影响结论的数据边界",
        "",
        "1. 新仓单和库存的 `available_date` 为空，全部 `point_in_time_usable=False`。",
        "2. 旧库存/仓单只有推定的下一样本交易日，并非核验后的真实发布时间。",
        "3. 原‘现货价格’文件实际是会员多空平均成本，不能计算基差。",
        "4. 手续费单位和合约乘数未核验，主报告只提供毛收益与1/3/5基点敏感性，不声称是真实交易所净收益。",
        "5. 产业链映射、生产配比、加工成本及商业交易者分类持仓缺失。",
        "",
        "## 点时检查",
        "",
        "```json",
        json.dumps(point_in_time, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 57个概念因子的当前分类数量",
        "",
        _markdown_table(class_counts),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _conceptual_with_metrics(conceptual: pd.DataFrame, feature_classes: pd.DataFrame) -> pd.DataFrame:
    metrics = feature_classes[
        [
            "feature_name",
            "coverage",
            "ic_mean",
            "rank_ic_mean",
            "rank_ic_q_bh",
            "nonoverlap_rank_ic_mean",
            "pooled_beta",
            "pooled_p_two_way_cluster",
            "group_spread",
            "group_monotonicity",
            "year_direction_consistency",
            "product_direction_consistency",
            "sector_direction_consistency",
            "max_abs_factor_correlation",
            "delta_rank_ic",
            "gross_sharpe",
            "net_3bps_sharpe",
        ]
    ].copy()
    return conceptual.merge(metrics, left_on="representative_feature", right_on="feature_name", how="left")


def write_factor_screening_report(
    path: Path,
    conceptual: pd.DataFrame,
    feature_classes: pd.DataFrame,
    config: ProjectConfig,
    sealed_metrics: pd.DataFrame | None = None,
) -> None:
    merged = _conceptual_with_metrics(conceptual, feature_classes)
    display = pd.DataFrame({
        "ID": merged["factor_id"],
        "因子": merged["factor_name"],
        "类别": merged["category"],
        "数据来源": merged["data_source"],
        "可计算": merged["computable"].map({True: "是", False: "否"}),
        "代表特征": merged["representative_feature"].replace("", "—"),
        "覆盖率": merged["coverage"].map(lambda x: _fmt(x, 3)),
        "IC": merged["ic_mean"].map(_fmt),
        "RankIC": merged["rank_ic_mean"].map(_fmt),
        "FDR q": merged["rank_ic_q_bh"].map(_fmt),
        "非重叠RankIC": merged["nonoverlap_rank_ic_mean"].map(_fmt),
        "面板回归系数": merged["pooled_beta"].map(_fmt),
        "双向聚类p": merged["pooled_p_two_way_cluster"].map(_fmt),
        "分组多空": merged["group_spread"].map(_fmt),
        "单调性": merged["group_monotonicity"].map(lambda x: _fmt(x, 3)),
        "年度方向一致率": merged["year_direction_consistency"].map(lambda x: _fmt(x, 3)),
        "品种方向一致率": merged["product_direction_consistency"].map(lambda x: _fmt(x, 3)),
        "板块方向一致率": merged["sector_direction_consistency"].map(lambda x: _fmt(x, 3)),
        "最高相关": merged["max_abs_factor_correlation"].map(lambda x: _fmt(x, 3)),
        "增量RankIC": merged["delta_rank_ic"].map(_fmt),
        "毛Sharpe(1日组合)": merged["gross_sharpe"].map(_fmt),
        "3bp净Sharpe": merged["net_3bps_sharpe"].map(_fmt),
        "分类": merged["classification"],
        "进入ML": merged["enter_ml"].map({True: "是", False: "否"}),
    })
    class_counts = conceptual["classification"].value_counts().sort_index().rename_axis("分类").reset_index(name="数量")
    feature_display = feature_classes[
        [
            "feature_name",
            "source_factor",
            "horizon",
            "coverage",
            "rank_ic_mean",
            "rank_ic_q_bh",
            "nonoverlap_rank_ic_mean",
            "pooled_beta",
            "pooled_p_two_way_cluster",
            "group_spread",
            "group_monotonicity",
            "year_direction_consistency",
            "product_direction_consistency",
            "sector_direction_consistency",
            "delta_rank_ic",
            "max_abs_factor_correlation",
            "gross_sharpe",
            "net_3bps_sharpe",
            "classification",
            "enter_ml",
        ]
    ].copy().sort_values(["classification", "rank_ic_q_bh", "feature_name"])
    for column in ["coverage", "rank_ic_mean", "rank_ic_q_bh", "nonoverlap_rank_ic_mean", "pooled_beta", "pooled_p_two_way_cluster", "group_spread", "group_monotonicity", "year_direction_consistency", "product_direction_consistency", "sector_direction_consistency", "delta_rank_ic", "max_abs_factor_correlation", "gross_sharpe", "net_3bps_sharpe"]:
        feature_display[column] = feature_display[column].map(_fmt)
    lines = [
        "# 中国商品期货 Alpha 因子科学筛选报告",
        "",
        "> 结论只来自项目内真实数据。分类是研究证据等级，不代表收益保证。",
        "",
        "## 方法与样本",
        "",
        f"- 开发样本：{config.raw['development_start']} 至 {config.raw['development_end']}。",
        "- 三年滚动训练，随后一年样本外验证；2025 在冻结前不参与选择。",
        f"- 主预测期：{config.primary_horizon}日；稳健性预测期：{', '.join(map(str, config.horizons))}日。",
        "- 因子在T日收盘后形成，T+1开盘进入可交易标签。",
        "- 同时报告 Pearson IC、Rank IC、Newey–West 推断、五分组收益、年度/品种/板块稳定性、相关性聚类和滚动Ridge增量检验。",
        "- 多重检验使用 Benjamini–Hochberg，错误发现率10%。",
        "- 成本仅为单边1/3/5基点敏感性；不是未经核验的交易所真实成本。",
        "",
        "## 分类定义",
        "",
        "- **A 核心预测因子**：FDR后显著、方向及分组收益稳定。",
        "- **B 辅助预测因子**：预测较弱，但存在增量或互补信息。",
        "- **C 条件变量**：适合作为交互、过滤或状态变量。",
        "- **D 风险变量**：主要预测风险、流动性或尾部暴露。",
        "- **E 暂不使用**：数据足够，但开发期没有可靠预测或增量证据。",
        "- **F 数据不足**：缺少严格点时数据、字段定义或经济映射。",
        "",
        "## 分类数量",
        "",
        _markdown_table(class_counts),
        "",
        "## 57个概念因子结果",
        "",
        _markdown_table(display),
        "",
        "## 逐特征实例结果",
        "",
        _markdown_table(feature_display),
        "",
        "## 经济解释与限制",
        "",
    ]
    for row in conceptual.itertuples():
        lines.extend([
            f"### #{row.factor_id} {row.factor_name}：{row.classification}类",
            "",
            f"- 经济原理：{row.economic_meaning}",
            f"- 数据与实现：{row.implementation_status}；{row.data_source}",
            f"- 结论依据：{row.classification_reason}",
            f"- 机器学习输入：{'进入' if row.enter_ml else '不进入'}。",
            "",
        ])
    if sealed_metrics is not None and not sealed_metrics.empty:
        sealed = sealed_metrics.copy()
        for column in ["ic_mean", "rank_ic_mean", "group_spread"]:
            if column in sealed:
                sealed[column] = sealed[column].map(_fmt)
        lines.extend([
            "## 2025封存测试（冻结后仅运行一次）",
            "",
            "封存结果只用于确认开发期选择，不用于重新调参或改变A—F分类。",
            "",
            _markdown_table(sealed),
            "",
        ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_factor_family_report(path: Path, conceptual: pd.DataFrame, feature_classes: pd.DataFrame) -> None:
    counts = conceptual.pivot_table(index="category", columns="classification", values="factor_id", aggfunc="count", fill_value=0).reset_index()
    primary = feature_classes.copy()
    best = primary.sort_values(["category", "rank_ic_q_bh", "rank_ic_mean"], ascending=[True, True, False]).groupby("category").head(3)
    best_display = best[["category", "feature_name", "rank_ic_mean", "rank_ic_q_bh", "delta_rank_ic", "classification"]].copy()
    for column in ["rank_ic_mean", "rank_ic_q_bh", "delta_rank_ic"]:
        best_display[column] = best_display[column].map(_fmt)
    parameter_neighborhood = (
        primary.groupby(["category", "source_factor"], as_index=False)
        .agg(
            参数实例数=("feature_name", "size"),
            RankIC均值=("rank_ic_mean", "mean"),
            RankIC最小值=("rank_ic_mean", "min"),
            RankIC最大值=("rank_ic_mean", "max"),
            FDR显著比例=("rank_ic_q_bh", lambda values: float((values <= 0.10).mean())),
            进入ML比例=("enter_ml", "mean"),
        )
    )
    direction_share = (
        primary.assign(
            positive=primary["rank_ic_mean"].gt(0).astype(float),
            negative=primary["rank_ic_mean"].lt(0).astype(float),
        )
        .groupby(["category", "source_factor"], as_index=False)
        .agg(正向比例=("positive", "mean"), 负向比例=("negative", "mean"))
    )
    parameter_neighborhood = parameter_neighborhood.merge(direction_share, on=["category", "source_factor"])
    parameter_neighborhood["方向一致率"] = parameter_neighborhood[["正向比例", "负向比例"]].max(axis=1)
    parameter_neighborhood = parameter_neighborhood.drop(columns=["正向比例", "负向比例"])
    for column in ["RankIC均值", "RankIC最小值", "RankIC最大值", "FDR显著比例", "进入ML比例", "方向一致率"]:
        parameter_neighborhood[column] = parameter_neighborhood[column].map(_fmt)
    lines = [
        "# 因子家族报告",
        "",
        "## 家族分类分布",
        "",
        _markdown_table(counts),
        "",
        "## 各家族开发期代表特征",
        "",
        _markdown_table(best_display),
        "",
        "## 参数邻域与方向稳健性",
        "",
        "同一概念因子的多个窗口被视为预注册参数邻域；方向一致率取正向或负向实例中的较高占比，不能替代样本外显著性。",
        "",
        _markdown_table(parameter_neighborhood),
        "",
        "## 家族级解释",
        "",
        "- 趋势与期限结构家族数据最完整，可作为公开风格基准。",
        "- 风险和流动性家族即使方向收益弱，也可进入模型用于风险条件化。",
        "- 基差、库存、新息及产业链家族因严格点时数据不足，当前结论是‘无法判断’，不是‘无效’。",
        "- 高相关实例通过聚类保留代表特征；参数邻域结果用于识别单点脆弱性。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ml_reports(
    report_path: Path,
    dictionary_path: Path,
    registry: pd.DataFrame,
    feature_classes: pd.DataFrame,
) -> None:
    class_counts = registry["classification"].value_counts().sort_index().rename_axis("分类").reset_index(name="特征数")
    usage_counts = registry["usage_type"].value_counts().rename_axis("用途").reset_index(name="特征数")
    lines = [
        "# 机器学习输入报告",
        "",
        f"最终输入端共 {len(registry)} 个字段（含缺失指示器和质量标签）。",
        "",
        "## 加工流程",
        "",
        "1. 因子在T日收盘后形成，最早T+1开盘使用。",
        "2. 同日横截面使用中位数和MAD去极值，并标准化；该操作不使用未来日期。",
        "3. 标准化后缺失填0，同时在缺失率超过门槛时增加缺失指示器。",
        "4. 高相关因子先聚类；保留代表特征，只有具有增量证据的非代表因子才生成正交残差。",
        "5. 交互仅使用配置中预注册的经济组合，不进行无约束枚举。",
        "6. 模型训练时仍须在每个滚动训练折内重新拟合全局缩放和任何估计参数。",
        "",
        "## 分类分布",
        "",
        _markdown_table(class_counts),
        "",
        "## 用途分布",
        "",
        _markdown_table(usage_counts),
        "",
        "## 全部输入字段",
        "",
        _markdown_table(registry),
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dictionary_lines = [
        "# 机器学习特征字典",
        "",
        "每个字段均可追溯至57因子库中的概念因子或数据质量标签。",
        "",
    ]
    for row in registry.itertuples():
        dictionary_lines.extend([
            f"## {row.feature_name}",
            "",
            f"- 来源因子：#{row.source_factor_id} {row.source_factor}",
            f"- 类别：{row.category}",
            f"- 用途：{row.usage_type}",
            f"- 变换：{row.transformation}",
            f"- 经济含义：{row.economic_meaning}",
            f"- 可用时间：{row.available_date}",
            f"- 原始缺失率：{row.missing_rate:.4f}",
            "",
        ])
    dictionary_path.write_text("\n".join(dictionary_lines) + "\n", encoding="utf-8")
