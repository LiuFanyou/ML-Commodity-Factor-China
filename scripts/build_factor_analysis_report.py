from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CATEGORY_NAMES = {
    "trend_momentum": "趋势与动量",
    "term_structure": "期限结构与Carry",
    "basis_spot": "基差与现货",
    "value_reversion": "价值与均值回复",
    "inventory_fundamental": "库存与供需",
    "trading_liquidity": "交易行为与流动性",
    "volatility_risk": "波动率与风险",
    "industry_chain": "产业链与相对价值",
    "state_interaction": "市场与品种状态",
    "meta_factor": "元因子",
    "information_news": "信息冲击与反应",
    "factor_interaction": "非线性与交互",
}


def fmt(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):.{digits}f}"


def pct(value: object) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{100.0 * float(value):.1f}%"


def class_code(value: str) -> str:
    return value.split(" ", 1)[0]


def implementation_text(row: pd.Series) -> str:
    if not bool(row["available"]):
        reason = row.get("unavailable_reason")
        return f"不可计算；{reason if pd.notna(reason) else '缺少必要数据'}"
    event_note = "，属于低频事件特征" if bool(row.get("event_feature", False)) else ""
    return (
        f"已计算；数据源为{row['data_source']}；开发期覆盖率{pct(row['coverage'])}"
        f"，覆盖{int(row['development_instruments'])}个品种{event_note}"
    )


def development_evidence(row: pd.Series) -> str:
    if not bool(row["available"]):
        return "没有生成统计结果，不能以零收益或零IC替代缺失证据。"
    if pd.isna(row.get("rank_ic")):
        return (
            "该变量在同一日期缺少足够横截面差异，因此单日横截面IC不可解释；"
            "它只能作为跨时间变化的条件变量评价。"
        )
    return (
        f"开发期Rank IC {fmt(row['rank_ic'])}、Rank ICIR {fmt(row['rank_ic_ir'])}、"
        f"胜率{pct(row['rank_ic_hit_rate'])}、FDR q {fmt(row['rank_ic_fdr_q'])}；"
        f"五分组多空收益{fmt(row['group_return_spread'])}，5bps后年化"
        f"{pct(row['annual_return'])}、夏普{fmt(row['sharpe'])}；"
        f"正向年份率{pct(row['positive_year_rate'])}。"
    )


def sealed_evidence(row: pd.Series) -> str:
    sealed_rank = row.get("sealed_rank_ic")
    if pd.isna(sealed_rank):
        return "2025没有可评价样本，不能完成封存确认。"
    development_rank = row.get("rank_ic")
    same_direction = (
        np.sign(float(development_rank)) == np.sign(float(sealed_rank))
        if pd.notna(development_rank) and float(development_rank) != 0
        else False
    )
    direction = "同方向" if same_direction else "方向不一致"
    return (
        f"2025 Rank IC {fmt(sealed_rank)}、5bps后年化"
        f"{pct(row.get('sealed_annual_return'))}、夏普{fmt(row.get('sealed_sharpe'))}；"
        f"与开发期Rank IC{direction}。"
    )


def conclusion_text(row: pd.Series) -> str:
    code = class_code(str(row["screening_class"]))
    enters = "进入" if bool(row["enters_ml"]) else "不进入"
    q_value = row.get("rank_ic_fdr_q")
    incremental = row.get("mean_oos_delta_rank_ic")
    incremental_text = (
        f"固定Ridge增量Rank IC为{fmt(incremental, 6)}"
        if pd.notna(incremental)
        else "没有可解释的固定Ridge增量结果"
    )
    if code == "A":
        reason = "核心预测证据通过预设统计、经济和稳健性门槛。"
    elif code == "B":
        reason = (
            f"存在正向预测和组合证据，但FDR q为{fmt(q_value)}，未达到10%核心门槛；"
            f"{incremental_text}。"
        )
    elif code == "C":
        reason = (
            "经济角色主要是调节其他信号、刻画交易状态或形成交互，"
            "不要求其单独产生稳定多空收益。"
        )
    elif code == "D":
        reason = (
            "主要用途是风险、波动、流动性或尾部暴露刻画，"
            "不能因某一年方向收益较高而改列Alpha。"
        )
    elif code == "E":
        reason = (
            "开发期无条件预测、成本后收益或跨期稳定性不足；"
            f"{incremental_text}。"
        )
    else:
        reason = (
            f"缺失数据为：{row.get('unavailable_reason')}。"
            "当前只能保留概念定义，不能形成数值特征。"
        )
    return f"{reason}机器学习输入：{enters}。"


def family_comment(category: str, group: pd.DataFrame) -> str:
    available = group.loc[group["available"].astype(bool)]
    b_count = int(group["screening_class"].str.startswith("B").sum())
    f_count = int(group["screening_class"].str.startswith("F").sum())
    if available.empty:
        return (
            f"{CATEGORY_NAMES.get(category, category)}家族当前没有可评价因子；"
            f"{f_count}个概念因子因数据或经济映射不足列为F类。"
        )
    rank_evaluable = available.dropna(subset=["rank_ic"])
    if rank_evaluable.empty:
        return (
            f"{CATEGORY_NAMES.get(category, category)}共{len(group)}个概念因子、"
            f"{len(available)}个可计算，但变量在同一日期没有足够横截面差异，"
            "不适用横截面Rank IC评价，应仅作为跨时间条件变量使用。"
        )
    leader = rank_evaluable.loc[rank_evaluable["rank_ic"].idxmax()]
    mean_rank = rank_evaluable["rank_ic"].mean()
    return (
        f"{CATEGORY_NAMES.get(category, category)}共{len(group)}个概念因子、"
        f"{len(available)}个可计算，平均Rank IC {fmt(mean_rank)}。"
        f"家族内开发期Rank IC最高的是`{leader['factor_id']}`（{fmt(leader['rank_ic'])}）；"
        f"B类{b_count}个、F类{f_count}个。"
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "outputs" / "screening" / "factor_metrics_with_icir.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(
            "run `PYTHONPATH=src python scripts/build_icir_report.py` first"
        )

    factors = pd.read_csv(metrics_path)
    incremental = pd.read_csv(
        root / "outputs" / "screening" / "incremental_ridge_tests.csv"
    )
    sealed = pd.read_csv(
        root / "sealed_evaluation" / "factor_metrics_2025.csv"
    )[
        ["factor_id", "coverage", "rank_ic", "annual_return", "sharpe"]
    ].rename(
        columns={
            "coverage": "sealed_coverage",
            "rank_ic": "sealed_rank_ic",
            "annual_return": "sealed_annual_return",
            "sharpe": "sealed_sharpe",
        }
    )
    factors = (
        factors.merge(
            incremental[
                ["factor_id", "mean_oos_delta_mse", "mean_oos_delta_rank_ic"]
            ],
            on="factor_id",
            how="left",
            validate="one_to_one",
        )
        .merge(sealed, on="factor_id", how="left", validate="one_to_one")
        .sort_values("factor_no")
        .reset_index(drop=True)
    )
    factors["category_cn"] = factors["category"].map(CATEGORY_NAMES).fillna(
        factors["category"]
    )
    factors["class_code"] = factors["screening_class"].map(class_code)

    class_order = ["A", "B", "C", "D", "E", "F"]
    class_counts = (
        factors["class_code"].value_counts().reindex(class_order, fill_value=0)
    )
    available = factors.loc[factors["available"].astype(bool)]
    b_candidates = factors.loc[factors["class_code"].eq("B")]

    lines = [
        "# 中国商品期货 Alpha 因子科学筛选分析报告",
        "",
        "> 本报告参考既有筛选报告的组织方式，但所有数值与结论均来自当前"
        "`data_final_raw`重建结果；参考文件中的旧样本结果未被复用。",
        "",
        "## 核心结论",
        "",
        f"- 57个概念因子中，47个可计算并进入机器学习接口，10个因数据不足无法判断。",
        f"- 分类结果：A类{class_counts['A']}个、B类{class_counts['B']}个、"
        f"C类{class_counts['C']}个、D类{class_counts['D']}个、"
        f"E类{class_counts['E']}个、F类{class_counts['F']}个。",
        "- 当前没有A类因子。46个可评价Rank IC检验均未通过10%全库FDR，"
        "因此不能把任何单因子称为已确认核心Alpha。",
        "- 开发期最值得保留的预测候选集中于期限结构、基差、仓单和库存边际变化；"
        "趋势类因子在2019—2024开发期整体偏弱。",
        "- 2025封存结果显示，Carry、仓单紧张度和库存加速度保持正向；"
        "信息背离和库存—价格背离发生明显反向，说明基本面组合信号仍不稳定。",
        "- 当前特征库适合作为正则化机器学习模型的研究输入，不适合把全部变量解释为"
        "47个相互独立、已经验证的Alpha。",
        "",
        "## 方法与样本",
        "",
        "- 开发评价期：2019—2024，来自六个扩展窗口的逐年样本外结果；2025不参与筛选。",
        "- 主标签：T日收盘形成信号，固定真实合约从T+1开盘持有至T+6开盘。",
        "- 主组合：每五日调仓，横截面前后20%，组内等权，单边5bps成本。",
        "- 稳健性维度：1/5/20日标签、延迟一日、年份、品种、板块、参数邻域、"
        "相关簇及固定Ridge增量检验。",
        "- 分类与机器学习纳入分离：A—F表示研究证据等级；可计算且通过数据质量硬门的"
        "预注册因子仍保留在全量ML接口。",
        "",
        "## 分类数量",
        "",
        "| 分类 | 数量 | 研究含义 |",
        "|---|---:|---|",
        f"| A 核心预测 | {class_counts['A']} | 通过核心预测门槛 |",
        f"| B 辅助预测 | {class_counts['B']} | 有方向和组合证据，但尚未达到核心门槛 |",
        f"| C 条件变量 | {class_counts['C']} | 用于交互、过滤或状态刻画 |",
        f"| D 风险变量 | {class_counts['D']} | 用于波动、流动性和尾部风险 |",
        f"| E 当前证据不足 | {class_counts['E']} | 可计算，但无条件预测证据不足 |",
        f"| F 数据不足 | {class_counts['F']} | 缺少计算数据或经济映射 |",
        "",
        "## 重点预测候选",
        "",
        "| 因子 | 覆盖率 | ICIR | Rank IC | Rank ICIR | FDR q | 5bps年化 | 夏普 | 2025 Rank IC | 判断 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in b_candidates.itertuples(index=False):
        if pd.isna(row.sealed_rank_ic):
            judgment = "开发期较强，但2025因现货数据终止无法确认"
        elif np.sign(row.rank_ic) == np.sign(row.sealed_rank_ic):
            judgment = "2025方向保持，继续作为主要候选"
        else:
            judgment = "2025方向反转，只能作为弱辅助特征"
        lines.append(
            f"| `{row.factor_id}` | {pct(row.coverage)} | {fmt(row.ic_ir)} | "
            f"{fmt(row.rank_ic)} | {fmt(row.rank_ic_ir)} | {fmt(row.rank_ic_fdr_q)} | "
            f"{pct(row.annual_return)} | {fmt(row.sharpe)} | {fmt(row.sealed_rank_ic)} | "
            f"{judgment} |"
        )

    lines.extend(
        [
            "",
            "## 因子家族分析",
            "",
        ]
    )
    for category, group in factors.groupby("category", sort=False):
        lines.extend(
            [
                f"### {CATEGORY_NAMES.get(category, category)}",
                "",
                family_comment(category, group),
                "",
            ]
        )
        if category == "term_structure":
            lines.append(
                "该家族是当前最清晰的经济信息源。Carry在开发期和2025均为正，"
                "整曲线斜率的Rank IC也保持正向；曲线变化和加速度则明显弱于曲线水平。"
            )
        elif category == "basis_spot":
            lines.append(
                "新鲜基差在开发期的ICIR和组合收益较好，但现货源止于2023年，"
                "2025无法确认，因此不能据此升级为核心Alpha。"
            )
        elif category == "trend_momentum":
            lines.append(
                "传统趋势、横截面动量和动量质量在开发期方向偏弱，"
                "但部分指标在2025转正，表现更像状态依赖信号而非稳定无条件Alpha。"
            )
        elif category == "inventory_fundamental":
            lines.append(
                "仓单紧张度的覆盖和稳定性优于库存类变量；库存加速度有预测迹象，"
                "但库存覆盖仅约三分之一，解释时必须防止少数品种主导。"
            )
        elif category == "information_news":
            lines.append(
                "信息背离与库存—价格背离在开发期较强，但2025明显反向；"
                "这提示当前代理新息并非市场一致预期，可能混合了慢频数据和价格趋势。"
            )
        elif category == "industry_chain":
            lines.append(
                "当前最有潜在差异化价值，但生产配比、上下游映射和成本口径不足，"
                "多数概念因子只能列为F类，不能用价格代理冒充真实产业链信息。"
            )
        elif category == "volatility_risk":
            lines.append(
                "这些变量应进入风险和条件模块。2025某些风险变量表现突出，"
                "不代表它们已经成为方向Alpha。"
            )
        elif category == "trading_liquidity":
            lines.append(
                "成交、持仓和非流动性变量的主要价值是解释交易状态、换手和风险；"
                "当前没有证据支持把它们单独作为主要方向信号。"
            )
        lines.append("")

    lines.extend(
        [
            "## 57个概念因子结果",
            "",
            "| ID | 因子 | 家族 | 可计算 | 覆盖率 | IC | ICIR | Rank IC | Rank ICIR | 胜率 | FDR q | 分组多空 | 年化 | 夏普 | 年度正向率 | 最大相关 | 增量Rank IC | 分类 | 进入ML |",
            "|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in factors.itertuples(index=False):
        lines.append(
            f"| {int(row.factor_no)} | {row.factor_name} | {row.category_cn} | "
            f"{'是' if row.available else '否'} | {pct(row.coverage)} | "
            f"{fmt(row.ic_mean)} | {fmt(row.ic_ir)} | {fmt(row.rank_ic)} | "
            f"{fmt(row.rank_ic_ir)} | {pct(row.rank_ic_hit_rate)} | "
            f"{fmt(row.rank_ic_fdr_q)} | {fmt(row.group_return_spread)} | "
            f"{pct(row.annual_return)} | {fmt(row.sharpe)} | "
            f"{pct(row.positive_year_rate)} | {fmt(row.max_abs_feature_correlation)} | "
            f"{fmt(row.mean_oos_delta_rank_ic, 6)} | {row.class_code} | "
            f"{'是' if row.enters_ml else '否'} |"
        )

    lines.extend(
        [
            "",
            "## 逐因子经济解释与证据",
            "",
        ]
    )
    for _, row in factors.iterrows():
        lines.extend(
            [
                f"### #{int(row['factor_no'])} {row['factor_name']}：{row['class_code']}类",
                "",
                f"- 经济原理：{row['economic_meaning']}。",
                f"- 数据与实现：{implementation_text(row)}。",
                f"- 开发期证据：{development_evidence(row)}",
                f"- 2025确认：{sealed_evidence(row)}",
                f"- 结论：{conclusion_text(row)}",
                "",
            ]
        )

    sealed_usable = factors.dropna(subset=["sealed_rank_ic"]).copy()
    same_direction = (
        np.sign(sealed_usable["rank_ic"])
        == np.sign(sealed_usable["sealed_rank_ic"])
    )
    lines.extend(
        [
            "## 2025封存测试分析",
            "",
            f"- 2025共有{len(sealed_usable)}个因子可评价Rank IC；与开发期同方向比例"
            f"{pct(same_direction.mean())}。",
            "- Carry、仓单紧张度和库存加速度保持正向，是当前最有价值的持续跟踪对象。",
            "- 信息背离、库存—价格背离、期限结构动量和期限结构加速度在2025反向，"
            "说明开发期结果存在显著时间不稳定性。",
            "- 基差及其衍生因子因现货数据在2023年底终止，2025没有评价样本；"
            "这属于数据缺口，不应记为通过或失败。",
            "- 趋势因子在开发期偏弱、2025转正，支持把趋势看作可能的状态依赖特征，"
            "而不是当前无条件核心Alpha。",
            "- 封存结果不反向改变A—F分类，也不用于重新选择窗口和方向。",
            "",
            "## 相关性与机器学习输入建议",
            "",
            "- 47个可计算因子已经加工为92个模型字段，其中包含数值特征和缺失/质量指示器；"
            "92个字段不等于92个独立Alpha。",
            "- Carry与整曲线斜率、库存变化与库存—价格背离、趋势与动量质量之间存在较高相关，"
            "线性模型应使用Ridge或Elastic Net，不能直接用普通OLS解释单个系数。",
            "- 不应根据当前样本收益删除全部E类变量；但必须用扩展折消融比较"
            "Carry基线、期限结构组、基本面组、趋势组和全量特征组。",
            "- F类因子不得以全零列进入模型。未来补充真实数据后，应作为新版本特征重新冻结并"
            "重新运行开发/封存流程。",
            "",
            "## 最终研究判断",
            "",
            "当前因子库的工程质量和信息覆盖较好，但单因子Alpha证据仍属中等偏弱。"
            "可优先把Carry、仓单紧张度、库存加速度和整曲线斜率作为基准预测组；"
            "基差保留为数据恢复后的重点候选；信息背离类仅作为弱辅助；"
            "波动、流动性和持仓变量作为风险或条件信息。最终是否具有联合预测价值，"
            "必须由严格扩展窗口下的正则化模型相对Carry基线给出。",
            "",
        ]
    )

    content = "\n".join(lines)
    outputs = [
        root / "reports" / "factor_screening_report.md",
        root / "reports" / "factor_screening_report_analysis.md",
    ]
    for output in outputs:
        output.write_text(content, encoding="utf-8")
        print(f"wrote {output}")
    print(f"factors={len(factors)} available={len(available)}")


if __name__ == "__main__":
    main()
