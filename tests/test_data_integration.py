from factor_screening.config import load_config
from factor_screening.data import load_panel


def test_development_loader_excludes_sealed_year():
    config = load_config()
    panel = load_panel(config, config.raw["development_end"])
    assert panel["trade_date"].max().year == 2024
    assert panel["product"].nunique() == 30
    assert not panel.duplicated(["trade_date", "product"]).any()
