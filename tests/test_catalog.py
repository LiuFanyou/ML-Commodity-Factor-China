from factor_screening.catalog import FACTOR_CATALOG


def test_catalog_contains_all_57_unique_factors():
    ids = [factor.factor_id for factor in FACTOR_CATALOG]
    assert ids == list(range(1, 58))
    assert len(set(ids)) == 57


def test_missing_factors_have_reasons():
    for factor in FACTOR_CATALOG:
        if factor.implementation_status in {"missing_data", "deferred"}:
            assert factor.missing_reason
