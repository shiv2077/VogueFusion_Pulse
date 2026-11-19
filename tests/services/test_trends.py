import json

import pytest

from voguefusion import config
from voguefusion.trends import (
    build_store,
    clean,
    document_frequency,
    parse_tags,
    read_captions,
    season_key,
    trend_lift,
    write_store,
)

CFG = config.load(str(config.DEFAULT_CONFIG))


def test_parse_tags_strips_lowercases_dedupes():
    assert parse_tags(" Flower, no humans,flower, , Leaf ") == ["flower", "no humans", "leaf"]


def test_document_frequency_counts_documents_not_mentions():
    assert document_frequency([["a", "a", "b"], ["a"]]) == {"a": 2, "b": 1}


def test_fall_and_autumn_are_the_same_season_and_order_is_chronological():
    assert season_key("Fall 2021") == season_key("Autumn 2021")
    labels = ["Winter 2021", "Fall 2020", "Spring 2021", "Summer 2020"]
    assert sorted(labels, key=season_key) == [
        "Summer 2020",
        "Fall 2020",
        "Winter 2021",
        "Spring 2021",
    ]
    with pytest.raises(ValueError):
        season_key("Monsoon 2021")


def test_trend_lift_closed_form():
    # Baseline (2 seasons, 4 items): paisley in 1/4. Recent (1 season, 2 items): paisley in 2/2.
    records = [
        ("Spring 2020", ["paisley", "floral"]),
        ("Spring 2020", ["floral"]),
        ("Summer 2020", ["floral"]),
        ("Summer 2020", ["checks"]),
        ("Fall 2020", ["paisley"]),
        ("Autumn 2020", ["paisley", "floral"]),
    ]
    rows = {r["keyword"]: r for r in trend_lift(records, recent_seasons=1, min_support=1)}
    assert rows["paisley"]["recent_share"] == 1.0
    assert rows["paisley"]["baseline_share"] == 0.25
    assert rows["paisley"]["lift"] == 4.0
    assert rows["checks"]["recent_share"] == 0.0
    assert rows["floral"]["lift"] == pytest.approx(0.5 / 0.75, abs=1e-3)


def test_trend_lift_needs_a_baseline():
    with pytest.raises(ValueError):
        trend_lift([("Spring 2020", ["a"])], recent_seasons=1, min_support=1)


def test_repo_captions_store_excludes_tagger_misfires():
    t = CFG["trends"]
    docs = read_captions(config.repo_path(t["captions_dir"]))
    assert len(docs) == 82
    store = build_store(
        source="captions",
        label="x",
        docs=docs,
        seasonal=None,
        exclude=set(t["exclude_tags"]),
        top_n=t["top_n"],
        recent_seasons=t["recent_seasons"],
        min_support=t["min_support"],
        caveats=[],
    )
    assert store["keywords"][0] == "flower"
    assert not set(store["keywords"]) & set(t["exclude_tags"])
    assert store["lift"] is None


def test_store_write_is_versioned_and_idempotent(tmp_path):
    store = build_store(
        source="s",
        label="l",
        docs=[["a", "b"], ["a"]],
        seasonal=None,
        exclude=set(),
        top_n=5,
        recent_seasons=1,
        min_support=1,
        caveats=[],
    )
    assert write_store(tmp_path, store) is True
    assert write_store(tmp_path, store) is False
    current = json.loads((tmp_path / "current.json").read_text())
    assert current == json.loads((tmp_path / f"{store['version']}.json").read_text())
    other = build_store(
        source="s",
        label="l",
        docs=[["c"]],
        seasonal=None,
        exclude=set(),
        top_n=5,
        recent_seasons=1,
        min_support=1,
        caveats=[],
    )
    assert other["version"] != store["version"]
    assert write_store(tmp_path, other) is True
    assert json.loads((tmp_path / "current.json").read_text())["keywords"] == ["c"]


def test_clean_removes_excluded():
    assert clean(["flower", "no humans"], {"no humans"}) == ["flower"]
