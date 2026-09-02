"""stats_buckets: quartile computation over per-collection means, sentinel
handling for min_pmi, un-log for collection_size and avg_doc_length."""
import pandas as pd

from scripts.stats_buckets import (
    MIN_PMI_SENTINEL, compute_quartiles, render_stats_block,
)


def _synth_stats() -> pd.DataFrame:
    """Four collections, per-collection means chosen so quartiles are known:
    for each feature the four values are 0.1, 0.3, 0.5, 0.7 -> q25=0.25,
    q50=0.4, q75=0.55."""
    rows = []
    for ds, v in [("A", 0.1), ("B", 0.3), ("C", 0.5), ("D", 0.7)]:
        rows.append(dict(dataset=ds, avg_idf=v, max_idf=v, vocab_overlap=v,
                         avg_doc_length=v, mean_pmi=v, oov_share=0.0,
                         collection_size=4.0, min_pmi=0.0))
    return pd.DataFrame(rows)


def test_quartiles_over_per_collection_means():
    q = compute_quartiles(_synth_stats())
    for feat in ("avg_idf", "max_idf", "vocab_overlap", "avg_doc_length", "mean_pmi"):
        assert abs(q.loc[feat, "q25"] - 0.25) < 1e-9
        assert abs(q.loc[feat, "q50"] - 0.40) < 1e-9
        assert abs(q.loc[feat, "q75"] - 0.55) < 1e-9


def test_render_places_query_row_into_correct_bucket():
    q = compute_quartiles(_synth_stats())

    # value 0.05 is below q25 -> "low" for avg_idf, "short" for avg_doc_length
    low_row = pd.Series(dict(avg_idf=0.05, max_idf=0.05, oov_share=0.0,
                             collection_size=4.0, avg_doc_length=0.05,
                             vocab_overlap=0.05, mean_pmi=0.05, min_pmi=0.0))
    out = render_stats_block(low_row, quartiles=q)
    assert "low (bottom quartile" in out
    assert "short (bottom quartile" in out
    assert "weak (bottom quartile" in out            # mean_pmi

    # value 0.9 is above q75 -> "very high" / "very long" / "very strong"
    high_row = low_row.copy(); high_row[:] = 0.9
    high_row["oov_share"] = 0.0; high_row["collection_size"] = 4.0
    high_row["min_pmi"] = 0.9
    out = render_stats_block(high_row, quartiles=q)
    assert "very high (top quartile" in out
    assert "very long (top quartile" in out
    assert "very strong (top quartile" in out


def test_min_pmi_sentinel_gets_named_bucket():
    q = compute_quartiles(_synth_stats())
    row = pd.Series(dict(avg_idf=0.4, max_idf=0.4, oov_share=0.1,
                         collection_size=4.7, avg_doc_length=2.0, vocab_overlap=0.4,
                         mean_pmi=0.4, min_pmi=MIN_PMI_SENTINEL))
    out = render_stats_block(row, quartiles=q)
    assert "never-co-occurring term pair" in out
    # sentinel must not leak as a raw number
    assert "-1.0" not in out and "-1.00" not in out


def test_collection_size_and_doc_length_are_un_logged():
    q = compute_quartiles(_synth_stats())
    # log10(1000000 + 1) ≈ 6.0 -> "~1.0M documents"
    row = pd.Series(dict(avg_idf=0.4, max_idf=0.4, oov_share=0.0,
                         collection_size=6.0, avg_doc_length=2.0, vocab_overlap=0.4,
                         mean_pmi=0.4, min_pmi=0.0))
    out = render_stats_block(row, quartiles=q)
    assert "~1.0M documents" in out
    # log10(100 + 1) ≈ 2.004 -> "~99 tokens" (since 10^2 - 1 = 99)
    assert "~99 tokens" in out


def test_oov_zero_gets_natural_language():
    q = compute_quartiles(_synth_stats())
    row = pd.Series(dict(avg_idf=0.4, max_idf=0.4, oov_share=0.0,
                         collection_size=4.7, avg_doc_length=2.0, vocab_overlap=0.4,
                         mean_pmi=0.4, min_pmi=0.0))
    out = render_stats_block(row, quartiles=q)
    assert "no query terms absent" in out
