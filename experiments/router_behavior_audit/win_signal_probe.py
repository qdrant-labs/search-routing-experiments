"""Cheap probe: on rows where base and zipf disagree, can trivial features
(zipf rarity + a few identifier flags) predict which expert wins? Out-of-fold
only — an in-sample fit on ~176 rows would beat the base rate for free."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from encoder_router.table import ZipfStats

HERE = Path(__file__).resolve().parent
BASE, ZIPF = "no_branches", "zipf_input_nocorpus"

_FLAGS = {  # the plan's identifier shortlist, cheap-regex form
    "id.cve": re.compile(r"\bCVE-\d{4}-\d{4,}\b", re.I),
    "id.uri": re.compile(r"\b(?:https?://|www\.|\w+://)\S+", re.I),
    "id.uuid": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    "id.path": re.compile(r"[/\\]\w|\w\.(?:py|js|json|md|txt|cpp|c|h|go|rs)\b", re.I),
    "id.acronym": re.compile(r"\b[A-Z]{2,}\b"),
    "id.errcode": re.compile(r"\b(?:[A-Z]+\d|\d{3,})\b"),
}
_WORD = re.compile(r"[a-z0-9]+", re.I)


def features(queries: pd.Series) -> pd.DataFrame:
    zipf = ZipfStats().frame(queries)  # reuse the serve-time rarity channel
    extra = pd.DataFrame({
        "len.tokens": queries.map(lambda q: len(_WORD.findall(str(q)))),
        "digit.share": queries.map(
            lambda q: np.mean([c.isdigit() for c in str(q)]) if q else 0.0
        ),
        **{name: queries.map(lambda q, p=pat: float(bool(p.search(str(q)))))
           for name, pat in _FLAGS.items()},
    })
    return pd.concat([zipf, extra], axis=1)


def main() -> None:
    df = pd.read_csv(HERE / "heldout_predictions.csv")
    routes_differ = df[BASE].ne(df[ZIPF])
    has_pref = df[f"{ZIPF}.captured"].ne(df[f"{BASE}.captured"])  # zero-gain rows excluded
    dis = df[routes_differ & has_pref].reset_index(drop=True)
    cap_b, cap_z = dis[f"{BASE}.captured"], dis[f"{ZIPF}.captured"]
    y = (cap_z > cap_b).astype(int).to_numpy()  # 1 = zipf wins this row

    x = features(dis["query"]).to_numpy(np.float32)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    pred = cross_val_predict(model, x, y, cv=cv)
    proba = cross_val_predict(model, x, y, cv=cv, method="predict_proba")[:, 1]

    always_base_acc = float((y == 0).mean())          # the "just keep base" bet
    cv_acc = float((pred == y).mean())
    routed = np.where(pred == 1, cap_z, cap_b)         # take zipf only when predicted
    print(f"disagreement rows: {len(dis)}  (zipf wins {int(y.sum())}, base wins {int((y==0).sum())})")
    print(f"accuracy   always-base {always_base_acc:.3f} | probe(OOF) {cv_acc:.3f}")
    print(f"captured   always-base {cap_b.mean():.4f} | probe {routed.mean():.4f} "
          f"| always-zipf {cap_z.mean():.4f} | perfect {np.maximum(cap_b, cap_z).mean():.4f}")
    if "provenance" in dis:
        for prov, g in dis.groupby(dis["provenance"].fillna("unknown")):
            yy = (g[f"{ZIPF}.captured"] > g[f"{BASE}.captured"]).astype(int)
            print(f"  [{prov:9}] n={len(g):3d}  zipf-win rate {yy.mean():.3f}")
    print("\nlogit |coef| ranking (fit on all rows, for reading only):")
    fit = model.fit(x, y)
    coef = fit.named_steps["logisticregression"].coef_[0]
    for name, c in sorted(zip(features(dis["query"]).columns, coef),
                          key=lambda kv: -abs(kv[1])):
        print(f"  {name:14} {c:+.3f}")


if __name__ == "__main__":
    main()
