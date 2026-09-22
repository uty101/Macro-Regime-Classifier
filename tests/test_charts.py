"""Step 2.4: the features review chart."""

import numpy as np
import pandas as pd

from regime.charts import features_review_chart
from regime.config import load_config


def test_features_review_chart_written(tmp_path) -> None:
    cfg = load_config()
    features = list(cfg.features_core) + list(cfg.features_robustness)
    index = pd.date_range(pd.Timestamp(cfg.sample_start), pd.Timestamp(cfg.sample_end), freq="ME", name="date")
    rng = np.random.default_rng(cfg.run_seed)
    raw = pd.DataFrame(rng.normal(size=(len(index), len(features))), index=index, columns=features)
    path = tmp_path / "charts" / "features_review.png"
    features_review_chart(raw, str(path), cfg)
    assert path.exists()
    assert path.stat().st_size > 10_000
