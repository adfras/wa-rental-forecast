# 2025-10-15 Forecast Run – Lodgement-Weighted Calibration

**Context.** Heavy walk-forward fit (Mar–Aug 2025) with supply-shock features, raw-probability isotonic calibration, and the new lodgement-weighting scheme (`LODGE_SMOOTHING=12`, `LODGE_WEIGHT_FLOOR=0.05`). PyMC ran 6 chains × 1 800 tune/draws without divergences; a final single-chain diagnostic pass was also emitted by the validator.

## Headline Metrics (Mar–Aug 2025)

| Metric | Sept 22 Baseline (ID 3) | Current Run (ID 8) | Δ |
| --- | --- | --- | --- |
| precision@10 | 0.800 | **0.833** | +0.033 |
| precision@20 | 0.758 | **0.758** | +0.000 |
| precision@50 | 0.663 | **0.683** | +0.020 |
| lift@10 | 2.35× | **2.43×** | +0.08× |
| lift@20 | 2.22× | **2.22×** | +0.00× |
| lift@50 | 1.87× | **1.92×** | +0.05× |
| AUC | **0.713** | 0.704 | −0.009 |
| Brier | 0.213 | **0.202** | −0.011 |

Precision/lift gains at the top of the ranking are retained from the supply-shock work, and lodgement weighting shaved a further 0.011 off the Brier loss despite a minor AUC dip (expected—the model is less confident on low-coverage spikes).

## Behaviour by Lodgement Coverage (Aug 2025)

| Bucket | Definition | Count (labelled) | Base Rate | AUC | Brier | Precision@10 |
| --- | --- | ---: | ---: | --- | --- | --- |
| Thin | `lodgement_weight` < 0.2 | 32 | 0.56 | 0.71 | 0.26 | 0.80 |
| Mid | 0.2 ≤ weight < 0.6 | 90 | 0.58 | 0.70 | 0.28 | 0.70 |
| Thick | weight ≥ 0.6 | 136 | 0.45 | 0.82 | 0.21 | 0.90 |

Thin and mid buckets now receive down-weighted logits and shrunk momentum features, so their volatility no longer dominates the overall calibration; thick coverage remains the most discriminative as expected.

## Top-K Snapshot (Aug 2025)

- Top 10 probabilities are all realised positives (precision@10 = 1.0) covering Perth CBD fringe (Highgate, Northbridge), affluent coastal SA2s (Mosman Park, Cottesloe, Sorrento–Marmion), and strong regional centres (Merredin, Roebourne).
- Precision@20 = 0.75 and precision@30 = 0.77 (23/30 hits). False positives at these cut-offs concentrate in known thin-data areas (City Beach, Toodyay, Serpentine–Jarrahdale, Mukinbudin, York–Beverley, Kewdale Commercial/Airport).

## Notable Changes vs Previous Run

1. **Lodgement-weighted features** – multiplicative shrinkage on `rent_mom_*`, rent-spike interactions, and coverage-weighted likelihood prevent extreme jumps from four lodgements from overwhelming the fit (key for City Beach, Morawa, Mukinbudin).
2. **Cluster-specific isotonic** – continues to cap overconfidence in high-price clusters while keeping low-price buckets within ~10 pp bias, preserving the precision@10 uplift achieved earlier.
3. **Top-K stability** – despite damping thin regions, the high-coverage SA2s still surface at the top with higher probabilities (e.g., Roebourne 0.79 vs 0.62 pre-weighting) and zero divergences across chains ensure reliable diagnostics.

## Remaining Gaps & Next Steps

- **Data quality:** City Beach and Morawa still reflect noisy source medians; the model now reports them at 0.72 and 0.17 respectively, but underlying volatility remains. Flag for upstream smoothing or manual review.
- **Cluster 1–2 precision:** mid-price clusters deliver only 0.37–0.41 precision; investigate additional external features or region-level pooling that better distinguishes these segments.
- **Pandas warning:** `add_time_since_spike` raises a future `groupby.apply` warning—switch to `include_groups=False` or vectorised logic before the pandas 3.x cut-over.

Overall, the current configuration is production-ready: it keeps the precision improvements from September’s feature work, introduces robustness for low-lodgement SA2s, and tightens calibration without sacrificing top-K recall.
