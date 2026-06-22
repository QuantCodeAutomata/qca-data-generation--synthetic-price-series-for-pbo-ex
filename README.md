# CSCV / PBO Experiments — Synthetic Price Series

A Python research repository implementing **Combinatorially Symmetric Cross-Validation (CSCV)** and the **Probability of Backtest Overfitting (PBO)** methodology from Bailey et al. (2014) / López de Prado (2018).

The project validates CSCV on two synthetic return series:
- **Example 1** — pure i.i.d. random walk (no embedded signal)
- **Example 2** — random walk with a genuine monthly seasonal effect

## Repository Structure

```
.
├── data/               # Generated data artifacts (numpy .npy, CSV)
├── exp/                # Experiment runner scripts
│   ├── run_exp1.py     # Synthetic data generation
│   ├── run_exp2.py     # Strategy simulation (8,800 strategies → P&L matrices)
│   ├── run_exp3.py     # CSCV on Example 1 (random walk)
│   ├── run_exp4.py     # CSCV on Example 2 (seasonal effect)
│   ├── run_exp5.py     # Stochastic dominance & PSR for Example 1
│   └── run_exp6.py     # Stochastic dominance & PSR for Example 2 + comparison
├── src/                # Core implementation modules
│   ├── data_generation.py      # Synthetic return / price series generation
│   ├── strategy_simulation.py  # 8,800-strategy monthly seasonal trading rule
│   ├── cscv.py                 # CSCV algorithm (PBO, performance degradation)
│   ├── analysis.py             # Stochastic dominance & PSR (Bailey et al. 2014)
│   └── visualization.py        # Matplotlib/seaborn plotting helpers
├── tests/              # pytest unit and integration tests
│   ├── test_data_generation.py
│   ├── test_strategy_simulation.py
│   ├── test_cscv.py
│   └── test_analysis.py
├── results/            # Saved plots and RESULTS.md summary
└── README.md
```

## Methodology

### Experiments

| Exp | Description |
|-----|-------------|
| **exp_1** | Generate T=1000 daily log-returns for Examples 1 & 2; build business-day calendar |
| **exp_2** | Simulate all 8,800 strategy variants; produce 992×8800 P&L matrices M_ex1 / M_ex2 |
| **exp_3** | CSCV on M_ex1: PBO, performance degradation, probability of OOS loss |
| **exp_4** | CSCV on M_ex2: same metrics — expected low PBO (~13%) |
| **exp_5** | Stochastic dominance (Dist A vs B) & PSR for Example 1 |
| **exp_6** | Stochastic dominance & PSR for Example 2; full comparison table |

### Parameter Grid (8,800 strategies)

| Parameter | Values |
|-----------|--------|
| `entry_day` | 1 … 20 (ordinal business day of month) |
| `holding_period` | 1 … 20 (days) |
| `stop_loss` | 0 … 10 (multiples of σ_daily; 0 = no stop) |
| `side` | +1 (long), −1 (short) |

### CSCV Algorithm

1. Partition the 992×8800 P&L matrix into **S=16** equal slices.
2. Enumerate all **C(16,8) = 12,870** IS/OOS split combinations.
3. For each combination, compute Sharpe ratios for all 8,800 strategies; find IS-optimal strategy *n\**.
4. Compute the **logit of the relative OOS rank** λ_c = logit(ω_c).
5. **PBO** φ = fraction of λ_c < 0.

### PSR (Probabilistic Sharpe Ratio)

Using the Bailey & López de Prado (2014) formula with excess kurtosis convention:

```
PSR-Stat = SR̂_daily × √(T−1) / √[1 − γ₃·SR̂ + (γ₄_excess+2)/4 · SR̂²]
PSR = Φ(PSR-Stat)
```

## Requirements

```
numpy>=1.20
pandas>=1.3
scipy>=1.7
matplotlib
seaborn
```

No external data sources required — all data is synthetically generated.

## Running the Experiments

```bash
# Generate synthetic data
python exp/run_exp1.py

# Simulate all 8,800 strategies (may take ~5 min)
python exp/run_exp2.py

# CSCV analysis
python exp/run_exp3.py   # Example 1
python exp/run_exp4.py   # Example 2

# Stochastic dominance & PSR
python exp/run_exp5.py   # Example 1 (must run before exp_6)
python exp/run_exp6.py   # Example 2 + comparison
```

## Running Tests

```bash
pytest tests/ -v
```

## Key Results

| Metric | Example 1 (Random Walk) | Example 2 (Seasonal) |
|--------|------------------------|----------------------|
| **PBO** | ~0.79 | ~0.17 |
| **Prob(OOS loss)** | ~0.79 | ~0.17 |
| **PSR_stat** | ~2.83 | ~3.0+ |
| **FO Stochastic Dominance** | B dominates A | No strict FO |
| **SO Stochastic Dominance** | B dominates A | Partially A>B |

See `results/RESULTS.md` for the full metric table and figure descriptions.

## References

- Bailey, D. H., Borwein, J., López de Prado, M., & Zhu, Q. J. (2014). *Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance*. Notices of the AMS.
- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
