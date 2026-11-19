# Mixture of Probing Experts (MoPE)

MoPE selects, for each tenant of a multi-tenant cloud, one classical probing design for isolating faulty virtual machines under noisy verification probes. The pool of designs contains individual testing with 1, 3, and 5 repeated probes, Dorfman splitting with 1 and 3 probes, Hwang's binary splitting, recursive halving, combinatorial orthogonal matching pursuit (COMP) with confirmation, and the sequential probability ratio test at three error levels.

The router sees only context that is observable at deployment: the tenant size, a pilot-based prevalence estimate, and the calibrated activation and false-alarm probabilities of the two probe readouts. For every design, bootstrap ensembles of gradient-boosted regressors predict the error rate and the executions per node. A risk-aware top-1 rule then picks the design with the lowest predicted error plus cost weight times executions plus an uncertainty term. The regressors do not depend on the cost weight, so one fitted model serves every operator trade-off without retraining.

The whole model is in one file, `mope.py`: noise calibration and pilot prevalence estimation, fitting, scoring, routing, and realised-utility evaluation.

## Installation

MoPE needs Python 3.9 or newer with numpy, pandas, and lightgbm:

```
pip install numpy pandas lightgbm
```

Copy `mope.py` next to your code, or clone this repository and import it from the repository folder.

## Usage

Tables are pandas DataFrames with one row per tenant. The context columns are `m` (number of virtual machines), `rho_hat` (pilot prevalence estimate), `a1`, `a2` (activation probabilities), and `e1`, `e2` (false-alarm probabilities). A training table also holds `<design>_err` and `<design>_execs` for each design, namely the observed error rate and executions per node. A table that is scored with `realised_utility` also needs `pilot_execs`, the pilot executions per node.

```python
import pandas as pd
import mope

# 1. Calibration: probe noise from singleton probes on labelled nodes, then prevalence from a pilot sample.
#    Each readout array is boolean with shape (n, 2).
noise = mope.estimate_noise(readouts_bit1_faulty, readouts_bit2_faulty, readouts_clean)   # (a1, a2, e1, e2)
rho_hat = mope.estimate_prevalence(pilot_positive_rate, noise)

# 2. Fit on a table of tenant outcomes.
train = pd.read_csv("tenants_train.csv")
model = mope.MoPE().fit(train)

# 3. Score and route new tenants for an operator cost weight c.
new_tenants = pd.read_csv("tenants_new.csv")
scores = model.score(new_tenants, c=0.01)           # risk-adjusted utility of every design, shape (n, 11)
choice = model.route(new_tenants, c=0.01)           # index of the selected design
names = model.route_names(new_tenants, c=0.01)      # name of the selected design

# 4. Realised cost-weighted error of the routing on tenants with observed outcomes.
print(model.realised_utility(new_tenants, choice, c=0.01))
```

`MoPE(experts, n_boot=5, n_estimators=150, beta=1.0, seed=0)` accepts a different list of designs, and `beta` can be overridden per call in `score` and `route`.

## Headline result

On 230 simulated fleets with ten seeds, MoPE lowered the cost-weighted error by 19% relative to the best fixed design (0.0318 against 0.0393, Holm-corrected p = 0.033) and closed 70% of the gap to an oracle router. Under stale probe calibration, repeated testing remained better than the router.

## Citation

```bibtex
@unpublished{massaoudi_mope,
  author = {Massaoudi, Mohamed and Ez Eddin, Maymouna},
  title  = {Learning to Route Probing Designs per Tenant under Noisy Verification Probes},
  note   = {Manuscript}
}
```

## License

Released under the MIT License. See the `LICENSE` file.

## Contact

Mohamed Massaoudi, mohamed.massaoudi@tamu.edu
