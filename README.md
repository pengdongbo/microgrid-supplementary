# Microgrid policy evaluation

Evaluation code accompanying the study on residual safe reinforcement learning for microgrid energy management. It includes a simulation environment, one-day input data, pretrained actor weights, and reference results.

Copyright @Dongbo Peng (dpeng017@ucr.edu)

## Run

Tested with Python 3.10 and CPU PyTorch 2.9.1.

```bash
pip install -r requirements.txt
python evaluate.py
```

Results are saved to `outputs/evaluation/`. For another run, use a new directory:

```bash
python evaluate.py --output outputs/repeat
```

## Files

- `environment.py`: two-EV microgrid simulation.
- `policy.py`: pretrained actor architecture and deterministic inference.
- `common.py`: configuration and input loading.
- `evaluate.py`: evaluation entry point.
- `requirements.txt`: dependency versions.
- `configs/default.json`: CPU thread count, day index, and EV schedules. Each EV entry is `[arrival_hour, departure_hour, required_kwh]`.
- `data/test_winter_hourly.csv`: 24 hourly rows. `hvac_power` and `solar_power` are in W; `alpha` is the purchase tariff in USD/kWh. The export tariff is 0.06 times the purchase tariff.
- `models/policy.pt`: actor weights.
- `results/rollout.csv`: reference hourly results.
- `results/summary.json`: reference aggregate results.

## Case and outputs

EV1 requires 120 kWh during 08:00-10:00; EV2 requires 100 kWh during 12:00-16:00. Schedules are observable before arrival. Each port delivers up to 75 kW. BESS energy bounds are 100-500 kWh, initial energy is 250 kWh, and maximum power is 150 kW. BESS and EV efficiencies are 0.96. The BESS residual action bound is 15 kW.

In `rollout.csv`, `hour` is the interval start; energy columns are end-of-interval values. `P_G` is grid power (positive for export), `P_B` is BESS power (positive for charging), `E_B` is BESS energy, `EV1_E`/`EV2_E` are delivered EV energy, and `EV1_P`/`EV2_P` are delivered charging power. Power is in kW and energy in kWh. `cost` is interval net electricity cost in USD; `constraint` is unmet EV energy recorded at departure.

Reference CPU results:

- Net electricity cost: -6.55561085 USD (negative means net revenue).
- Unmet EV energy: 0 kWh.
- Mean constraint per step: 0 kWh (total unmet energy divided by 24).


Small numerical differences may occur across platforms. This package reproduces the supplied single-day evaluation. The input profile was used during training; it is not a held-out generalization benchmark. Training code and the full training dataset are not included.
