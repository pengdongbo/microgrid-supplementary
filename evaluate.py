"""Deterministic 24-hour evaluation of a saved actor."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import pandas as pd
from common import ROOT, setup, load_data, make_env, make_agent


def evaluate(agent, env, ev_specs):
    state = env.reset(ev_specs=ev_specs)
    rows = []
    for hour in range(env.n):
        action = agent.act(state, deterministic=True).cpu().numpy().flatten()
        state, reward, done, info, cost = env.step(action)
        rows.append(dict(hour=hour, **{key: info[key] for key in ['P_G', 'P_B', 'E_B', 'EV1_E', 'EV2_E', 'EV1_P', 'EV2_P', 'cost', 'constraint']}))
        if done:
            break
    summary = dict(net_cost_usd=sum(r['cost'] for r in rows),
                   unmet_ev_kwh=sum(r['constraint'] for r in rows),
                   mean_constraint_kwh_per_step=sum(r['constraint'] for r in rows)/env.n,
                   terminal_bess_kwh=float(env.E_B))
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/default.json')
    parser.add_argument('--data', type=Path, default=ROOT/'data/test_winter_hourly.csv')
    parser.add_argument('--model', type=Path, default=ROOT/'models/policy.pt')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/evaluation')
    parser.add_argument('--day', type=int)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    args = parser.parse_args()
    config = setup(args.config, args.device)
    agent = make_agent(config, args.device)
    checkpoint = torch.load(args.model, map_location=args.device, weights_only=True)
    agent.actor.load_state_dict(checkpoint['actor'])
    agent.actor.eval()
    day = config['evaluation_day'] if args.day is None else args.day
    rows, summary = evaluate(agent, make_env(load_data(args.data), day), config['evaluation_ev_specs'])
    if args.output.exists():
        parser.error('output already exists; choose a new directory')
    args.output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(args.output/'rollout.csv', index=False)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
