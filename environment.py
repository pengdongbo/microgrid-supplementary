"""Two-EV microgrid simulation with a custom step API."""
import numpy as np
import gym

OBS_DIM = 17

class EV:
    """Electric Vehicle with configurable charging behavior"""
    def __init__(self, P_max, capacity=275.0):
        self.P_max = P_max
        self.capacity = capacity
        self.E = 0.0
        self.E_req = 0.0
        self.arrival = 0
        self.depart = 0
        self.type = None

    def configure_episode(
        self,
        n,
        del_t=1.0,
        arrival=None,
        depart=None,
        E_req=None,
        min_window=2,
        max_window=4,
        randomize=True,
    ):
        """Configure EV parameters for one episode"""
        if randomize:
            peak_hours = [h for h in [7, 8, 9, 10, 12, 13, 14, 15] if h < n]
            off_peak = list(set(range(n)) - set(peak_hours))

            if len(peak_hours) > 0 and np.random.rand() < 0.7:
                self.arrival = np.random.choice(peak_hours)
            else:
                self.arrival = np.random.choice(off_peak)

            window = np.random.randint(min_window, max_window + 1)
            self.depart = min(self.arrival + window, n)

            base_levels = np.array([30, 60, 90, 120])  # kWh
            base = np.random.choice(base_levels)
            noise = np.random.uniform(-5, 5)

            self.E_req = np.clip(base + noise, 10.0, self.capacity)
            self.type = base

            # Feasibility correction
            max_deliverable = (self.depart - self.arrival) * del_t * self.P_max
            self.E_req = min(self.E_req, max_deliverable)

        else:
            assert arrival is not None and depart is not None and E_req is not None

            self.arrival = max(0, min(arrival, n - 1))
            self.depart = max(self.arrival + 1, min(depart, n))
            self.E_req = np.clip(E_req, 0.0, self.capacity)

            max_deliverable = (self.depart - self.arrival) * del_t * self.P_max
            self.E_req = min(self.E_req, max_deliverable)

        self.E = 0.0

    def step(self, t, P_norm, del_t):
        """Execute charging step"""
        if t < self.arrival or t >= self.depart:
            return 0.0

        if self.E >= self.E_req:
            return 0.0

        remaining_energy = self.E_req - self.E

        P = np.clip(P_norm * self.P_max, 0.0, self.P_max)

        P = min(P, remaining_energy / del_t)

        self.E += P * del_t
        self.E = np.clip(self.E, 0.0, self.capacity)

        return P

class MicrogridEnv2EV_Baseline(gym.Env):
    """Microgrid environment with 2 EVs - CMDP formulation"""
    def __init__(self, load, solar, alpha_plus=None, alpha_minus=None):
        super().__init__()

        self.load = load
        self.solar = solar
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus

        self.n = len(load)
        self.del_t = 1.0  # 1 hour timestep

        self.Emax = 500
        self.Emin = 100
        self.E_init = 250
        self.eta_plus = 0.96
        self.eta_minus = 0.96
        self.eta_ev = 0.96  # AC-to-EV delivered-energy efficiency (independent of BESS)
        self.P_B_max = 150

        self.bess_target = self.E_init

        self.evs = [EV(P_max=75), EV(P_max=75)]

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )

    def reset(self, ev_specs=None):
        """Reset environment for new episode"""
        self.t = 0
        self.peak_load = 0.0
        self.E_B = self.E_init

        for i, ev in enumerate(self.evs):
            if ev_specs is None:
                ev.configure_episode(
                    n=self.n,
                    del_t=self.del_t,
                    min_window=2,
                    max_window=4,
                    randomize=True
                )
            else:
                arrival, depart, E_req = ev_specs[i]
                ev.configure_episode(
                    n=self.n,
                    del_t=self.del_t,
                    arrival=arrival,
                    depart=depart,
                    E_req=E_req,
                    randomize=False,
                )

        return self._obs()

    def _ev_obs(self, ev):
        """Scheduled arrival/departure and demand are observable before arrival."""
        connected = ev.arrival <= self.t < ev.depart
        if self.t < ev.arrival:
            return [0.0, float(ev.arrival - self.t),
                    float(ev.depart - self.t), float(ev.E_req - ev.E)]
        return [float(connected), 0.0, float(max(0, ev.depart - self.t)),
                float(max(0.0, ev.E_req - ev.E)) if connected else 0.0]

    def _obs(self):
        t = min(self.t, self.n - 1)
        common = [self.E_B, self.load[t], self.solar[t], self.alpha_plus[t],
                  self.alpha_minus[t], np.sin(2 * np.pi * t / self.n),
                  np.cos(2 * np.pi * t / self.n), self.evs[0].E, self.evs[1].E]
        return np.array(common + self._ev_obs(self.evs[0]) + self._ev_obs(self.evs[1]),
                        dtype=np.float32)

    def step(self, action):
        """Execute one step in environment"""
        a_bess, a_ev1, a_ev2 = np.clip(action, -1, 1)

        P_EV1, P_EV2, P_EV_total = self._update_ev(a_ev1, a_ev2)

        # Power-balance baseline.
        P_base = -(
            self.load[self.t]
            + P_EV_total / self.eta_ev
            - self.solar[self.t]
        )

        # RL residual
        P_res = 0.1 * a_bess * self.P_B_max

        P_cmd = np.clip(P_base, -self.P_B_max, self.P_B_max) + P_res
        P_cmd = np.clip(P_cmd, -self.P_B_max, self.P_B_max)

        P_B = self._update_bess(P_cmd)

        P_G = self._compute_grid_power(P_B, P_EV_total)
        P_net = self._compute_net_load(P_B, P_EV_total)

        cost, ev_cost = self._compute_cost(P_G, P_B, P_EV_total)

        reward = -cost

        # Constraint cost
        c_t = 0.0
        for ev in self.evs:
            if self.t + 1 == ev.depart:
                if ev.E < ev.E_req:
                    c_t += ev.E_req - ev.E

        self.t += 1
        done = self.t >= self.n

        c_terminal = 0.0  # No terminal BESS penalty.

        info = {
            "P_G": P_G,
            "P_B": P_B,
            "E_B": self.E_B,
            "EV1_E": self.evs[0].E,
            "EV2_E": self.evs[1].E,
            "EV1_P": P_EV1,
            "EV2_P": P_EV2,
            "cost": cost,
            "ev_cost": ev_cost,
            "reward_cost": reward,
            "reward_terminal": c_terminal,
            "reward_total": reward,
            "constraint": c_t,
            "EV1_unmet": max(0.0, self.evs[0].E_req - self.evs[0].E) if self.t == self.evs[0].depart else 0.0,
            "EV2_unmet": max(0.0, self.evs[1].E_req - self.evs[1].E) if self.t == self.evs[1].depart else 0.0,
        }

        return self._obs(), reward, done, info, c_t

    def _update_bess(self, P_cmd):
        """Update BESS state"""
        P_cmd = np.clip(P_cmd, -self.P_B_max, self.P_B_max)

        max_charge = (self.Emax - self.E_B) / (self.del_t * self.eta_plus)
        max_discharge = (self.E_B - self.Emin) * self.eta_minus / self.del_t

        if P_cmd >= 0:
            P_B = min(P_cmd, max_charge)
            self.E_B += self.del_t * self.eta_plus * P_B
        else:
            P_B = max(P_cmd, -max_discharge)
            self.E_B += self.del_t * P_B / self.eta_minus

        self.E_B = np.clip(self.E_B, self.Emin, self.Emax)

        return P_B

    def _update_ev(self, a_ev1, a_ev2):
        """Update EV states"""
        EV1_frac = (a_ev1 + 1) / 2
        EV2_frac = (a_ev2 + 1) / 2

        P_EV1 = self.evs[0].step(self.t, EV1_frac, self.del_t)
        P_EV2 = self.evs[1].step(self.t, EV2_frac, self.del_t)

        return P_EV1, P_EV2, P_EV1 + P_EV2

    def _compute_grid_power(self, P_B, P_EV_total):
        """AC bus convention: P_B positive charges, negative discharges.

        eta_plus/eta_minus are applied in BESS state dynamics.
        eta_ev converts delivered EV kWh to AC-side input demand.
        Positive P_G denotes export, negative denotes import.
        """
        return (self.solar[self.t] - self.load[self.t]
                - P_EV_total / self.eta_ev - P_B)

    def _compute_net_load(self, P_B, P_EV_total):
        return -self._compute_grid_power(P_B, P_EV_total)

    def _compute_cost(self, P_G, P_B, P_EV_total):
        buy_cost = self.alpha_plus[self.t] * max(-P_G, 0.0) * self.del_t
        sell_revenue = self.alpha_minus[self.t] * max(P_G, 0.0) * self.del_t
        net_cost = buy_cost - sell_revenue

        # Diagnostic incremental EV cost at the same BESS dispatch.
        P_G_no_EV = self.solar[self.t] - self.load[self.t] - P_B
        cost_no_EV = (self.alpha_plus[self.t] * max(-P_G_no_EV, 0.0)
                      - self.alpha_minus[self.t] * max(P_G_no_EV, 0.0)) * self.del_t
        return net_cost, net_cost - cost_no_EV
