"""
Synthetic network-traffic generator for intrusion detection.

Modelled on the feature structure of the NSL-KDD benchmark (the standard
academic dataset for network intrusion detection). Real labelled attack
traffic is sensitive and licence-restricted, so we generate traffic whose
statistical fingerprints match the four classic attack families:

  - normal   : ordinary connections
  - dos      : denial-of-service (flooding) -> huge counts, few bytes back
  - probe    : port/network scanning        -> many hosts, short connections
  - r2l      : remote-to-local (password/login attacks) -> failed logins
  - u2r      : user-to-root (privilege escalation)       -> rare, odd shells

The point is to demonstrate a real deep-learning intrusion-detection
pipeline. On genuine NSL-KDD the same code runs unchanged.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

ATTACK_TYPES = ["normal", "dos", "probe", "r2l", "u2r"]


def generate_traffic(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # class mix roughly reflects NSL-KDD: mostly normal + DoS, probe common,
    # r2l uncommon, u2r rare (the hardest, most dangerous class).
    proportions = {"normal": 0.53, "dos": 0.30, "probe": 0.12,
                   "r2l": 0.04, "u2r": 0.01}
    frames = []
    for attack, p in proportions.items():
        k = int(n * p)
        frames.append(_make_class(attack, k, rng))
    df = pd.concat(frames, ignore_index=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _make_class(attack: str, k: int, rng) -> pd.DataFrame:
    """Generate k rows for one attack family with realistic feature shapes."""
    if attack == "normal":
        d = dict(
            duration=rng.exponential(30, k),
            src_bytes=rng.lognormal(6, 1.5, k),
            dst_bytes=rng.lognormal(6, 1.5, k),
            count=rng.poisson(8, k),
            srv_count=rng.poisson(8, k),
            dst_host_count=rng.poisson(30, k),
            failed_logins=rng.choice([0, 1], k, p=[0.98, 0.02]),
            logged_in=rng.choice([0, 1], k, p=[0.2, 0.8]),
            same_srv_rate=rng.beta(6, 2, k),
            diff_srv_rate=rng.beta(1, 8, k),
            serror_rate=rng.beta(1, 12, k),
            root_shell=rng.choice([0, 1], k, p=[0.995, 0.005]),
        )
    elif attack == "dos":
        # flooding: massive connection counts, high error rates, tiny replies
        d = dict(
            duration=rng.exponential(2, k),
            src_bytes=rng.lognormal(3, 1.0, k),
            dst_bytes=rng.lognormal(1, 0.8, k),
            count=rng.poisson(300, k),
            srv_count=rng.poisson(300, k),
            dst_host_count=rng.poisson(250, k),
            failed_logins=rng.choice([0, 1], k, p=[0.99, 0.01]),
            logged_in=rng.choice([0, 1], k, p=[0.95, 0.05]),
            same_srv_rate=rng.beta(8, 1, k),
            diff_srv_rate=rng.beta(1, 10, k),
            serror_rate=rng.beta(8, 2, k),
            root_shell=np.zeros(k),
        )
    elif attack == "probe":
        # scanning: touches many different services/hosts, short connections
        d = dict(
            duration=rng.exponential(1, k),
            src_bytes=rng.lognormal(2, 1.0, k),
            dst_bytes=rng.lognormal(2, 1.0, k),
            count=rng.poisson(40, k),
            srv_count=rng.poisson(15, k),
            dst_host_count=rng.poisson(200, k),
            failed_logins=rng.choice([0, 1], k, p=[0.97, 0.03]),
            logged_in=rng.choice([0, 1], k, p=[0.9, 0.1]),
            same_srv_rate=rng.beta(1, 6, k),
            diff_srv_rate=rng.beta(6, 2, k),   # hits MANY different services
            serror_rate=rng.beta(3, 4, k),
            root_shell=np.zeros(k),
        )
    elif attack == "r2l":
        # remote-to-local: normal-looking but with failed login attempts
        d = dict(
            duration=rng.exponential(40, k),
            src_bytes=rng.lognormal(5, 1.5, k),
            dst_bytes=rng.lognormal(5, 1.5, k),
            count=rng.poisson(6, k),
            srv_count=rng.poisson(6, k),
            dst_host_count=rng.poisson(25, k),
            failed_logins=rng.poisson(2.5, k),   # signature: failed logins
            logged_in=rng.choice([0, 1], k, p=[0.5, 0.5]),
            same_srv_rate=rng.beta(5, 3, k),
            diff_srv_rate=rng.beta(1, 6, k),
            serror_rate=rng.beta(1, 8, k),
            root_shell=rng.choice([0, 1], k, p=[0.97, 0.03]),
        )
    else:  # u2r — privilege escalation: rare, involves root shells
        d = dict(
            duration=rng.exponential(60, k),
            src_bytes=rng.lognormal(7, 1.5, k),
            dst_bytes=rng.lognormal(7, 1.5, k),
            count=rng.poisson(4, k),
            srv_count=rng.poisson(4, k),
            dst_host_count=rng.poisson(15, k),
            failed_logins=rng.choice([0, 1], k, p=[0.9, 0.1]),
            logged_in=np.ones(k),                # must be logged in first
            same_srv_rate=rng.beta(5, 3, k),
            diff_srv_rate=rng.beta(1, 6, k),
            serror_rate=rng.beta(1, 10, k),
            root_shell=rng.choice([0, 1], k, p=[0.4, 0.6]),  # signature
        )

    df = pd.DataFrame(d)
    df["attack_type"] = attack
    df["is_attack"] = 0 if attack == "normal" else 1
    return df


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    (root / "data").mkdir(exist_ok=True)
    df = generate_traffic()
    df.to_csv(root / "data" / "network_traffic.csv", index=False)
    print(f"Generated {len(df):,} connections -> data/network_traffic.csv")
    print(f"Attack rate: {df.is_attack.mean():.1%}")
    print("\nClass distribution:")
    print(df.attack_type.value_counts())
