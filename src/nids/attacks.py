"""Map the 39 NSL-KDD attack labels onto the four standard attack families."""

from __future__ import annotations

ATTACK_CATEGORIES: dict[str, str] = {
    # Denial of service: exhaust a resource so legitimate users are refused.
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "mailbomb": "dos",
    "processtable": "dos", "udpstorm": "dos",
    # Probing: surveillance and scanning ahead of an attack.
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "satan": "probe", "mscan": "probe", "saint": "probe",
    # Remote to local: gain local access from a remote machine.
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l",
    "multihop": "r2l", "phf": "r2l", "spy": "r2l", "warezclient": "r2l",
    "warezmaster": "r2l", "named": "r2l", "sendmail": "r2l",
    "snmpgetattack": "r2l", "snmpguess": "r2l", "xlock": "r2l",
    "xsnoop": "r2l", "worm": "r2l",
    # User to root: escalate an ordinary account to root.
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r",
    "rootkit": "u2r", "httptunnel": "u2r", "ps": "u2r", "sqlattack": "u2r",
    "xterm": "u2r",
}


def to_category(label: str) -> str:
    """Return the attack family for a raw label, or ``normal``."""
    if label == "normal":
        return "normal"
    try:
        return ATTACK_CATEGORIES[label]
    except KeyError:
        raise ValueError(f"Unknown NSL-KDD label: {label!r}") from None
