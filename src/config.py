from dataclasses import dataclass


@dataclass
class BSParams:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    S_min: float = 0.0
    S_max: float = 160
    T: float = 1.0


@dataclass
class PinnConfig:
    pass
