"""Host-validated multi-rate schedule. Topology is static during a rollout."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Schedule:
    physics_dt_s: float = 0.01
    sensor_every: int = 2
    guidance_every: int = 10
    control_every: int = 2

    def __post_init__(self):
        if not math.isfinite(self.physics_dt_s) or self.physics_dt_s <= 0:
            raise ValueError("physics_dt_s must be finite and positive")
        for name in ("sensor_every", "guidance_every", "control_every"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def control_dt_s(self):
        return self.physics_dt_s * self.control_every

    @property
    def guidance_dt_s(self):
        return self.physics_dt_s * self.guidance_every
