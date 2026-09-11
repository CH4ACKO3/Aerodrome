"""Ideal-state-feedback flight experiment; navigation/actuators are explicit ideals."""
from dataclasses import dataclass
from typing import NamedTuple,Any
import jax
import jax.numpy as jnp
from aerodrome.core.integrators import rk4
from aerodrome.models.f16_longitudinal import rhs,Airframe
from .contracts import SystemSpec


class FlightParameters(NamedTuple):
    trim_state: Any
    trim_input: Any
    gain: Any
    airframe: Any


class FlightState(NamedTuple):
    aircraft: Any
    held_input: Any


class FlightRecord(NamedTuple):
    aircraft: Any
    observation: Any
    command: Any
    applied: Any
    saturated: Any


@dataclass(frozen=True)
class F16LongitudinalAssembly:
    backend: str = "jax"
    initial_signals: tuple = ()
    systems: tuple = (SystemSpec("ideal_navigation","algebraic",writes=("state_estimate",)),
                      SystemSpec("trim_regulation","discrete",reads=("state_estimate",),state_slots=("held_input",)),
                      SystemSpec("longitudinal_airframe","continuous",state_slots=("aircraft",)))

    def initialize(self,initial,key):
        return initial

    def make_tick(self,schedule):
        def tick(state,reference,p,context):
            observation = state.aircraft  # Explicit ideal estimator, not a sensor-fusion baseline.
            command = p.trim_input-p.gain@(observation-reference)
            limited = jnp.clip(command,jnp.array([-jnp.deg2rad(25.),0.]),jnp.array([jnp.deg2rad(25.),19000*4.4482216152605]))
            applied = jnp.where(context.tick%schedule.control_every == 0,limited,state.held_input)
            following = rk4(lambda t,x,u,airframe:rhs(x,u,context.resources,airframe),
                            context.tick*context.dt_s,state.aircraft,applied,p.airframe,context.dt_s)
            record = FlightRecord(state.aircraft,observation,command,applied,jnp.any(command != limited))
            return FlightState(following,applied),record
        return tick
