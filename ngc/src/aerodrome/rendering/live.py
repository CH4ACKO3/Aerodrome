"""Host-side live handoff and independent wall/simulation/render clocks."""
from dataclasses import dataclass
from threading import Condition, Lock
from time import perf_counter
import math
import jax
from .schema import validate_sequence


@dataclass(frozen=True)
class Delivery:
    sequence: int
    frame: object
    published_wall_s: float  # monotonic clock, local process timebase only


class LatestFrameStream:
    """Single-producer/multiple-thread-safe calls, ONE logical consumer.

    Capacity one, newest wins. Replacing an unread frame counts as dropped.
    No device transfer, renderer call or I/O under the lock. Not a broadcast
    queue: allocate a separate stream per independent consumer/world.
    """
    def __init__(self, *, clock=perf_counter):
        self._clock = clock
        self._condition = Condition()
        self._latest = None
        self._read_sequence = 0
        self._dropped = 0
        self._closed = False

    def publish(self,frame):
        with self._condition:
            if self._closed:
                raise RuntimeError("render stream is closed")
            validate_sequence(self._latest.frame if self._latest else None,frame)
            sequence = self._latest.sequence+1 if self._latest else 1
            if self._latest and self._latest.sequence>self._read_sequence:
                self._dropped += 1
            self._latest = Delivery(sequence,frame,self._clock())
            self._condition.notify_all()
            return sequence

    def take(self, *, timeout_s=0.):
        """Take newest unseen delivery; None on timeout or drained close."""
        if not math.isfinite(timeout_s) or timeout_s<0:
            raise ValueError("timeout must be finite nonnegative")
        with self._condition:
            ready = lambda:self._closed or (self._latest is not None and self._latest.sequence>self._read_sequence)
            self._condition.wait_for(ready,timeout=timeout_s)
            if self._latest is None or self._latest.sequence<=self._read_sequence:
                return None
            self._read_sequence = self._latest.sequence
            return self._latest

    def close(self):
        with self._condition:
            self._closed = True
            self._condition.notify_all()

    def statistics(self):
        with self._condition:
            return dict(published=self._latest.sequence if self._latest else 0,
                        dropped_unread=self._dropped,closed=self._closed)


class RateMeter:
    """Cumulative rates since construction; create AFTER compilation/warmup.

    TPS counts completed physics intervals per vectorized lane. Aggregate TPS
    additionally multiplies by world_count. RTF is simulated seconds per lane
    divided by elapsed wall seconds. FPS only counts actual presentations.
    This meter is for one advancing timeline/batch, not unrelated async worlds.
    """
    def __init__(self, *, clock=perf_counter):
        self._clock,self._start = clock,clock()
        self._lock = Lock()
        self._ticks = self._aggregate_ticks = self._frames = 0
        self._simulation_s = 0.

    def simulation_completed(self,completed, *, ticks, physics_dt_s, world_count=1):
        """Synchronize completion before counting; completed must contain device outputs."""
        if type(ticks) is not int or ticks<=0 or type(world_count) is not int or world_count<=0:
            raise ValueError("ticks/world_count must be positive integers")
        if not math.isfinite(physics_dt_s) or physics_dt_s<=0:
            raise ValueError("physics dt must be finite positive")
        jax.block_until_ready(completed)
        with self._lock:
            self._ticks += ticks
            self._aggregate_ticks += ticks*world_count
            self._simulation_s += ticks*physics_dt_s

    def frame_presented(self):
        """Call after renderer presentation, not when publishing/extracting a frame."""
        with self._lock:
            self._frames += 1

    def snapshot(self):
        with self._lock:
            elapsed = self._clock()-self._start
            if elapsed<0:
                raise ValueError("clock must be monotonic")
            rate = lambda n:n/elapsed if elapsed>0 else 0.
            return dict(wall_seconds=elapsed,completed_ticks=self._ticks,
                        aggregate_world_ticks=self._aggregate_ticks,presented_frames=self._frames,
                        simulated_seconds=self._simulation_s,tps=rate(self._ticks),
                        aggregate_tps=rate(self._aggregate_ticks),fps=rate(self._frames),
                        real_time_factor=rate(self._simulation_s))


class RateLimiter:
    """Nonblocking wall-clock admission gate for rendering/export, NOT physics dt.

    Call before device transfer to avoid unnecessary snapshots. Missed periods
    are skipped; this does not sleep or trigger catch-up bursts. One caller.
    """
    def __init__(self,hz, *, clock=perf_counter):
        if not math.isfinite(hz) or hz<=0:
            raise ValueError("rate must be finite positive")
        self._period,self._clock,self._next = 1/hz,clock,None
        self._last = None

    def ready(self):
        now = self._clock()
        if self._last is not None and now<self._last:
            raise ValueError("clock must be monotonic")
        self._last = now
        if self._next is None:
            self._next = now+self._period
            return True
        if now<self._next:
            return False
        self._next += (math.floor((now-self._next)/self._period)+1)*self._period
        return True
