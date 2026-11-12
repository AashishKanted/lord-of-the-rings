"""Keeps kit tracks alive between detections using the conveyor encoder.

A kit on the belt moves exactly as far as the belt does, so between detections each track
is carried forward by the belt travel since its last detection. A detection is matched to
the track whose predicted bottle position is nearest (within ``perception.match_radius_m``;
kits are a 450 mm pitch apart), otherwise it starts a new track.

The ring is carried forward with the belt as well. That is right while it lies on the kit
and approximately right while the arm tracks the belt with it; a fresh detection replaces
the prediction either way.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from ...interfaces import ConveyorSignal
from ..frames import StationFrame
from .types import KitDetection, KitObservation


@dataclass
class _Track:
    track_id: int
    detection: KitDetection
    travel: float            # belt travel at the detection's capture time


class KitTracker:
    def __init__(self, config):
        self.cfg = config.perception
        self.reset(None)

    def reset(self, station: StationFrame | None) -> None:
        self.station = station
        self._tracks: dict[int, _Track] = {}
        self._history: deque[tuple[float, float]] = deque()   # (t, belt travel)
        self._signal: ConveyorSignal | None = None
        self._next_id = getattr(self, "_next_id", 1)          # ids stay unique across stations

    # ---- belt ----
    @property
    def belt_speed(self) -> float:
        return 0.0 if self._signal is None else self._signal.speed_mps

    def _travel_now(self, t: float) -> float:
        s = self._signal
        return 0.0 if s is None else s.encoder_m + s.speed_mps * (t - s.t)

    def _travel_at(self, t: float) -> float:
        """Belt travel at a past time, from the recent history (nearest stored tick)."""
        if not self._history:
            return 0.0
        return min(self._history, key=lambda item: abs(item[0] - t))[1]

    # ---- update ----
    def update(self, t: float, conveyors: list[ConveyorSignal],
               detections: list[KitDetection] | None, source: str) -> list[KitObservation]:
        for signal in conveyors:
            if self.station is not None and signal.station == self.station.name:
                self._signal = signal
        travel = self._travel_now(t)
        self._history.append((t, travel))
        while t - self._history[0][0] > self.cfg.history_s:
            self._history.popleft()
        if detections is not None:
            self._associate(detections)
        observations = [self._observe(track, t, travel, source) for track in self._tracks.values()]
        for obs in observations:        # drop tracks that are stale or have left the view
            gone = abs(self.station.local_x(obs.bottle_in_base)) > self.cfg.view_half_length_m + 0.1
            if gone or obs.age_s > self.cfg.track_timeout_s:
                del self._tracks[obs.track_id]
        return [obs for obs in observations if obs.track_id in self._tracks]

    def _associate(self, detections: list[KitDetection]) -> None:
        free = dict(self._tracks)
        for det in detections:
            travel = self._travel_at(det.t)
            best, best_d = None, self.cfg.match_radius_m
            for track in free.values():
                predicted = track.detection.bottle_in_base + \
                    (travel - track.travel) * self.station.belt_dir
                d = float(np.linalg.norm((predicted - det.bottle_in_base)[:2]))
                if d < best_d:
                    best, best_d = track, d
            if best is None:
                if det.ring_in_base is None:
                    continue             # a bottle alone does not start a track
                best = _Track(self._next_id, det, travel)
                self._next_id += 1
                self._tracks[best.track_id] = best
            else:
                del free[best.track_id]
                if det.ring_in_base is None:
                    # Ring picked up (or hidden): keep where it was on the kit, moved along.
                    shift = (travel - best.travel) * self.station.belt_dir
                    det = KitDetection(det.t, best.detection.ring_in_base + shift,
                                       det.bottle_in_base)
                best.detection, best.travel = det, travel

    def _observe(self, track: _Track, t: float, travel: float, source: str) -> KitObservation:
        shift = (travel - track.travel) * self.station.belt_dir
        det = track.detection
        return KitObservation(t, track.track_id, det.ring_in_base + shift,
                              det.bottle_in_base + shift, self.belt_speed * self.station.belt_dir,
                              t - det.t, source)
