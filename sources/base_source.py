from abc import ABC, abstractmethod

from schemas.telemetry import CanonicalTelemetry


class TelemetrySource(ABC):
    @abstractmethod
    def get_event(self) -> CanonicalTelemetry:
        raise NotImplementedError

    @abstractmethod
    def get_stream(self):
        raise NotImplementedError
