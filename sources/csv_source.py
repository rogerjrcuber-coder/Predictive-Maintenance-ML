from datetime import datetime

import pandas as pd

from schemas.telemetry import CanonicalTelemetry
from sources.base_source import TelemetrySource
from sources.schema_mapper import SchemaMapper


class CSVSource(TelemetrySource):
    def __init__(self, file_path):
        self.file_path = file_path
        self.mapper = SchemaMapper()

    def _read_rows(self):
        df = pd.read_csv(self.file_path)
        for _, row in df.iterrows():
            yield row.to_dict()

    def get_recent_events(self, window_size=100, chunk_size=50_000):
        if hasattr(self.file_path, "seek"):
            self.file_path.seek(0)

        recent_rows = []
        for chunk in pd.read_csv(self.file_path, chunksize=chunk_size):
            recent_rows.extend(chunk.tail(window_size).to_dict("records"))
            if len(recent_rows) > window_size:
                recent_rows = recent_rows[-window_size:]

        return [self.mapper.normalize(row) for row in recent_rows]

    def get_event(self):
        recent_events = self.get_recent_events(window_size=1)
        if not recent_events:
            raise ValueError("No rows available in CSV source")
        return recent_events[0]

    def get_stream(self):
        for row in self._read_rows():
            yield self.mapper.normalize(row)

    def get_events(self):
        return self.get_recent_events()
