# From User Part 3 - Labelled "Layer 13: Monitoring / Telemetry Layer"
# Maps to README Layer 13: Introspective Meta-Reporter (IMR) - telemetry aspect
import json # MonitoringLayer uses json

class MonitoringLayer:
    def run(self, state_dict): # state is a dict
        if "metrics" in state_dict: # metrics produced by MetricsObservationLayer
            # Basic print telemetry. A real system might use logging, send to a dashboard, etc.
            print("[Metrics]", json.dumps(state_dict["metrics"], indent=2))
