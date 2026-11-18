import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.main import predict_trip, TripPredictionRequest, list_trips

class DummyReq:
    client = None

trips = list_trips("student_test")
print(f"Testing all {len(trips)} student-test trips through /api/predict...")

for trip_id in trips:
    req = TripPredictionRequest(trip_id=trip_id, partition="student_test")
    resp = predict_trip(req, DummyReq())
    assert resp.status_code == 200, f"Trip {trip_id} failed with {resp.status_code}"
    data = json.loads(resp.body.decode("utf-8"))
    m = data["evaluation_metrics"]
    t = data["pipeline_timing"]["total_seconds"]
    print(f"  Trip {trip_id}: Acc={m.get('hmm_accuracy_pct')}% | Cont={m.get('hmm_continuity_pct')}% | Latency={t}s -> [OK]")

print("\nALL 10 STUDENT TEST TRIPS COMPLETED 100% CLEANLY!")
