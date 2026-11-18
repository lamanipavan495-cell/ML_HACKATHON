import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from visualisation.generate_maps import build_interactive_folium_map
from preprocessing.data_loader import list_trips

def regenerate_all_test_maps():
    os.makedirs(os.path.join("results", "maps"), exist_ok=True)
    trips = list_trips("student_test")
    print(f"Regenerating optimized fast maps for all {len(trips)} student-test trips...")

    for trip_id in trips:
        t0 = time.time()
        out_path = os.path.join("results", "maps", f"trip_{trip_id}_map.html")
        build_interactive_folium_map(trip_id, "student_test", out_path)
        sz_kb = os.path.getsize(out_path) / 1024
        dur = time.time() - t0
        print(f"  Trip {trip_id}: Size = {sz_kb:.1f} KB, Generated in {dur:.2f}s -> [OK]")

    # Also regenerate sample train trips 00000001 and 00000002
    for trip_id in ["00000001", "00000002"]:
        t0 = time.time()
        out_path = os.path.join("results", "maps", f"trip_{trip_id}_map.html")
        build_interactive_folium_map(trip_id, "train", out_path)
        sz_kb = os.path.getsize(out_path) / 1024
        dur = time.time() - t0
        print(f"  Train Trip {trip_id}: Size = {sz_kb:.1f} KB, Generated in {dur:.2f}s -> [OK]")

    print("\nALL MAPS REGENERATED AND OPTIMIZED FOR INSTANT RENDERING!")

if __name__ == "__main__":
    regenerate_all_test_maps()
