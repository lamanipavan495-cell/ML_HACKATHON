import pandas as pd

df = pd.read_csv("results/evaluator_test/per_trip_results.csv", dtype={"trip_id": str})
for idx, r in df.iterrows():
    sum_stages = (
        r["time_candidate_generation_s"]
        + r["time_feature_extraction_s"]
        + r["time_model_inference_s"]
        + r["time_hmm_decoding_s"]
    )
    diff = r["time_total_pipeline_s"] - sum_stages
    print(f"Trip {r['trip_id']}: StageSum={sum_stages:.3f}s | ReportedTotal={r['time_total_pipeline_s']:.3f}s | Overhead={diff:.3f}s")
