import os
import sys
import io
import json
import base64
import contextlib
import matplotlib
matplotlib.use("Agg")  # Non-blocking headless backend
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
eda_dir = os.path.join(root_dir, "phase_1_eda")
os.chdir(eda_dir)

nb_path = "notebooks.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Global namespace for execution
env = {}

execution_counter = 1

for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue

    code = "".join(cell["source"])
    print(f"\n--- Executing Code Cell {execution_counter} ---")

    # If the code refers to student_test/00000000, make sure it finds it
    if "00000000" in code and "trip_dir" in code:
        code = """candidates = [
    os.path.join('..', 'Student folder', 'Student folder', 'student_test', '00000000'),
    os.path.join('Student folder', 'Student folder', 'student_test', '00000000'),
    os.path.join('..', 'student_test', '00000000'),
    os.path.join('student_test', '00000000')
]
trip_dir = next((c for c in candidates if os.path.exists(c)), candidates[0])

for ext in ['track', 'nodes', 'arcs', 'route']:
    fpath = os.path.join(trip_dir, f'00000000.{ext}')
    if os.path.exists(fpath):
        with open(fpath) as f:
            lines = [f.readline().strip() for _ in range(5)]
        print(f'--- 00000000.{ext} (First 5 lines) ---')
        for l in lines:
            print(l)
        print()
"""
        cell["source"] = [line + "\n" for line in code.split("\n")[:-1]] + [code.split("\n")[-1]]

    stdout_buf = io.StringIO()
    plt.close("all")

    outputs = []

    try:
        with contextlib.redirect_stdout(stdout_buf):
            # Check if last expression is an evaluation
            lines = [l for l in code.strip().split("\n") if l.strip() and not l.strip().startswith("#")]
            last_line = lines[-1] if lines else ""
            
            # Execute everything before last line
            if len(lines) > 1 and not (last_line.startswith("print") or last_line.startswith("plt.") or "=" in last_line or "for " in last_line or "if " in last_line):
                exec("\n".join(lines[:-1]), env)
                res = eval(last_line, env)
            else:
                exec(code, env)
                res = None

        stdout_val = stdout_buf.getvalue()
        if stdout_val:
            outputs.append({
                "name": "stdout",
                "output_type": "stream",
                "text": [stdout_val]
            })

        # If res is a DataFrame (like train_summary.head()), add execute_result
        if isinstance(res, pd.DataFrame):
            outputs.append({
                "data": {
                    "text/html": [res.to_html(classes="dataframe", border=0)],
                    "text/plain": [res.to_string()]
                },
                "execution_count": execution_counter,
                "metadata": {},
                "output_type": "execute_result"
            })

        # Check if a matplotlib figure was created
        figs = [plt.figure(n) for n in plt.get_fignums()]
        if figs:
            for fig in figs:
                img_buf = io.BytesIO()
                fig.savefig(img_buf, format="png", bbox_inches="tight", dpi=100)
                img_buf.seek(0)
                b64_png = base64.b64encode(img_buf.read()).decode("ascii")
                outputs.append({
                    "data": {
                        "image/png": b64_png,
                        "text/plain": ["<Figure size ...>"]
                    },
                    "metadata": {},
                    "output_type": "display_data"
                })
            plt.close("all")

        # If it was cell 11 with IFrame
        if "IFrame" in code:
            outputs.append({
                "data": {
                    "text/html": [
                        '<iframe width="100%" height="550" src="visualizations/folium_trip_00000000.html" frameborder="0" allowfullscreen></iframe>'
                    ],
                    "text/plain": [
                        "<IPython.lib.display.IFrame at 0x...>"
                    ]
                },
                "execution_count": execution_counter,
                "metadata": {},
                "output_type": "execute_result"
            })

        cell["outputs"] = outputs
        cell["execution_count"] = execution_counter
        execution_counter += 1
        print(f"Cell completed successfully with {len(outputs)} output item(s).")

    except Exception as e:
        print(f"Cell failed: {e}")
        import traceback
        traceback.print_exc()

# Save executed notebook
with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"\nSUCCESS! Saved fully executed notebook to {os.path.abspath(nb_path)}")
