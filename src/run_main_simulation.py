# %%
# # GTFT Simulation Runner Notebook
# This notebook wraps the command-line functionality of `main.py` and runs it with custom arguments.

# %%
import sys
import os
from datetime import datetime
import subprocess

# Set the path to main.py
main_script = os.path.abspath("src/main.py")  # Adjust if main.py is elsewhere
assert os.path.isfile(main_script), f"main.py not found at {main_script}"

# %%
# ## Configuration Section
# You can change these values to experiment with different settings

# Simulation settings
rounds = [100]                       # List of rounds
forgiveness_values = [0.1, 0.3, 0.5] # List of forgiveness probabilities
error_rates = [0.01, 0.05, 0.1]      # List of error rates

# Simulation mode
parameter_sweep = True              # Set to False to run a single simulation
parallel_mode = "thread"            # "thread" or "process"

# Output settings
results_dir = "results"
log_level = "INFO"
random_seed = 42                    # Set to None for non-deterministic

# %%
# ## Construct CLI arguments for main.py
args = [
    "python", main_script,
    "--rounds", *map(str, rounds),
    "--forgiveness", *map(str, forgiveness_values),
    "--error", *map(str, error_rates),
    "--results_dir", results_dir,
    "--parallel", parallel_mode,
    "--log_level", log_level
]

if parameter_sweep:
    args.append("--sweep")

if random_seed is not None:
    args += ["--seed", str(random_seed)]

# Display the command for reference
print("Running with command:\n", " ".join(args))

# %%
# ## Run the main simulation script
try:
    result = subprocess.run(args, capture_output=True, text=True, check=True)
    print("✅ Simulation completed successfully.\n")
    print(result.stdout)
except subprocess.CalledProcessError as e:
    print("❌ An error occurred while running the simulation.")
    print("Error message:\n", e.stderr)

# %%
# ## View output folder contents
from pathlib import Path
import shutil

# Show the latest created folder in the results directory
base_path = Path(results_dir)
assert base_path.exists(), "Results directory not found."

latest_folder = max(base_path.glob("*"), key=os.path.getmtime)
print(f"📁 Latest output folder: {latest_folder}")

# List contents
for file in sorted(latest_folder.glob("*")):
    print("-", file.name)

# %%
# ## Optional: View a result summary CSV (if exists)
import pandas as pd

csv_files = list(latest_folder.glob("*.csv"))
if csv_files:
    df = pd.read_csv(csv_files[0])
    display(df.head())
else:
    print("No CSV results found.")
