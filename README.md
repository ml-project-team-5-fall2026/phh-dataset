# Poker Hand Histories
Modified version of the poker hand histories from Juho Kim for our machine learning application.

# Usage

Ensure you are in a venv:

> python3 -m venv .venv
> source .venv/bin/activate

Ensure dependencies are installed:

> pip install -r requirements.txt

Or if you are in the Coursera environment:

> pip install -r requirements_coursera.txt

## Data Cleaning/Scripts

phh_to_parquet.py to generate a database format more compatible with Python/Pandas, basic usage is:

> ./scripts/phh_to_parquet.py --data_path data/ --workers 1

---

metrics.py to retrieve statistics on the contents of the database, basic usage is:

> ./scripts/metrics.py --data_path data/

Can also run on `--data_path parquet/` if it was generated.


---

train_preflop.py to train a torch network that predicts pre-flop actions (fold, call, raise) from the Parquet decisions table, basic usage is:

> ./scripts/train_preflop.py --data_path parquet/decisions

If you don't have an NVIDIA GPU, install the smaller CPU-only torch build first:

> pip install torch --index-url https://download.pytorch.org/whl/cpu

Can change `/cpu` to `/xpu` if you have decent integrated graphics.
