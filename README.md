# Poker Hand Histories
Modified version of the poker hand histories from Juho Kim for our machine learning application.

# Usage

Ensure dependencies are installed:

> pip install -r requirements.txt

Or if you are in the Coursera environment:

> pip install -r requirements_coursera.txt

## Data Cleaning/Scripts

phh_to_parquet.py to generate a database format more compatible with Python/Pandas, basic usage is:

> ./scripts/phh_to_parquet.py --data_path data/

---

metrics.py to retrieve statistics on the contents of the database, basic usage is:

> ./scripts/metrics.py --data_path data/

Can also run on `--data_path parquet/` if it was generated.

