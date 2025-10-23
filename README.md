# File Compare Utility

Streamlit-based utility for comparing two CSV or Excel files. The app checks column names, validates data types, highlights cell-level mismatches, and builds an exportable discrepancy report.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run File_Compare_Utility.py
```

Upload both files (CSV/XLS/XLSX) to see:

- Column name alignment and missing columns
- Data type comparisons for shared columns
- Cell-level mismatches with row/column details
- Overall match percentages across columns, types, and values
- Downloadable CSV of discrepancies for further review

> Note: Row order is used for comparisons. Sort or align rows before uploading if they should match by a specific key.
