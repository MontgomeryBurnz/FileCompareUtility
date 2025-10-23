from __future__ import annotations

from io import BytesIO
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import streamlit as st


SUPPORTED_EXTENSIONS = ("csv", "xls", "xlsx")
CSV_ENCODING_CANDIDATES = ("utf-8", "utf-8-sig", "cp1252", "latin-1")


def column_index_to_excel_letter(index: int) -> str:
    """Translate a zero-based column index to an Excel-style column label."""
    label: List[str] = []
    idx = index + 1
    while idx > 0:
        idx, remainder = divmod(idx - 1, 26)
        label.append(chr(ord("A") + remainder))
    return "".join(reversed(label))


def read_csv_with_fallback(
    uploaded_file: BytesIO, encodings: Iterable[str] | None = None
) -> pd.DataFrame:
    """Read a CSV file, retrying with alternative encodings when decoding fails."""
    encoding_candidates = (
        tuple(encodings) if encodings is not None else CSV_ENCODING_CANDIDATES
    )
    last_error: Exception | None = None

    for encoding in encoding_candidates:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc

    uploaded_file.seek(0)
    if last_error:
        tried = ", ".join(encoding_candidates)
        raise ValueError(
            f"Unable to decode CSV with tried encodings: {tried}"
        ) from last_error
    raise ValueError("Unable to decode CSV: no encodings supplied.")


def load_tabular_file(uploaded_file: BytesIO) -> pd.DataFrame:
    """Load supported CSV or Excel files into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return read_csv_with_fallback(uploaded_file)
    if name.endswith((".xls", ".xlsx")):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Please upload CSV or Excel files.")


def compare_column_names(
    df_left: pd.DataFrame, df_right: pd.DataFrame
) -> Dict[str, List[str]]:
    """Compare column names and report differences."""
    cols_left = set(df_left.columns)
    cols_right = set(df_right.columns)
    return {
        "matched": sorted(cols_left & cols_right),
        "missing_in_left": sorted(cols_right - cols_left),
        "missing_in_right": sorted(cols_left - cols_right),
    }


def compare_dtypes(
    df_left: pd.DataFrame, df_right: pd.DataFrame, common_columns: List[str]
) -> Tuple[List[Dict[str, str]], float]:
    """Compare data types for columns shared between both DataFrames."""
    mismatches: List[Dict[str, str]] = []
    match_count = 0

    for column in common_columns:
        dtype_left = str(df_left[column].dtype)
        dtype_right = str(df_right[column].dtype)
        if dtype_left == dtype_right:
            match_count += 1
        else:
            mismatches.append(
                {
                    "column": column,
                    "left_dtype": dtype_left,
                    "right_dtype": dtype_right,
                }
            )

    total_columns = len(common_columns)
    match_ratio = match_count / total_columns if total_columns else 1.0
    return mismatches, match_ratio


def compute_cell_mismatches(
    df_left: pd.DataFrame, df_right: pd.DataFrame, common_columns: List[str]
) -> Tuple[pd.DataFrame, float]:
    """Identify cell-level mismatches for shared columns."""
    left_column_positions = {
        column: df_left.columns.get_loc(column) for column in common_columns
    }
    right_column_positions = {
        column: df_right.columns.get_loc(column) for column in common_columns
    }

    df_left_common = df_left[common_columns].copy()
    df_right_common = df_right[common_columns].copy()

    # Align row counts so that comparisons include extra rows.
    max_len = max(len(df_left_common), len(df_right_common))
    df_left_common = df_left_common.reindex(range(max_len))
    df_right_common = df_right_common.reindex(range(max_len))

    mismatch_records: List[Dict[str, object]] = []

    for row_index in range(max_len):
        for column in common_columns:
            left_value = df_left_common.at[row_index, column]
            right_value = df_right_common.at[row_index, column]

            left_is_na = pd.isna(left_value)
            right_is_na = pd.isna(right_value)

            if left_is_na and right_is_na:
                continue

            if left_value == right_value:
                continue

            if left_is_na and not right_is_na:
                mismatch_type = "missing_in_left"
            elif right_is_na and not left_is_na:
                mismatch_type = "missing_in_right"
            else:
                mismatch_type = "value_mismatch"

            mismatch_records.append(
                {
                    "row_index": row_index,
                    "column": column,
                    "left_value": left_value,
                    "right_value": right_value,
                    "mismatch_type": mismatch_type,
                    "left_cell": f"{column_index_to_excel_letter(left_column_positions[column])}{row_index + 2}",
                    "right_cell": f"{column_index_to_excel_letter(right_column_positions[column])}{row_index + 2}",
                }
            )

    mismatches_df = pd.DataFrame(mismatch_records)
    total_cells = max_len * len(common_columns)
    match_ratio = (
        1.0 - len(mismatch_records) / total_cells if total_cells else 1.0
    )

    return mismatches_df, match_ratio


def build_summary_section(
    column_diff: Dict[str, List[str]],
    dtype_mismatches: List[Dict[str, str]],
    cell_mismatches: pd.DataFrame,
) -> None:
    """Render summary details in the Streamlit UI."""
    st.subheader("Summary of Discrepancies")

    if column_diff["missing_in_left"] or column_diff["missing_in_right"]:
        st.write("**Column name discrepancies**")
        if column_diff["missing_in_left"]:
            st.write(
                "- Columns missing in first file: "
                + ", ".join(column_diff["missing_in_left"])
            )
        if column_diff["missing_in_right"]:
            st.write(
                "- Columns missing in second file: "
                + ", ".join(column_diff["missing_in_right"])
            )
    else:
        st.write("- Column names: no discrepancies detected")

    if dtype_mismatches:
        st.write("**Data type mismatches**")
        dtype_df = pd.DataFrame(dtype_mismatches)
        st.dataframe(dtype_df, use_container_width=True)
    else:
        st.write("- Data types: all matching for shared columns")

    if not cell_mismatches.empty:
        st.write("**Data differences**")
        st.dataframe(cell_mismatches, use_container_width=True)
    else:
        st.write("- Data values: no mismatches found")


def generate_download(mismatches: pd.DataFrame) -> BytesIO:
    """Create a downloadable CSV in memory."""
    buffer = BytesIO()
    mismatches.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer


def main() -> None:
    st.set_page_config(page_title="File Compare Utility", layout="wide")
    st.title("File Compare Utility")
    st.caption(
        "Upload two files to compare column names, data types, and cell-level differences."
    )

    with st.sidebar:
        st.markdown("### Instructions")
        st.markdown(
            "- Supported formats: CSV, XLS, XLSX\n"
            "- Comparisons use row order. Reorder rows first if they should align differently.\n"
            "- Download the mismatch report for a full breakdown of discrepancies."
        )

    col_left, col_right = st.columns(2)

    with col_left:
        file_left = st.file_uploader(
            "Upload first file",
            type=SUPPORTED_EXTENSIONS,
            key="file_left",
        )
    with col_right:
        file_right = st.file_uploader(
            "Upload second file",
            type=SUPPORTED_EXTENSIONS,
            key="file_right",
        )

    if not (file_left and file_right):
        st.info("Upload both files to run the comparison.")
        return

    try:
        df_left = load_tabular_file(file_left)
        df_right = load_tabular_file(file_right)
    except Exception as exc:
        st.error(f"Unable to read files: {exc}")
        return

    column_diff = compare_column_names(df_left, df_right)
    common_columns = column_diff["matched"]
    dtype_mismatches, dtype_match_ratio = compare_dtypes(
        df_left, df_right, common_columns
    )
    cell_mismatches, cell_match_ratio = compute_cell_mismatches(
        df_left, df_right, common_columns
    )

    total_unique_columns = (
        len(column_diff["matched"])
        + len(column_diff["missing_in_left"])
        + len(column_diff["missing_in_right"])
    )
    column_match_ratio = (
        len(column_diff["matched"]) / total_unique_columns
        if total_unique_columns
        else 1.0
    )

    st.markdown("### Match Overview")
    col_metrics = st.columns(3)
    col_metrics[0].metric(
        "Column Match",
        f"{column_match_ratio * 100:.1f}%",
        help="Percentage of column names that are present in both files.",
    )
    col_metrics[1].metric(
        "Data Type Match",
        f"{dtype_match_ratio * 100:.1f}%",
        help="Percentage of shared columns with matching data types.",
    )
    col_metrics[2].metric(
        "Data Value Match",
        f"{cell_match_ratio * 100:.1f}%",
        help="Percentage of shared cells with matching values.",
    )

    build_summary_section(column_diff, dtype_mismatches, cell_mismatches)

    if not cell_mismatches.empty:
        download_buffer = generate_download(cell_mismatches)
        st.download_button(
            label="Download mismatch report (CSV)",
            data=download_buffer,
            file_name="comparison_mismatches.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    main()
