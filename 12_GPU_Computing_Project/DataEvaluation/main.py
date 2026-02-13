import pandas as pd
from pathlib import Path

# Input files
FILES = {
    "DFT in cpu": "../DFT_CPU/outdir/performance_dft_cpu.csv",
    "FFT in cpu": "../FFT_CPU/outdir/performance_fft_cpu.csv",
    "DFT in gpu": "../DFT_GPU/outdir/performance_dft_gpu.csv",
    "FFT in gpu": "../FFT_GPU/outdir/performance_fft_gpu.csv",
}

# Output file
OUT_CSV = "./summary_total_time.csv"

TEST_ORDER = list("ABCDEF")


def load_and_sum_by_test(csv_path: str) -> pd.Series:
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["test", "N", "stage", "ms"],
    )

    # Extract leading test letter
    df["test_letter"] = df["test"].astype(str).str.extract(r"^([A-F])_")[0]

    df = df[df["test_letter"].isin(TEST_ORDER)].copy()

    df["ms"] = pd.to_numeric(df["ms"], errors="coerce").fillna(0.0)

    sums = df.groupby("test_letter")["ms"].sum()

    return sums.reindex(TEST_ORDER, fill_value=0.0)


def main():
    rows = {}
    for row_name, path in FILES.items():
        path = str(Path(path))
        rows[row_name] = load_and_sum_by_test(path)

    out = pd.DataFrame(rows).T
    out = out[TEST_ORDER]

    # keep 4 decimal places
    out = out.round(4)

    out.to_csv(OUT_CSV, index=True, float_format="%.4f")

    print(f"Saved: {OUT_CSV}")
    print(out)


if __name__ == "__main__":
    main()
