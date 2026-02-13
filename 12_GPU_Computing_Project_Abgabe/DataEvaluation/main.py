import pandas as pd
from pathlib import Path

# Input files
FILES = {
    "DFT in cpu": "../DFT_CPU/outdir/performance_dft_cpu.csv",
    "FFT in cpu": "../FFT_CPU/outdir/performance_fft_cpu.csv",
    "DFT in gpu": "../DFT_GPU/outdir/performance_dft_gpu.csv",
    "FFT in gpu": "../FFT_GPU/outdir/performance_fft_gpu.csv",
}

# Output files
OUT_SUMMARY_CSV = "./summary_total_time.csv"
OUT_COMPARISON_1_CSV = "./summary_comparison_1.csv"
OUT_COMPARISON_2_CSV = "./summary_comparison_2.csv"

TEST_ORDER = list("ABCDEF")


def load_and_sum_by_test(csv_path: str) -> pd.Series:
    """
    Sum ms for each test letter A..F.
    Assumes each row format like:
      test,N,stage,ms
    Example:
      A_gauss_lowpass_blur,2048,DFT2,51751.1
    """
    df = pd.read_csv(
        csv_path,
        header=None,
        names=["test", "N", "stage", "ms"],
    )

    # Extract leading test letter (A..F) from "A_...."
    df["test_letter"] = df["test"].astype(str).str.extract(r"^([A-F])_")[0]

    # Keep only valid test rows
    df = df[df["test_letter"].isin(TEST_ORDER)].copy()

    # Make sure ms is numeric
    df["ms"] = pd.to_numeric(df["ms"], errors="coerce").fillna(0.0)

    sums = df.groupby("test_letter")["ms"].sum()

    # Ensure A..F always exist (missing -> 0.0)
    return sums.reindex(TEST_ORDER, fill_value=0.0)


def write_summary_total_time():
    rows = {}
    for row_name, path in FILES.items():
        path = str(Path(path))
        rows[row_name] = load_and_sum_by_test(path)

    out = pd.DataFrame(rows).T
    out = out[TEST_ORDER]

    # keep 4 decimals
    out = out.round(4)

    # header: algorithm,A,B,C,D,E,F
    out.index.name = "algorithm"
    out.to_csv(OUT_SUMMARY_CSV, index=True, float_format="%.4f")

    print(f"Saved summary: {OUT_SUMMARY_CSV}")
    print(out)


def write_comparison_from_summary_1(out_csv: str = OUT_COMPARISON_1_CSV):
    """
    Rows:
    1) |DFTcpu - FFTcpu| / DFTcpu
    2) |DFTcpu - DFTgpu| / DFTcpu
    3) |FFTcpu - FFTgpu| / FFTcpu
    4) |DFTcpu - FFTgpu| / DFTcpu
    """
    df = pd.read_csv(OUT_SUMMARY_CSV).set_index("algorithm")
    df = df[TEST_ORDER].apply(pd.to_numeric, errors="coerce")

    dft_cpu = df.loc["DFT in cpu"]
    fft_cpu = df.loc["FFT in cpu"]
    dft_gpu = df.loc["DFT in gpu"]
    fft_gpu = df.loc["FFT in gpu"]

    comp = pd.DataFrame(
        {
            "abs(DFTcpu-FFTcpu)/DFTcpu": (dft_cpu.subtract(fft_cpu).abs() / dft_cpu),
            "abs(DFTcpu-DFTgpu)/DFTcpu": (dft_cpu.subtract(dft_gpu).abs() / dft_cpu),
            "abs(FFTcpu-FFTgpu)/FFTcpu": (fft_cpu.subtract(fft_gpu).abs() / fft_cpu),
            "abs(DFTcpu-FFTgpu)/DFTcpu": (dft_cpu.subtract(fft_gpu).abs() / dft_cpu),
        }
    ).T

    comp = comp[TEST_ORDER].round(4)
    comp.index.name = "algorithm comparision"
    comp.to_csv(out_csv, index=True, float_format="%.4f")

    print(f"Saved comparison 1: {out_csv}")
    print(comp)


def write_comparison_from_summary_2(out_csv: str = OUT_COMPARISON_2_CSV):
    """
    Rows:
    1) |DFTcpu - FFTcpu| / FFTcpu
    2) |DFTcpu - DFTgpu| / DFTgpu
    3) |FFTcpu - FFTgpu| / FFTgpu
    4) |DFTcpu - FFTgpu| / FFTgpu
    """
    df = pd.read_csv(OUT_SUMMARY_CSV).set_index("algorithm")
    df = df[TEST_ORDER].apply(pd.to_numeric, errors="coerce")

    dft_cpu = df.loc["DFT in cpu"]
    fft_cpu = df.loc["FFT in cpu"]
    dft_gpu = df.loc["DFT in gpu"]
    fft_gpu = df.loc["FFT in gpu"]

    comp = pd.DataFrame(
        {
            "abs(DFTcpu-FFTcpu)/FFTcpu": (dft_cpu.subtract(fft_cpu).abs() / fft_cpu),
            "abs(DFTcpu-DFTgpu)/DFTgpu": (dft_cpu.subtract(dft_gpu).abs() / dft_gpu),
            "abs(FFTcpu-FFTgpu)/FFTgpu": (fft_cpu.subtract(fft_gpu).abs() / fft_gpu),
            "abs(DFTcpu-FFTgpu)/FFTgpu": (dft_cpu.subtract(fft_gpu).abs() / fft_gpu),
        }
    ).T

    comp = comp[TEST_ORDER].round(4)
    comp.index.name = "algorithm comparision"
    comp.to_csv(out_csv, index=True, float_format="%.4f")

    print(f"Saved comparison 2: {out_csv}")
    print(comp)


def main():
    # 1) summary_total_time.csv
    write_summary_total_time()

    # 2) comparison csvs
    write_comparison_from_summary_1()
    write_comparison_from_summary_2()


if __name__ == "__main__":
    main()
