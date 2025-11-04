import os

if "results_df" not in locals():
    raise ValueError("'results_df' is not defined. Have you executed the workload?")

output_path = "{{ output_path }}"
output_ext = os.path.splitext(output_path)[1].lower()

match output_ext:
    case ".csv":
        results_df.to_csv(output_path, index=False)
    case ".parquet":
        results_df.to_parquet(output_path, index=False)
    case ".feather":
        results_df.to_feather(output_path)
    case ".json":
        results_df.to_json(output_path)
    case ".xlsx" | ".xls":
        results_df.to_excel(output_path, index=False)
    case ".html":
        results_df.to_html(output_path, index=False)
    case ".tex":
        results_df.to_latex(output_path, index=False)
    case _:
        raise ValueError(f"Unsupported file extension: { output_ext }")

def format_bytes(num, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.2f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.2f}Yi{suffix}"

saved_results_size = os.path.getsize(output_path)
format_bytes(saved_results_size)