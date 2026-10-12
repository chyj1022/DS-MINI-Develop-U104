"""Generate the README in the supplied order and a result-filled notebook."""
import numpy as np
import pandas as pd
from .preprocess import ROOT

OUT = ROOT / "DAY2/results"


def markdown_table(frame):
    columns = list(frame.columns)
    def fmt(value):
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}" if np.isfinite(value) else "—"
        return str(value).replace("|", "/").replace("\n", " ")
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join("| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False,name=None))


def batch3_reporting_format(performance):
    """Match the assignment's four-column additional-batch layout."""
    rows=[]
    for _,row in performance.iterrows():
        label=row['구분']; is_gap=label.startswith('Gap (')
        rows.append({'구분': '' if is_gap else label,
                     '비교 항목': label if is_gap else '',
                     'MAPE (%)': row['MAPE (%)'], '비고': row['비고']})
    return pd.DataFrame(rows)


def build_report(*args, **kwargs):
    from .plan_report import build
    build()


def main():
    build_report()


if __name__ == '__main__':
    main()
