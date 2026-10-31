"""
Generate a synthetic dataset with the SAME schema as UCI Online Retail II.
Use it to test the pipeline end to end. Never report results from it as real.

    python src/make_synthetic_data.py
    python src/load_clean.py --input data/raw/synthetic_retail.csv
"""
import numpy as np
import pandas as pd

from config import RAW_DIR, SEED

START, END = pd.Timestamp("2009-12-01"), pd.Timestamp("2011-12-09")


def main(n_customers: int = 2500) -> None:
    rng = np.random.default_rng(SEED)
    n_products = 400
    product_codes = [str(10000 + i) for i in range(n_products)]
    product_price = np.round(rng.lognormal(1.0, 0.7, n_products), 2) + 0.2
    rows, invoice_no = [], 500000

    for k in range(n_customers):
        cid = 12000 + k
        country = "United Kingdom" if rng.random() < 0.9 else rng.choice(["Germany", "France", "EIRE", "Spain"])
        t = START + pd.Timedelta(days=float(rng.exponential(220)))
        rate = rng.gamma(2.0, 0.012)                    # orders per day
        lifetime = rng.exponential(260)                 # days until the customer silently churns
        stop = min(t + pd.Timedelta(days=float(lifetime)), END)
        while t < stop:
            invoice_no += 1
            for _ in range(int(rng.poisson(7)) + 1):
                p = int(rng.integers(0, n_products))
                rows.append((str(invoice_no), product_codes[p], f"ITEM {p}", int(rng.lognormal(1.6, 0.7)),
                             t, float(product_price[p]), float(cid), country))
            if rng.random() < 0.04:                     # cancellation shortly after
                invoice_no += 1
                p = int(rng.integers(0, n_products))
                rows.append((f"C{invoice_no}", product_codes[p], f"ITEM {p}", -int(rng.lognormal(1.2, 0.6)),
                             t + pd.Timedelta(days=2), float(product_price[p]), float(cid), country))
            t = t + pd.Timedelta(days=float(rng.exponential(1.0 / max(rate, 0.004))))

    df = pd.DataFrame(rows, columns=["Invoice", "StockCode", "Description", "Quantity",
                                     "InvoiceDate", "Price", "Customer ID", "Country"])
    df = df[df["InvoiceDate"] <= END]

    # Add the kinds of mess the real data has
    junk = df.sample(frac=0.02, random_state=SEED).copy()
    junk["StockCode"] = "POST"
    guest = df.sample(frac=0.15, random_state=SEED + 1).copy()
    guest["Customer ID"] = np.nan
    guest["Invoice"] = guest["Invoice"].map(lambda s: "9" + s[-5:])
    df = pd.concat([df, junk, guest, df.sample(500, random_state=SEED)], ignore_index=True)  # incl. duplicates

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / "synthetic_retail.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df):,} rows -> {out}")


if __name__ == "__main__":
    main()
