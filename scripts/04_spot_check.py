"""Stage-0: print top 10 labeled JPM 2019 events for human spot-check.

Pick the 10 events with the largest |CAR_z| so we can verify the most consequential
labels match reality (e.g., earnings beats → up; weak quarter → down).
"""
from __future__ import annotations

import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

CACHE = ROOT / "cache"

ITEM_DESCRIPTIONS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition (EARNINGS)",
    "2.03": "Creation of a Direct Financial Obligation",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "3.01": "Notice of Delisting / Failure to Comply with Listing Standards",
    "3.03": "Material Modification to Rights of Security Holders",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "4.02": "Non-Reliance on Previous Financial Statements",
    "5.02": "Departure / Appointment of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "5.07": "Submission of Matters to a Vote of Security Holders",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def describe_items(items_str: str) -> str:
    if not items_str:
        return ""
    out = []
    for it in items_str.split(","):
        it = it.strip()
        out.append(f"  Item {it}: {ITEM_DESCRIPTIONS.get(it, '?')}")
    return "\n".join(out)


def main():
    df = pd.read_parquet(CACHE / "labeled_JPM_2019.parquet")
    df = df.sort_values("car_z_0_1", key=abs, ascending=False).head(10)

    for i, (_, row) in enumerate(df.iterrows(), 1):
        print("=" * 90)
        print(f"#{i}  filed {row['filingDate'].date()} (UTC {row['acceptanceDateTime']})  t0={row['t0'].date()}")
        print(describe_items(row["items"]))
        print(
            f"  CAR[0,+1]={row['car_0_1']*100:+.2f}%  z={row['car_z_0_1']:+.2f}  "
            f"direction={row['direction']}  material={row['material']}  "
            f"factor_R²={row['factor_r2']:.2f}"
        )
        print(f"  CAR[-1,0]={row['car_minus1_0']*100:+.2f}%  z={row['car_z_minus1_0']:+.2f}  (canary)")
        # Show a slice of the body where the substantive content begins (skip SEC cover).
        body = row["body_text"]
        start = max(0, body.lower().find("exhibit 99"))
        snippet = body[start : start + 600].replace("\n", " ")
        if snippet:
            print("  body excerpt:")
            for line in textwrap.wrap(snippet, width=86):
                print(f"    {line}")
        print()


if __name__ == "__main__":
    main()
