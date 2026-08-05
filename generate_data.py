"""
Synthetic marketing dataset generator
======================================

Builds a small, realistic, fully reproducible marketing dataset for the
analysis in this repo. Everything downstream (channel performance, CAC/ROAS,
attribution, LTV, anomaly detection) reads the three CSVs written here, so the
numbers are internally consistent: users are generated *from* the paid clicks,
which makes cost-per-acquisition and ROAS meaningful rather than arbitrary.

Three tables are produced in ``data/``:

* ``spend.csv``        - daily media spend per channel / country / campaign,
                         with impressions and clicks.
* ``users.csv``        - one row per acquired user: signup date, country,
                         acquisition channel, whether they converted, and their
                         commission revenue / lifetime value.
* ``touchpoints.csv``  - the ordered marketing touches that led to each user,
                         for multi-touch attribution.

A deliberate anomaly is baked in (a Paid Social spend spike in Brazil that does
*not* lift conversions) so the anomaly-detection and wasted-spend analysis has
something real to find.

Run:  python generate_data.py

Author: Muhammad Nasiruddin
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

SEED = 42
DATA_DIR = Path(__file__).parent / "data"

# One full year of daily data.
START_DATE = pd.Timestamp("2025-07-01")
END_DATE = pd.Timestamp("2026-06-30")

# Acquisition channels and their economics. These are the *base* per-day,
# per-country figures; seasonality, country weighting and noise are layered on
# top. ``cpm`` is cost per 1,000 impressions, ``ctr`` the click-through rate,
# ``cvr`` the click-to-signup rate, ``base_spend`` the typical daily spend in a
# mid-size market. ``rev`` and ``months`` drive lifetime value downstream.
CHANNELS: dict[str, dict[str, float]] = {
    # channel        base_spend  cpm    ctr     cvr     rev   months
    # cpm is set so the implied CPC (cpm / (ctr * 1000)) is realistic per channel
    # e.g. Paid Search ~$1.60, Paid Social ~$0.70, Display ~$0.60.
    # cvr here is the click -> signup rate (~0.5-2%); combined with CPC it sets a
    # realistic customer-acquisition cost. rev/months drive lifetime value.
    "Paid Search": dict(base_spend=900, cpm=72.0, ctr=0.045, cvr=0.0190, rev=90, months=9),
    "Paid Social": dict(base_spend=750, cpm=13.0, ctr=0.018, cvr=0.0080, rev=70, months=6),
    "Display": dict(base_spend=300, cpm=5.0, ctr=0.008, cvr=0.0025, rev=55, months=5),
    "Email": dict(base_spend=90, cpm=9.0, ctr=0.090, cvr=0.0060, rev=110, months=12),
    "SEO": dict(base_spend=40, cpm=7.0, ctr=0.070, cvr=0.0120, rev=120, months=14),
    "Referral": dict(base_spend=160, cpm=45.0, ctr=0.050, cvr=0.0170, rev=100, months=11),
}

# Markets (multi-country). The weight scales spend and volume; ``cvr_mult`` and
# ``rev_mult`` let conversion quality and monetisation differ by market.
COUNTRIES: dict[str, dict[str, float]] = {
    "UK": dict(weight=1.00, cvr_mult=1.10, rev_mult=1.20),
    "US": dict(weight=1.35, cvr_mult=1.05, rev_mult=1.35),
    "DE": dict(weight=0.75, cvr_mult=1.00, rev_mult=1.05),
    "BR": dict(weight=0.60, cvr_mult=0.80, rev_mult=0.55),
    "IN": dict(weight=0.55, cvr_mult=0.70, rev_mult=0.45),
}

# A couple of named campaigns per channel, just for realistic granularity.
CAMPAIGNS: dict[str, list[str]] = {
    "Paid Search": ["brand", "nonbrand"],
    "Paid Social": ["prospecting", "retargeting"],
    "Display": ["prospecting", "retargeting"],
    "Email": ["newsletter", "lifecycle"],
    "SEO": ["organic"],
    "Referral": ["affiliates"],
}

# The baked-in anomaly: for this window, Paid Social spend in Brazil is inflated
# ~3.4x with no matching lift in clicks/conversions (a misconfigured campaign
# burning budget). Day 5's anomaly detection should catch this.
ANOMALY = dict(
    channel="Paid Social",
    country="BR",
    start=pd.Timestamp("2026-02-02"),
    end=pd.Timestamp("2026-02-22"),
    spend_multiplier=3.4,
)


# --------------------------------------------------------------------------- #
# Spend
# --------------------------------------------------------------------------- #

def _seasonality(dates: pd.DatetimeIndex) -> np.ndarray:
    """A gentle demand curve: a summer dip, a Q4 peak and a January lull.

    Returned as a multiplier centred near 1.0 so annual totals stay realistic.
    """
    day_of_year = dates.dayofyear.to_numpy()
    # Broad seasonal wave (peak late in the year) plus a weekly weekend lift.
    seasonal = 1.0 + 0.18 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    q4_boost = np.where(np.isin(dates.month.to_numpy(), [11, 12]), 1.15, 1.0)
    jan_lull = np.where(dates.month.to_numpy() == 1, 0.90, 1.0)
    weekend = np.where(dates.dayofweek.to_numpy() >= 5, 1.08, 1.0)
    return seasonal * q4_boost * jan_lull * weekend


def generate_spend(rng: np.random.Generator) -> pd.DataFrame:
    """Daily spend / impressions / clicks per channel, country and campaign."""
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    season = _seasonality(dates)

    rows: list[dict] = []
    for channel, ch in CHANNELS.items():
        for campaign in CAMPAIGNS[channel]:
            # Split the channel's spend across its campaigns.
            campaign_share = 1.0 / len(CAMPAIGNS[channel])
            for country, co in COUNTRIES.items():
                base = ch["base_spend"] * campaign_share * co["weight"]
                # A slow linear growth trend over the year (marketing scaling up).
                trend = np.linspace(0.85, 1.20, len(dates))
                noise = rng.normal(1.0, 0.12, len(dates)).clip(0.5, 1.6)
                spend = base * season * trend * noise

                # Apply the Brazil Paid Social anomaly.
                if channel == ANOMALY["channel"] and country == ANOMALY["country"]:
                    mask = (dates >= ANOMALY["start"]) & (dates <= ANOMALY["end"])
                    spend = spend * np.where(mask, ANOMALY["spend_multiplier"], 1.0)

                impressions = spend / ch["cpm"] * 1000.0
                clicks = impressions * ch["ctr"] * rng.normal(1.0, 0.06, len(dates)).clip(0.7, 1.3)

                for d, sp, im, cl in zip(dates, spend, impressions, clicks):
                    rows.append(
                        dict(
                            date=d.date().isoformat(),
                            channel=channel,
                            country=country,
                            campaign=campaign,
                            spend=round(float(sp), 2),
                            impressions=int(round(im)),
                            clicks=int(round(cl)),
                        )
                    )

    df = pd.DataFrame(rows)
    return df.sort_values(["date", "channel", "country", "campaign"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)

    spend = generate_spend(rng)
    spend.to_csv(DATA_DIR / "spend.csv", index=False)
    print(f"spend.csv        {len(spend):>6,} rows  "
          f"(${spend['spend'].sum():,.0f} total spend, "
          f"{spend['clicks'].sum():,.0f} clicks)")


if __name__ == "__main__":
    main()
