import pandas as pd
import numpy as np
from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log


class TradingStrategy(Strategy):

    def __init__(self):
        #self._assets = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC", "UUP", "BIL"]
        self._assets = ["SSO", "GPIX", "BCX", "ROBO", "BLOK", "DGT", "QTUM", "IDVO", "IDMO", "SETM", "NLR", "LQDW", "CSHI", "BIL"]
        #self.risk_assets = ["SPY", "QQQ", "TLT", "IEF", "GLD", "DBC", "UUP"]
        self.risk_assets = ["SSO", "GPIX", "BCX", "ROBO", "BLOK", "DGT", "QTUM", "IDVO", "IDMO", "SETM", "NLR", "LQDW", "CSHI"]
        self.safe_asset = "BIL"

        self.last_alloc = {a: 0.0 for a in self._assets}
        self.last_alloc[self.safe_asset] = 1.0

        self.prev_top_asset = None

    @property
    def assets(self):
        return self._assets

    @property
    def interval(self):
        return "1day"

    # -------------------------------------------------
    # INDICATORS
    # -------------------------------------------------

    def tsi(self, close, period=10):
        diff = close.diff()
        abs_diff = diff.abs()

        ema1 = diff.ewm(span=period).mean()
        ema2 = ema1.ewm(span=period).mean()

        abs1 = abs_diff.ewm(span=period).mean()
        abs2 = abs1.ewm(span=period).mean()

        return ema2 / abs2

    def ichimoku_base(self, high, low, period=26):
        return (high.rolling(period).max() +
                low.rolling(period).min()) / 2

    # -------------------------------------------------
    # MAIN LOGIC
    # -------------------------------------------------

    def run(self, data):

        ohlcv = data["ohlcv"]
        if len(ohlcv) < 80:
            return TargetAllocation(self.last_alloc)

        asset_data = {}

        # -----------------------------
        # STEP 1: Compute signals
        # -----------------------------
        for asset in self.risk_assets:

            df = pd.DataFrame({
                "close": [d[asset]["close"] for d in ohlcv],
                "high":  [d[asset]["high"] for d in ohlcv],
                "low":   [d[asset]["low"] for d in ohlcv],
            }, index=pd.to_datetime([d[asset]["date"] for d in ohlcv]))

            weekly = df.resample("W-FRI").last()
            monthly = df.resample("M").last()

            weekly_tsi = self.tsi(weekly["close"], 10).dropna()
            monthly_tsi = self.tsi(monthly["close"], 10).dropna()

            if len(weekly_tsi) < 10 or len(monthly_tsi) < 4:
                continue

            weekly_smooth = weekly_tsi.rolling(5).mean().dropna()

            if len(weekly_smooth) < 6:
                continue

            score = (
                0.75 * weekly_smooth.iloc[-1] +
                0.25 * monthly_tsi.iloc[-1]
            )

            roc = (weekly_smooth.iloc[-1] - weekly_smooth.iloc[-5])

            # Ichimoku base
            base_line = self.ichimoku_base(df["high"], df["low"])
            price = df["close"].iloc[-1]
            base = base_line.iloc[-1]

            if np.isnan(base):
                continue

            if price > base:
                cloud_mult = 1.0
            elif price > base * 0.97:
                cloud_mult = 0.9
            else:
                cloud_mult = 0.7

            asset_data[asset] = {
                "score": score,
                "roc": roc,
                "cloud_mult": cloud_mult,
                "prices": df["close"]
            }

        if len(asset_data) < 2:
            return TargetAllocation(self.last_alloc)

        assets = list(asset_data.keys())

        # -----------------------------
        # STEP 2: Relative normalization
        # -----------------------------
        scores = np.array([asset_data[a]["score"] for a in assets])
        rocs = np.array([asset_data[a]["roc"] for a in assets])

        score_ranks = pd.Series(scores).rank(pct=True).values
        roc_ranks = pd.Series(rocs).rank(pct=True).values

        for i, asset in enumerate(assets):
            strength = 0.7 * score_ranks[i] + 0.3 * roc_ranks[i]
            asset_data[asset]["strength"] = strength

        # -----------------------------
        # STEP 3: Preliminary ranking
        # -----------------------------
        ranked = sorted(
            asset_data.items(),
            key=lambda x: x[1]["strength"],
            reverse=True
        )

        provisional_top = ranked[0][0]

        # -----------------------------
        # STEP 4: Correlation penalty
        # -----------------------------
        try:
            returns_df = pd.DataFrame({
                asset: asset_data[asset]["prices"].pct_change()
                for asset in asset_data
            }).dropna().tail(60)

            corr_matrix = returns_df.corr()
        except:
            corr_matrix = None

        lambda_penalty = 0.4

        for asset in asset_data:

            if asset == provisional_top or corr_matrix is None:
                corr = 0.0
            else:
                try:
                    corr = corr_matrix.loc[asset, provisional_top]
                except:
                    corr = 0.0

            corr = max(0.0, corr)  # only penalize positive correlation

            penalty = 1 - lambda_penalty * corr

            asset_data[asset]["adj_strength"] = (
                asset_data[asset]["strength"] * penalty
            )

        # -----------------------------
        # STEP 5: Final ranking
        # -----------------------------
        ranked = sorted(
            asset_data.items(),
            key=lambda x: x[1]["adj_strength"],
            reverse=True
        )

        top_asset, top_data = ranked[0]
        second_strength = ranked[1][1]["adj_strength"]

        # -----------------------------
        # STEP 6: Conviction
        # -----------------------------
        spread = top_data["adj_strength"] - second_strength

        strengths = np.array([x[1]["adj_strength"] for x in ranked])
        mean = np.mean(strengths)
        std = np.std(strengths) if np.std(strengths) > 0 else 1.0

        z_score = (top_data["adj_strength"] - mean) / std

        conviction = 0.5 * spread + 0.5 * (z_score / 3.0)
        conviction = max(0.0, min(conviction, 1.0))

        # Persistence boost
        if self.prev_top_asset == top_asset:
            conviction = min(conviction * 1.1, 1.0)

        # -----------------------------
        # STEP 7: Allocation mapping
        # -----------------------------
        if conviction > 0.6:
            exposure = 1.0
        elif conviction > 0.4:
            exposure = 0.75
        elif conviction > 0.25:
            exposure = 0.5
        elif conviction > 0.1:
            exposure = 0.3
        else:
            exposure = 0.0

        final_exposure = exposure * top_data["cloud_mult"]

        # -----------------------------
        # STEP 8: Allocation
        # -----------------------------
        alloc = {a: 0.0 for a in self._assets}
        alloc[top_asset] = float(final_exposure)
        alloc[self.safe_asset] = float(1.0 - final_exposure)

        # Reduce micro-rebalancing
        prev_exposure = self.last_alloc.get(top_asset, 0.0)
        if abs(final_exposure - prev_exposure) < 0.05:
            return TargetAllocation(self.last_alloc)

        self.prev_top_asset = top_asset
        self.last_alloc = alloc

        log(f"Allocation: {alloc}")
        log(
            f"Top: {top_asset} | AdjStrength: {round(top_data['adj_strength'],3)} "
            f"| Conviction: {round(conviction,3)} | Exposure: {round(final_exposure,2)}"
        )

        return TargetAllocation(self.last_alloc)