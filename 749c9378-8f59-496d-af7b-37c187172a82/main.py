import pandas as pd
import numpy as np
from surmount.base_class import Strategy, TargetAllocation
from surmount.logging import log
class TradingStrategy(Strategy):
    def __init__(self):
        self._assets = [
            "SSO", "GPIX", "BCX", "ROBO", "BLOK",
            "DGT", "QTUM", "IDVO", "IDMO",
            "SETM", "NLR", "LQDW", "CSHI", "BIL"
        ]
        self.risk_assets = [
            "SSO", "GPIX", "BCX", "ROBO", "BLOK",
            "DGT", "QTUM", "IDVO", "IDMO",
            "SETM", "NLR", "LQDW", "CSHI"
        ]
        self.safe_asset = "BIL"
        self.last_alloc = {a: 0.0 for a in self._assets}
        self.last_alloc[self.safe_asset] = 1.0
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
        return (
            high.rolling(period).max() +
            low.rolling(period).min()
        ) / 2
    # -------------------------------------------------
    # MAIN LOGIC
    # -------------------------------------------------
    def run(self, data):
        ohlcv = data["ohlcv"]
        if len(ohlcv) < 80:
            return TargetAllocation(self.last_alloc)
        asset_data = {}
        # -------------------------------------------------
        # STEP 1: SIGNAL GENERATION
        # -------------------------------------------------
        for asset in self.risk_assets:
            try:
                df = pd.DataFrame({
                    "close": [d[asset]["close"] for d in ohlcv],
                    "high":  [d[asset]["high"] for d in ohlcv],
                    "low":   [d[asset]["low"] for d in ohlcv],
                }, index=pd.to_datetime([
                    d[asset]["date"] for d in ohlcv
                ]))
                weekly = df.resample("W-FRI").last()
                monthly = df.resample("M").last()
                weekly_tsi = self.tsi(
                    weekly["close"], 10
                ).dropna()
                monthly_tsi = self.tsi(
                    monthly["close"], 10
                ).dropna()
                if len(weekly_tsi) < 10 or len(monthly_tsi) < 4:
                    continue
                weekly_smooth = (
                    weekly_tsi
                    .rolling(5)
                    .mean()
                    .dropna()
                )
                if len(weekly_smooth) < 6:
                    continue
                # -----------------------------------------
                # Multi-timeframe blended score
                # -----------------------------------------
                score = (
                    0.75 * weekly_smooth.iloc[-1] +
                    0.25 * monthly_tsi.iloc[-1]
                )
                roc = (
                    weekly_smooth.iloc[-1] -
                    weekly_smooth.iloc[-5]
                )
                # -----------------------------------------
                # Ichimoku confidence modifier
                # -----------------------------------------
                base_line = self.ichimoku_base(
                    df["high"],
                    df["low"]
                )
                base = base_line.iloc[-1]
                price = df["close"].iloc[-1]
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
            except Exception as e:
                log(f"{asset} processing error: {e}")
        if len(asset_data) < 3:
            return TargetAllocation(self.last_alloc)
        assets = list(asset_data.keys())
        # -------------------------------------------------
        # STEP 2: RELATIVE NORMALIZATION
        # -------------------------------------------------
        scores = np.array([
            asset_data[a]["score"]
            for a in assets
        ])
        rocs = np.array([
            asset_data[a]["roc"]
            for a in assets
        ])
        score_ranks = (
            pd.Series(scores)
            .rank(pct=True)
            .values
        )
        roc_ranks = (
            pd.Series(rocs)
            .rank(pct=True)
            .values
        )
        # -------------------------------------------------
        # STEP 3: COMPOSITE STRENGTH
        # -------------------------------------------------
        for i, asset in enumerate(assets):
            strength = (
                0.7 * score_ranks[i] +
                0.3 * roc_ranks[i]
            )
            asset_data[asset]["strength"] = strength
        # -------------------------------------------------
        # STEP 4: PRELIMINARY RANKING
        # -------------------------------------------------
        prelim_ranked = sorted(
            asset_data.items(),
            key=lambda x: x[1]["strength"],
            reverse=True
        )
        provisional_top = prelim_ranked[0][0]
        # -------------------------------------------------
        # STEP 5: CORRELATION PENALTY
        # -------------------------------------------------
        try:
            returns_df = pd.DataFrame({
                asset: asset_data[asset]["prices"].pct_change()
                for asset in asset_data
            }).dropna().tail(60)
            corr_matrix = returns_df.corr()
        except Exception:
            corr_matrix = None
        lambda_penalty = 0.4
        for asset in asset_data:
            if asset == provisional_top or corr_matrix is None:
                corr = 0.0
            else:
                try:
                    corr = corr_matrix.loc[
                        asset,
                        provisional_top
                    ]
                except Exception:
                    corr = 0.0
            corr = max(0.0, corr)
            penalty = 1 - lambda_penalty * corr
            asset_data[asset]["adj_strength"] = (
                asset_data[asset]["strength"] * penalty
            )
        # -------------------------------------------------
        # STEP 6: FINAL RANKING
        # -------------------------------------------------
        ranked = sorted(
            asset_data.items(),
            key=lambda x: x[1]["adj_strength"],
            reverse=True
        )
        top_3 = ranked[:3]
        # -------------------------------------------------
        # STEP 7: FIXED WEIGHT MODEL
        # -------------------------------------------------
        target_weights = [0.50, 0.30, 0.20]
        alloc = {
            a: 0.0 for a in self._assets
        }
        total_alloc = 0.0
        for i, (asset, data_dict) in enumerate(top_3):
            base_weight = target_weights[i]
            # ---------------------------------------------
            # Apply Ichimoku confidence multiplier
            # ---------------------------------------------
            final_weight = (
                base_weight *
                data_dict["cloud_mult"]
            )
            final_weight = round(final_weight, 4)
            alloc[asset] = final_weight
            total_alloc += final_weight
        # -------------------------------------------------
        # STEP 8: SAFE ASSET BUFFER
        # -------------------------------------------------
        remaining = max(0.0, 1.0 - total_alloc)
        alloc[self.safe_asset] = remaining
        # -------------------------------------------------
        # STEP 9: MICRO-REBALANCE SUPPRESSION
        # -------------------------------------------------
        total_change = 0.0
        for asset in self._assets:
            prev = self.last_alloc.get(asset, 0.0)
            curr = alloc.get(asset, 0.0)
            total_change += abs(curr - prev)
        if total_change < 0.10:
            return TargetAllocation(self.last_alloc)
        # -------------------------------------------------
        # STEP 10: FINALIZE
        # -------------------------------------------------
        self.last_alloc = alloc
        log(f"Allocation: {alloc}")
        log(
            f"Top3: "
            f"{top_3[0][0]} / "
            f"{top_3[1][0]} / "
            f"{top_3[2][0]}"
        )
        return TargetAllocation(self.last_alloc)
