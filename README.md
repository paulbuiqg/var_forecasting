# Asset price volatility risk quantification

## Data

The data is composed of 4 timeseries of daily prices for:

- Bitcoin (BTC)
- Ether (ETH)
- Status Network Token (SNT)
- Euro (EUR)

from 2021-06-05 to 2024-06-02.

<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/BTC.png" />
</p>
<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/ETH.png" />
</p>
<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/SNT.png" />
</p>
<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/EUR.png" />
</p>

## Probleme Statement

We aim at computing the one-day-ahead value-at-risk (VaR) for the log-returns of the 4 assets.

The VaR at level $\alpha$ is defined as the value such that:

$$ P[X < \text{VaR}_\alpha] < 1 - \alpha $$

where $X$ is the log-return. We set $\alpha = 0.95$.

The training period is from 2021-06-05 to 2023-10-27 (80% of the data). The test period is from 2023-10-28 to 2024-06-02.

## Methods

We calculate the VaR with two methods:

- [historical VaR](https://www.financestrategists.com/wealth-management/fundamental-vs-technical-analysis/historical-var/)
- [VARMAX model](https://www.statsmodels.org/stable/examples/notebooks/generated/statespace_varmax.html).

The VARMAX model in the `statsmodels` library allows to compute multivariate and distributional timeseries forecasts. In particular, confidence intervals - hence VaRs - can be extracted from marginal (univariate) distributional forecasts.

The training period is for initial inference. During the test period, one-day-ahead VaR forecasts are computed every day; each new observation is taken into account sequentially to recompute the historical VaR or recalibrate the VARMAX model.

## Results

The following table gathers the percentage of VaR violations in the test period.

|     | Historical | Model |
|-----|:----------:|:-----:|
| BTC | 2.7%       | 7.3%  |
| ETH | 2.3%       | 8.2%  |
| SNT | 5.9%       | 5.0%  |
| EUR | 1.4%       | 1.8%  |

<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/BTC_VaR.png" />
</p>

<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/ETH_VaR.png" />
</p>

<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/SNT_VaR.png" />
</p>

<p align="center">
  <img src="https://github.com/paulbuiqg/var_forecasting/blob/main/viz/EUR_VaR.png" />
</p>

As a VaR forecast performance metric, in the below table, we compute the absolute difference between the empirical VaR violation rate (in percentage) and the theoretical value of 5%. The metric is expressed in percentage points (pp).

|         | Historical | Model |
|---------|:----------:|:-----:|
| BTC     | 2.3 pp     | 2.3 pp|
| ETH     | 2.7 pp     | 3.2 pp|
| SNT     | 0.9 pp     | 0.0 pp|
| EUR     | 3.6 pp     | 3.2 pp|
| average | 2.5 pp     | 2.2 pp|

The best method is the VARMAX model (average error: 2.2 pp). The asset which yields the best performance is SNT (error: 0.9 pp, 0.0 pp).

## Discussion

VaR forecasting performance is heterogeneous over the 4 assets and the 2 methods. The VARMAX model method outperforms the historical VaR method. The average error of both methods is quite large compared to the 95% VaR level. However, results are very good for SNT, especially with the VARMAX model.
