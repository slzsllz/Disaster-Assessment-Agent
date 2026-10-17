import argparse

from pathlib import Path
from fastmcp import FastMCP


mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool(description='''
Description

Computes the linear trend (slope and intercept) of a time series by fitting a line of the form:

y = a \\cdot x + b

using the least squares method.

Parameters
    • y (list):
The dependent variable — typically your time series data.
    • x (list):
The independent variable — usually time indices. If not provided, the function will use np.arange(len(y)) as a default.

Returns
    • slope (float):
The coefficient a — represents the trend.
    • > 0: upward trend
    • < 0: downward trend
    • ≈ 0: no trend
    • intercept (float):
The y-intercept b of the fitted line.
''')
def compute_linear_trend(y: list, x: list|None = None):
    '''
    Description

    Computes the linear trend (slope and intercept) of a time series by fitting a line of the form:

    y = a \\cdot x + b

    using the least squares method.

    Parameters
        • y (list):
    The dependent variable — typically your time series data.
        • x (list):
    The independent variable — usually time indices. If not provided, the function will use np.arange(len(y)) as a default.

    Returns
        • slope (float):
    The coefficient a — represents the trend.
        • > 0: upward trend
        • < 0: downward trend
        • ≈ 0: no trend
        • intercept (float):
    The y-intercept b of the fitted line.
    '''
    import numpy as np
    y = np.asarray(y)
    if x is None:
        x = np.arange(len(y))
    else:
        x = np.asarray(x)

    if len(x) != len(y):
        raise ValueError("len(x) != len(y)")

    A = np.vstack([x, np.ones_like(x)]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]

    return float(a), float(b)
    # return list(a), list(b)


@mcp.tool(description='''
Description:
Perform the non-parametric Mann-Kendall trend test on a univariate time series.
The test evaluates whether there is a monotonic upward or downward trend
without requiring the data to conform to any particular distribution.

Parameters:
- x (list[float]): Input time series values (numeric). Missing values should be removed before calling.

Returns:
- trend (str): Type of detected trend. One of:
    * "increasing" — statistically significant upward trend
    * "decreasing" — statistically significant downward trend
    * "no trend" — no statistically significant trend
- p_value (float): Two-tailed p-value of the test.
- z (float): Standard normal test statistic.
- tau (float): Kendall’s Tau statistic (rank correlation, between -1 and 1).
''')
def mann_kendall_test(x: list):
    """
    Description:
        Conduct the Mann-Kendall trend test on a time series to assess
        whether a statistically significant monotonic trend exists.
        The test is non-parametric and does not assume normality.
        Handles tied ranks with variance correction.

    Parameters:
        x (list[float]):
            The input time series data as a list of floats or ints.
            Any missing values (NaN) should be removed beforehand.

    Returns:
        trend (str):
            Type of detected trend:
              - "increasing" if a significant upward trend is found
              - "decreasing" if a significant downward trend is found
              - "no trend" if no significant trend is detected
        p_value (float):
            Two-tailed p-value of the test.
        z (float):
            Standard normal test statistic.
        tau (float):
            Kendall’s Tau statistic (measure of rank correlation, range -1 to 1).

    Example:
        >>> mann_kendall_test([1, 2, 3, 4, 5])
        ('increasing', 0.03, 2.12, 0.9)
    """
    import numpy as np
    from scipy.stats import norm
    x = np.asarray(x)
    n = len(x)

    s = 0
    for k in range(n - 1):
        s += np.sum(np.sign(x[k+1:] - x[k]))

    # Calculate the unique data ranks for variance correction
    unique_x, counts = np.unique(x, return_counts=True)
    g = len(counts)

    # Variance of S
    if n == len(np.unique(x)):
        var_s = (n*(n-1)*(2*n+5)) / 18
    else:
        var_s = (n*(n-1)*(2*n+5) - np.sum(counts*(counts-1)*(2*counts+5))) / 18

    # Compute Z statistic
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0

    # Two-tailed p-value
    p = 2 * (1 - norm.cdf(abs(z)))

    # Kendall's Tau
    tau = s / (0.5 * n * (n - 1))

    # Determine trend
    alpha = 0.05  # significance level
    if p < alpha:
        trend = "increasing" if z > 0 else "decreasing"
    else:
        trend = "no trend"

    return trend, float(p), float(z), float(tau)


@mcp.tool(description='''
Description:
Compute Sen’s Slope estimator for a univariate time series.
Sen’s Slope is a robust non-parametric method for estimating
the median rate of change over time, often used with the
Mann-Kendall test to assess both trend and magnitude.

Parameters:
- x (list[float]): Input time series values (numeric).
                   Must contain at least two observations.

Returns:
- slope (float): Sen’s Slope estimate (the median of all pairwise slopes).
- slopes (list[float]): List of all individual pairwise slopes,
                        useful for optional further analysis.
''')
def sens_slope(x: list):
    """
    Description:
        Compute Sen’s Slope estimator for a univariate time series.
        This robust non-parametric method calculates the median of
        all pairwise slopes between observations, providing an estimate
        of the overall monotonic trend magnitude.

    Parameters:
        x (list[float]):
            The input time series data as a list of floats or ints.
            Must have at least two data points.

    Returns:
        slope (float):
            The Sen’s Slope estimate (median of all pairwise slopes).
        slopes (list[float]):
            List of all pairwise slopes, which can be used for
            further distributional or variability analysis.

    Example:
        >>> sens_slope([2, 4, 6, 8, 10])
        (2.0, [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    """
    import numpy as np
    x = np.asarray(x)
    n = len(x)

    if n < 2:
        raise ValueError("At least two data points are required.")

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            slope_ij = (x[j] - x[i]) / (j - i)
            slopes.append(slope_ij)

    slopes = np.array(slopes)
    median_slope = np.median(slopes)

    return float(median_slope), [float(s) for s in slopes]


@mcp.tool(description='''
Description:
Detect structural change points in a univariate time series using the
ruptures library with the PELT algorithm. A change point marks a location
where the statistical properties of the signal shift (e.g., mean or variance).

Parameters:
- signal (list[float]): Input 1D time series data.
- model (str, optional): Cost model type for segmentation.
                         Options include "l1", "l2" (default; mean shift), "rbf", etc.
- penalty (float, optional): Penalty value that controls sensitivity.
                             Higher penalty = fewer detected change points. Default = 10.

Returns:
- change_points (list[int]): Indices where change points are detected.
                             Includes the last index of the signal by default.
''')
def detect_change_points(signal, model="l2", penalty=10):
    """
    Description:
        Detect change points in a one-dimensional time series using the
        PELT algorithm from the ruptures library. This identifies indices
        where the statistical structure of the signal changes.

    Parameters:
        signal (list[float]):
            Input time series data.
        model (str, default="l2"):
            Segmentation cost model to use.
            - "l1": robust to outliers (absolute loss)
            - "l2": mean shift model (squared loss, default)
            - "rbf": kernel-based model for nonlinear changes
            - others supported by ruptures
        penalty (float, default=10):
            Penalty value controlling sensitivity.
            Higher values detect fewer change points (more conservative).

    Returns:
        change_points (list[int]):
            List of indices marking change points in the series.
            The final index of the signal is always included.

    Example:
        >>> detect_change_points([1, 1, 1, 10, 10, 10, 2, 2, 2], model="l2", penalty=5)
        [3, 6, 9]
    """
    import numpy as np
    import ruptures as rpt

    signal = np.asarray(signal)
    algo = rpt.Pelt(model=model).fit(signal)
    change_points = algo.predict(pen=penalty)
    return [int(cp) for cp in change_points]


if __name__ == "__main__":
    mcp.run()
