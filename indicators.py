"""This python module computes the simple and exponential moving averages of a dataset.
Support and resistance lines are also calculated. These are then graphically displayed
using matplotlib.

More details about computations can be found in the final report.

By Freeman Cheng and Idris Tarwala
"""
from typing import List, Tuple
import math
import matplotlib.pyplot as plt
from filter import Dataset
from constants import MONTHS
import regressions

plt.ioff()


def plot_simple_moving_average(dataset: Dataset, show_scatter: bool = False, show_sr: bool = False) \
        -> plt.Figure:
    """Given a dataset, plot its simple moving average. If show_scatter, also plot a scatter
    plot of dataset. If show_sr, also plot the support and resistance lines.

    Return the corresponding figure.

    Preconditions:
        - len(dataset.data) >= 20
    """
    # we are fixing the period to always be 20
    # we can do this because we know beforehand that our dataset contains more than 20 rows
    period = 20

    x_values = regressions.sort_x_values(dataset)
    y_values = regressions.sort_y_values(dataset)

    sma = calculate_simple_moving_average(period, y_values)

    figure = plt.figure()
    plt.xlabel('Year')
    plt.title('Simple Moving Average')
    plt.ylabel('Temperature (°C)')

    start_year = int(x_values[0])
    end_year = int(x_values[-1])

    plt.plot(x_values, sma, label='Simple Moving Average')
    if show_scatter:
        plt.scatter(x_values, y_values, label=MONTHS[dataset.month] + ' Temperature Data', color='k', s=10)
    if show_sr:
        points = regressions.format_data_points(x_values, sma)
        # calculate support; we're using the same period of 20 here
        # here, we're starting from the index period
        # this is because moving average doesn't stabilize until the period
        a, b = determine_support(period, points[period:])
        plt.plot(list(range(start_year + period, end_year + 1)),
                 [a + b * year for year in range(start_year + period, end_year + 1)],
                 label='Support Line')

        # calculate resistance similarly
        a, b = determine_resistance(period, points[period:])
        plt.plot(list(range(start_year + period, end_year + 1)),
                 [a + b * year for year in range(start_year + period, end_year + 1)],
                 label='Resistance Line')

    plt.legend(loc='upper left')
    # plt.show()
    return figure


def plot_ema(dataset: Dataset, show_scatter: bool = False, show_sr: bool = False) \
        -> plt.Figure:
    """Given a dataset, plot its exponential moving average. If show_scatter, also plot a scatter
    plot of dataset. If show_sr, also plot the support and resistance lines.

    Return the corresponding figure.

    Preconditions:
        - len(dataset.data) >= 20 * 2
    """
    # we are fixing the period to always be 20
    # we can do this because we know beforehand that our dataset contains more than 20 rows
    period = 20

    x_values = regressions.sort_x_values(dataset)
    y_values = regressions.sort_y_values(dataset)

    ema = calculate_ema(period, y_values)

    figure = plt.figure()
    plt.xlabel('Year')
    plt.title('Exponential Moving Average')
    plt.ylabel('Temperature (°C)')

    start_year = int(x_values[-1]) - len(ema) + 1
    end_year = int(x_values[-1])

    plt.plot(list(range(start_year, end_year + 1)), ema, label='Exponential Moving Average')
    if show_scatter:
        plt.scatter(x_values, y_values, label=MONTHS[dataset.month] + ' Temperature Data', color='k', s=10)
    if show_sr:
        points = regressions.format_data_points(list(range(start_year, end_year + 1)), ema)
        # calculate support; we're using the same period of 20 here
        # here, we're starting from the index 0 (different from simple moving average)
        # because the exponential moving average already starts stable
        a, b = determine_support(period, points)
        plt.plot(list(range(start_year + period, end_year + 1)),
                 [a + b * year for year in range(start_year + period, end_year + 1)],
                 label='Support Line')

        # calculate resistance similarly
        a, b = determine_resistance(period, points)
        plt.plot(list(range(start_year, end_year + 1)),
                 [a + b * year for year in range(start_year, end_year + 1)],
                 label='Resistance Line')

    plt.legend(loc='upper left')
    # plt.show()
    return figure


###############################################################################
# Helper Functions
###############################################################################
def calculate_simple_moving_average(period: int, values: List[float]) -> List[float]:
    """Given a period and an array of float values

    return an array of floats having value's the simple moving average

    Preconditions:
        - period > 0
    """
    rolling_length = 0
    moving_average = []
    for i in range(0, len(values)):
        if rolling_length < period:
            rolling_length += 1
        rolling_sum = sum(values[i + 1 - rolling_length:i + 1])
        moving_average.append(rolling_sum / rolling_length)
    return moving_average


def calculate_ema(period: int, values: List[float], smoothing: float = 2) -> List[float]:
    """Given a period and an array of float values

    return an array of floats having the value's exponential moving average

    NOTE: This requires len(values) > period unlike simple moving average!

    Preconditions:
        - period > 0
        - len(values) > period
    """
    weight = smoothing / (period + 1)
    simple_moving_average = calculate_simple_moving_average(period, values)
    exponential_moving_average = [simple_moving_average[period - 1]]  # This initial value will be removed at end
    for i in range(period, len(values)):
        next_value = values[i] * weight + exponential_moving_average[i - period] * (1 - weight)
        exponential_moving_average.append(next_value)
    exponential_moving_average.pop(0)
    return exponential_moving_average


def get_pivots(period: int, points: List[Tuple[int, float]]) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Given a period and formatted points (see regressions format_data_points())

    return the points with the highest and lowest y-values in each segment of period points

    Preconditions:
        - 1 <= period <= len(points)
    """
    high = []
    low = []

    i = 0
    while i < len(points):
        if i + period > len(points):
            segment = points[i:]
        else:
            segment = points[i:i + period]

        hi = -math.inf
        lo = math.inf

        for point in segment:
            if point[1] < lo:
                lo = point[1]
            if point[1] > hi:
                hi = point[1]

        for point in segment:
            if point[1] == lo:
                low.append((point[0], point[1]))
            if point[1] == hi:
                high.append((point[0], point[1]))
        i += period

    return high, low


def determine_support(period: int, points: List[Tuple[int, float]]) -> Tuple[float, float]:
    """Given a period and a list of points, determine via linear regression of the
    pivot points, the support line.

    Return floats a, b describing the line y = a + b * x.

    Preconditions:
        - 1 <= period <= len(points)
    """
    _, low = get_pivots(period, points)
    return regressions.linear_regression(low)


def determine_resistance(period: int, points: List[Tuple[int, float]]) -> Tuple[float, float]:
    """Given a period and a list of points, determine via linear regression of the
    pivot points, the resistance line.

    Return floats a, b describing the line y = a + b * x.

    Preconditions:
        - 1 <= period <= len(points)
    """
    high, _ = get_pivots(period, points)
    return regressions.linear_regression(high)


if __name__ == '__main__':
    import python_ta
    import python_ta.contracts

    # uncomment this if want to see python_ta report

    # python_ta.contracts.DEBUG_CONTRACTS = False
    # python_ta.contracts.check_all_contracts()
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'filter',
    #         'constants',
    #         'math',
    #         'regressions',
    #         'matplotlib.pyplot',
    #         'python_ta.contracts'
    #     ],  # the names (strs) of imported modules
    #     'allowed-io': [],  # the names (strs) of functions that call print/open/input
    #     'max-line-length': 150,
    #     'disable': ['R1705', 'C0200']
    # })
