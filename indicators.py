from typing import List
import matplotlib.pyplot as plt
from filter import filtered_data, Dataset


def get_x_values(dataset: Dataset) -> List[str]:
    """Given a Dataset
    return an array of its x-values
    """
    return [point[0] for point in dataset.data]


def get_y_values(dataset: Dataset) -> List[float]:
    """Given a Dataset
    return an array of its y-values
    """
    return [point[1] for point in dataset.data]


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


def calculate_exponential_moving_average(period: int, values: List[float], smoothing: float = 2) -> List[float]:
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


def determine_support():
    return


def determine_resistance():
    return


def calculate_fibonacci_retracement():
    return
