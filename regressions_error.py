"""This module calculates the average percentage error by using the first half of the dataset
to predict the second half. Details can be found in the final report.

By Freeman Cheng and Idris Tarwala.
"""
from typing import Tuple, List
from filter import Dataset
import regressions


def linear_regression_error(head: List[Tuple[int, float]], tail: List[Tuple[int, float]]) -> float:
    """Given the head and tail of a dataset
    return the average percentage error of the regression when using the head to predict the tail

    Preconditions:
        - len(head) > 0
        - len(tail) > 0
    """
    a, b = regressions.linear_regression(head)
    denominator = len(tail)
    cumulative_error = 0
    for point in tail:
        x = point[0]
        y = point[1]
        prediction = a + b * x
        error = abs((prediction - y) / y) * 100
        cumulative_error += error
    return cumulative_error / denominator


def exponential_regression_error(head: List[Tuple[int, float]], tail: List[Tuple[int, float]]) -> float:
    """Given the head and tail of a dataset
    return the average percentage error of the regression when using the head to predict the tail

    Preconditions:
        - len(head) > 0
        - len(tail) > 0
    """
    a, r, c = regressions.exponential_regression(head)
    denominator = len(tail)
    cumulative_error = 0
    for point in tail:
        x = point[0]
        y = point[1]
        prediction = a * (r ** x) + c
        error = abs((prediction - y) / y) * 100
        cumulative_error += error
    return cumulative_error / denominator


###############################################################################
# Helper Functions
###############################################################################
def get_head_tail(dataset: Dataset) \
        -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Given a Dataset, split it in two parts - a head and a tail: the head consisting of
    first half of data, tail consisting of second half (head rounds up if odd number
    of points);
    return the head and tail as formatted data points.

    Preconditions:
        - len(dataset.data) > 1

    >>> dataset = Dataset(month=1, data=[('1800-01', -2.3), ('1801-01', -3.2)])
    >>> expected = ([(1800, -2.3)], [(1801, -3.2)])
    >>> actual = get_head_tail(dataset)
    >>> actual == expected
    True

    >>> dataset = Dataset(month=1, data=[('1800-01', -2.3), ('1801-01', -3.2), ('1802-01', -7.1)])
    >>> expected = ([(1800, -2.3), (1801, -3.2)], [(1802, -7.1)])
    >>> actual = get_head_tail(dataset)
    >>> actual == expected
    True
    """
    x_values = regressions.sort_x_values(dataset)
    y_values = regressions.sort_y_values(dataset)
    points = regressions.format_data_points(x_values, y_values)
    n = len(points)
    head = points[0:n - n // 2]
    tail = points[n - n // 2: n]
    return head, tail


def get_head_tail_custom(dataset: Dataset, denominator: float) \
        -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Given a Dataset, split it in two parts - a head and a tail: the head consisting of
    first 1/denominator of data, tail consisting of the rest (head rounds up);
    return the head and tail as formatted data points.

    Preconditions:
        - denominator > 1
        - len(dataset.data[0:n - n//denominator]) < len(dataset.data)

    >>> dataset = Dataset(month=1, data=[('1800-01', -2.3), ('1801-01', -3.2)])
    >>> expected = ([(1800, -2.3)], [(1801, -3.2)])
    >>> actual = get_head_tail_custom(dataset, 2)
    >>> actual == expected
    True

    >>> dataset = Dataset(month=1, data=[('1800-01', -2.3), ('1801-01', -3.2), ('1802-01', -7.1)])
    >>> expected = ([(1800, -2.3), (1801, -3.2)], [(1802, -7.1)])
    >>> actual = get_head_tail_custom(dataset, 2)
    >>> actual == expected
    True
    """
    x_values = regressions.sort_x_values(dataset)
    y_values = regressions.sort_y_values(dataset)
    points = regressions.format_data_points(x_values, y_values)
    n = len(points)
    head = points[0:n - int(n // denominator)]
    tail = points[n - int(n // denominator): n]
    return head, tail


if __name__ == '__main__':
    import doctest
    import python_ta
    import python_ta.contracts

    doctest.testmod(verbose=True)

    # uncomment this if want to see python_ta report

    # python_ta.contracts.DEBUG_CONTRACTS = False
    # python_ta.contracts.check_all_contracts()
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'filter',
    #         'regressions',
    #         'python_ta.contracts'
    #     ],  # the names (strs) of imported modules
    #     'allowed-io': [],  # the names (strs) of functions that call print/open/input
    #     'max-line-length': 150,
    #     'disable': ['R1705', 'C0200', 'W0611', 'C0103']
    # })
