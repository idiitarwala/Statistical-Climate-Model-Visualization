from typing import List, Tuple, Dict
from dataclasses import dataclass
import math


@dataclass
class Dataset:
    """A dataclass holding dataset filtered by a specific month.

    Representation Invariants:
        - 1 <= self.month <= 12
        - len(self.data) > 0
    """
    month: int
    data: List[Tuple[str, float]]


def read_dataset(path: str) -> List[Tuple[str, float]]:
    """Given a path, read tab delimited dataset and
    return a list of rows

    Preconditions:
        - len(path) > 0
    """
    data = []
    with open(path, 'r+') as fp:
        for line in fp:
            text = line.split('\t')
            data.append((text[0], float(text[1])))
    return data


def central_tendency(values: List[float]) -> Dict[str, float]:
    """Given a list of floats
    return the mean and median

    Preconditions:
        - len(values) > 0

    >>> expected = {'mean': 3.0, 'median': 2.0}
    >>> actual = central_tendency([1.0, 6.0, 2.0])
    >>> actual == expected
    True

    >>> expected = {'mean': 13.25, 'median': 12.0}
    >>> actual = central_tendency([17.0, 25.0, 4.0, 7.0])
    >>> actual == expected
    True
    """
    v = sorted(values)
    ret = {}
    n = len(v)
    ret['mean'] = sum(v) / n
    if n % 2 == 0:
        ret['median'] = (v[n // 2 - 1] + v[n // 2]) / 2
    else:
        ret['median'] = v[(n - 1) // 2]
    return ret


def filter_data(dataset: List[Tuple[str, float]]) -> Dict[str, Dataset]:
    """Given a dataset, filter out the hottest, median, and coldest months
    return a dictionary containing this information

    Preconditions:
        - len(dataset) > 0
        - int(dataset[0][0].split('-')[1]) == 1
    """
    # Precondition assures the first row is a January

    avg_temperatures = []
    filtered_dataset = []

    for month in range(1, 12 + 1):
        month_data = Dataset(month, [dataset[i]
                                     for i in range(0, len(dataset))
                                     if i % 12 == month - 1])
        filtered_dataset.append(month_data)
        temps = [row[1] for row in month_data.data]
        stats = central_tendency(temps)
        avg_temperatures.append(stats['mean'])

    # Now process the information
    max_t = max(avg_temperatures)
    min_t = min(avg_temperatures)
    med_t = central_tendency(avg_temperatures)['median']

    # Take month (any if two are same) with max, min, med average temperatures
    ret = {'hottest': [filtered_dataset[i] for i in range(0, len(filtered_dataset))
                       if max_t == avg_temperatures[i]][0],
           'coldest': [filtered_dataset[i] for i in range(0, len(filtered_dataset))
                       if min_t == avg_temperatures[i]][0]}

    # If the median is not actually in avg_temperatures (ie. the length is even), then take one of two closest
    if len(avg_temperatures) % 2 == 0:
        # We actually know this will always be even since 12 months
        # central_tendency is a good helper function for other modules, so instead of modifying central_tendency,
        # we modify this instead (albeit it's a little crude)

        index = 0
        diff = math.inf

        for i in range(0, len(avg_temperatures)):
            if abs(avg_temperatures[i] - med_t) < diff:
                diff = abs(avg_temperatures[i] - med_t)
                index = i

        ret['median'] = filtered_dataset[index]
    else:
        ret['median'] = [filtered_dataset[i] for i in range(0, len(filtered_dataset))
                         if med_t == avg_temperatures[i]][0]
    return ret


filtered_data = filter_data(read_dataset('data/temperatures.tab'))

if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
