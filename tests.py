import filter


def test_read_dataset() -> None:
    """Ensure that all dates are in the right format
    """
    data = filter.read_dataset('data/temperatures.tab')
    assert (all(['-' in row[0] for row in data]))
    assert (all([len(row[0].split('-')[0]) == 4 and
                 len(row[0].split('-')[1]) == 2 for row in data]))
