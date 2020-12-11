from typing import List, Tuple, Any
from math import log10
import matplotlib.pyplot as plt
from filter import filtered_data, Dataset


def plot_data_linear_regression(dataset: Dataset) -> Any:
    """This function takes in a dataset and the corresponding a and b values calculated in
    linear_regression to graph a scatter plot and its corresponding linear regression
    with the equation y = a + bx

    This function returns the figure with the graphed scatter plot and its corresponding
    linear regression
    """
    x_value = sort_x_values(dataset)
    y_value = sort_y_values(dataset)
    # calculates the a and b value for the corresponding dataset
    formatted_points = format_data_points(x_value, y_value)
    a_and_b = linear_regression(formatted_points)

    a = a_and_b[0]
    b = a_and_b[1]

    figure = plt.figure()  # creates a new figure
    # plots a scatter plot with the given x and y values
    plt.scatter(x_value, y_value, label='points', color='k', s=10)
    plt.xlabel('Year')
    plt.title('Climate Data with a Linear Regression')
    plt.ylabel('Temperature (°C)')
    # Now we want to overlay the regression function on the scatter plot
    start_year = x_value[0]
    end_year = x_value[-1]
    # calculates the x and y values of the regression
    x_regression_values = [year for year in range(start_year, end_year + 1)]
    y_regression_values = [a + (b * x) for x in range(start_year, end_year + 1)]
    # plots the x and y values of the regression over the current scatter plot
    plt.plot(x_regression_values, y_regression_values)

    plt.show()

    return figure


def plot_data_exponential_regression(dataset: Dataset) -> Any:
    """This function takes in a dataset and the corresponding a and r values calculated in
    exponential_regression to graph a scatter plot and its corresponding exponential regression
     with the equation y = a * (r) ** x

    This function returns the figure with the graphed scatter plot and its corresponding
    exponential regression
    """
    x_value = sort_x_values(dataset)
    y_value = sort_y_values(dataset)
    # calculates the a and r value for the corresponding dataset
    formatted_points = format_data_points(x_value, y_value)
    a_and_r = exponential_regression(formatted_points)

    a = a_and_r[0]
    r = a_and_r[1]

    figure = plt.figure()  # creates a new figure
    # plots a scatter plot with the given x and y values
    plt.scatter(x_value, y_value, label='points', color='k', s=10)
    plt.xlabel('Year')
    plt.title('Climate Data with an Exponential Regression')
    plt.ylabel('Temperature (°C)')
    # Now we want to overlay the regression function on the scatter plot
    start_year = x_value[0]
    end_year = x_value[-1]
    # plots the x and y values of the regression over the current scatter plot
    x_regression_values = [year for year in range(start_year, end_year + 1)]
    y_regression_values = [a * (r ** x) for x in range(start_year, end_year + 1)]
    # plots the x and y values of the regression over the current scatter plot
    plt.plot(x_regression_values, y_regression_values)

    plt.show()

    return figure


def predict_linear(dataset: Dataset, year: int) -> float:
    """This function determines the temperature at any given year based on the linear regression
    of the given dataset.

    This function returns a float representing the temperature at the given year

    Preconditions:
        - year >= sort_x_values(dataset)[0]
        - year <= 2200
    """
    x_values = sort_x_values(dataset)
    y_values = sort_y_values(dataset)

    formatted_points = format_data_points(x_values, y_values)
    a_and_b = linear_regression(formatted_points)
    a = a_and_b[0]
    b = a_and_b[1]

    return a + (b * year)


def predict_exponential(dataset: Dataset, year: int) -> float:
    """This function determines the temperature at any given year based on the
     exponential regression of the given dataset.

     This function returns a float representing the temperature at the given year

     Preconditions:
         - year >= sort_x_values(dataset)[0]
         - year <= 2200
     """
    x_values = sort_x_values(dataset)
    y_values = sort_y_values(dataset)

    formatted_points = format_data_points(x_values, y_values)
    a_and_b = exponential_regression(formatted_points)
    a = a_and_b[0]
    r = a_and_b[1]

    return a * r ** year


###############################################################################
# Helper Functions
###############################################################################
def sort_x_values(dataset: Dataset) -> List[int]:
    """The x values of the the plot correspond to the year of a given temperature value.

    Returns a list of integers containing all the years in the dataset.

    Preconditions:
        - len(dataset.data != 0)

    """
    data_list = dataset.data
    # makes a new list with only dates as strings in the form 'xxxx-xx'
    date_string_list = [data_list[i][0] for i in range(0, len(data_list))]
    # splits the strings at the "-" to separate year and month
    split_data = [date_string_list[i].split('-') for i in range(0, len(date_string_list))]
    # makes a new list containing only years as strings which will serve as the x values of the plot
    str_list_value = [split_data[i][0] for i in range(0, len(split_data))]
    # finally creates a new list with years represented as integers instead of strings
    x_value_list = [int(str_list_value[i]) for i in range(0, len(str_list_value))]

    return x_value_list


def sort_y_values(dataset: Dataset) -> List[float]:
    """The y values of the plot corresponds to the temperature in a given year.

    Returns a list of floats containing all the temperature values for a given year.

    Preconditions:
        - len(dataset.data != 0)

    """
    data_list = dataset.data
    # takes all the temperature values from data_list and creates a separate list
    y_value_list = [data_list[i][1] for i in range(0, len(data_list))]

    return y_value_list


def format_data_points(x_values: List[int], y_values: List[float]) -> List[Tuple[int, float]]:
    """This function takes in a list of x_values and y_values and converts them into the
    form: (x_value, y_value)

    Return a list of tuples containing x and y values

    Preconditions:
     - len(x_values) == len(y_values)
    """
    formatted_list = []
    for i in range(0, len(x_values)):
        tpl = (x_values[i], y_values[i])
        formatted_list.append(tpl)

    return formatted_list


def linear_regression(points: List[Tuple[int, float]]) -> tuple:
    """This function takes in a list of points from the dataset and performs a linear regression
    on them.

    points is a list of pairs of floats: [(x, y), (x, y), ...]
    This function returns a tuple containing (a, b) which correspond to the y-intercept and slope
    of a linear equation in the form y = a + bx

    >>> dataset = filtered_data['hottest']
    >>> x = sort_x_values(dataset)
    >>> y = sort_y_values(dataset)
    >>> pts = format_data_points(x, y)
    >>> expected = (-6.946112543863368, 0.010389450245916662)
    >>> actual = linear_regression(pts)
    >>> expected == actual
    True
    """
    # list of all x and y values 
    all_x = [point[0] for point in points]
    all_y = [point[1] for point in points]
    # average x and y value for the given list of x and y values
    avg_x = sum(all_x) / len(all_x)
    avg_y = sum(all_y) / len(all_y)
    # Calculating the numerator and the denominator of the linear regression equation
    overall_numerator = sum((all_x[i] - avg_x) * (all_y[i] - avg_y) for i in range(0, len(points)))
    overall_denominator = sum([(all_x[i] - avg_x) ** 2 for i in range(0, len(points))])
    # calculating the corresponding a and b values for the linear regression
    b = overall_numerator / overall_denominator
    a = avg_y - (b * avg_x)

    return a, b


def exponential_regression(points: List[Tuple[int, float]]) -> tuple:
    """This function takes in a list of points from the dataset and performs an exponential
    regression on them.

       points is a list of pairs of floats: [(x, y), (x, y), ...]
       This function returns a tuple containing (a, r) which correspond to the the coefficients
       of an exponential equation in the form y = a * (r) ** x

    >>> dataset = filtered_data['hottest']
    >>> x = sort_x_values(dataset)
    >>> y = sort_y_values(dataset)
    >>> pts = format_data_points(x, y)
    >>> expected = (2.7544259293807842, 1.0008052637216431)
    >>> actual = exponential_regression(pts)
    >>> expected == actual
    True
    """
    # list of all x and y values
    all_x = [point[0] for point in points]
    all_y = [point[1] for point in points]
    # to calculate exponential regression first we want to find the linear regression of (x, log(y))
    all_log_y = []
    for number in all_y:
        all_log_y.append(log10(number))
    # reformatting new x and y points in the form (x, log(y))
    new_formatted_points = format_data_points(all_x, all_log_y)
    # calculating regression with reformatted points
    temp_regression_values = linear_regression(new_formatted_points)
    # based on the given values of temp_regression_values we can determine the coefficients a and r
    a = 10 ** temp_regression_values[0]
    r = 10 ** temp_regression_values[1]

    return a, r


if __name__ == '__main__':
    import doctest

    doctest.testmod(verbose=True)
