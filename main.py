"""This is the main module. The user should run this module to generate a HTML file
displaying quantitative and qualitative findings of computations performed on dataset.

For a more detailed analysis, please refer to the final report.

By Freeman Cheng and Idris Tarwala
"""
from filter import filtered_data
from sheet import Sheet
from constants import MONTHS

import regressions
import regressions_error
import indicators


def execute() -> None:
    """Create an output HTML file describing the findings of various computations
    performed on dataset.
    """
    output = Sheet(filtered_data)

    output.add_annotation('<h1>', 'CSC110 Final Project Output Sheet')
    output.add_annotation('<p>', 'By Idris Tarwala and Freeman Cheng')

    output.add_annotation('<h1>', 'General Overview')
    output.add_annotation('<p>', '''
    The dataset we used is taken from Pangaea, a data publisher for Earth and environmental studies.
    The data showcases the average monthly temperature in the Adda basin river area in the Central Alps. Taking the
    data from this area was a good choice as the temperature in the Central Alps varies throughout the year. This
    will allow us to make better quantitative and qualitative estimations about climate trends. In order to get a clear
    understanding of trends in the dataset, we have decided to use data dating back to the 1800’s. With our large scope
    of data, it will allow us to be more accurate with our findings and hence our results. <br />
    ''')
    output.add_annotation('<p>', '''
    We determined which month was the hottest, median, and coldest on average. 
    Our results involve these three months.
    ''')

    # Quantitative Findings
    output.add_annotation('<h2>', 'Quantitative Findings')

    types = ['Linear', 'Exponential', 'Polynomial']
    graphs = [output.get_data()['hottest'], output.get_data()['median'], output.get_data()['coldest']]
    titles = ['hottest', 'median', 'coldest']
    for t in types:
        for i in range(0, len(graphs)):
            head, tail = regressions_error.get_head_tail(graphs[i])

            figure = None
            error = None
            prediction = None

            if t == 'Linear':
                figure = regressions.plot_data_linear_regression(graphs[i])
                prediction = regressions.predict_linear(graphs[i], 2100)
                error = regressions_error.linear_regression_error(head, tail)
            elif t == 'Exponential':
                figure = regressions.plot_data_exponential_regression(
                    graphs[i])
                prediction = regressions.predict_exponential(graphs[i], 2100)
                error = regressions_error.exponential_regression_error(
                    head, tail)
            elif t == 'Polynomial':
                figure = regressions.plot_polynomial_regression(graphs[i])

            output.add_annotation(
                '<h3>', t + ' regression performed on ' + titles[i] + ' month.')
            output.add_annotation(
                '<p>', 'The ' + titles[i] + ' month was ' + MONTHS[graphs[i].month] + '.')

            output.add_figure(figure)

            if t in ('Linear', 'Exponential'):
                output.add_annotation('<h4>', 'Error and Predictions')
                output.add_annotation('<p>', 'Using the first half of our ' + MONTHS[graphs[i].month]
                                      + ' data, we can predict the second half with an average percentage error of '
                                      + str(round(error, 2)) + '%.')

                output.add_annotation('<p>', 'The regression predicts the average ' + MONTHS[graphs[i].month]
                                      + ' temperature to be '
                                      + str(round(prediction, 2)) + ' °C in the year 2100.')

    # Qualitative Findings
    output.add_annotation('<h2>', 'Qualitative Findings')

    types = ['Simple', 'Exponential']
    for t in types:
        for i in range(0, len(graphs)):
            if t == 'Simple':
                figure_1 = indicators.plot_simple_moving_average(graphs[i], True, False)
                figure_2 = indicators.plot_simple_moving_average(graphs[i], False, True)
            else:
                figure_1 = indicators.plot_ema(graphs[i], True, False)
                figure_2 = indicators.plot_ema(graphs[i], False, True)

            output.add_annotation(
                '<h3>', t + ' moving average performed on ' + titles[i] + ' month.')
            output.add_annotation(
                '<p>', 'The ' + titles[i] + ' month was ' + MONTHS[graphs[i].month] + '.')
            output.add_figure(figure_1)
            output.add_figure(figure_2)
            output.add_annotation('<p>', 'The support and resistance lines indicate'
                                         'the general trend which the temperatures '
                                         'are heading towards.')

    output.generate_sheet('output/sheet.html')


if __name__ == '__main__':
    import pytest
    import python_ta
    import python_ta.contracts

    pytest.main(['tests.py', '-vv'])

    # uncomment this if want to see python_ta report

    # python_ta.contracts.DEBUG_CONTRACTS = False
    # python_ta.contracts.check_all_contracts()
    # python_ta.check_all(config={
    #     'extra-imports': [
    #         'filter',
    #         'sheet',
    #         'constants',
    #         'regressions',
    #         'regressions_error',
    #         'indicators',
    #         'python_ta.contracts'
    #     ],  # the names (strs) of imported modules
    #     'allowed-io': [
    #         'execute'
    #     ],  # the names (strs) of functions that call print/open/input
    #     'max-line-length': 150,
    #     'disable': ['R1705', 'C0200']
    # })

    execute()
