"""This module contains the SheetInterface and Sheet class. They are used to generate the
HTML output. Details can be found in the final report.

By Freeman Cheng and Idris Tarwala.
"""
from typing import Dict
import matplotlib.pyplot as plt
import mpld3
from filter import filtered_data, Dataset


class SheetInterface:
    """Interface for the Sheet class."""
    _data: Dict[str, Dataset]
    _top: str
    _bot: str
    _content: str

    def __init__(self, data: Dict[str, Dataset]) -> None:
        """Initialize the Sheet object."""
        self._data = data
        self._top = '''
        <!DOCTYPE html>
        <html>
            <head>
                <title>
                    CSC110 Final Project Output Sheet
                </title>
            </head>
            <body>
        '''
        self._bot = '''
            </body>
        </html>
        '''
        self._content = ''

    def get_data(self) -> Dict[str, Dataset]:
        """Return the data from Sheet."""
        raise NotImplementedError

    def get_html(self) -> str:
        """Return the HTML code from Sheet."""
        raise NotImplementedError

    def add_figure(self, figure: plt.Figure) -> None:
        """Given a plt Figure, convert it to HTML and
        add it to the Sheet.
        """
        raise NotImplementedError

    def add_annotation(self, tag: str, text: str) -> None:
        """Add some text-based HTML attribute given the tag (<h1>, for example) and the text.

        Preconditions:
            - len(tag) >= 3

        >>> sheet = Sheet(filtered_data)
        >>> sheet.add_annotation('<h1>', 'This is an annotation.')
        >>> '<h1>This is an annotation.</h1>' in sheet.get_html()
        True
        """
        raise NotImplementedError

    def generate_sheet(self, path: str) -> None:
        """Given a path, save the HTML file to that path."""
        raise NotImplementedError


class Sheet(SheetInterface):
    """Sheet class containing methods to add annotations and figures
    to the HTML."""
    def __init__(self, data: Dict[str, Dataset]) -> None:
        """Initialize the Sheet object."""
        super().__init__(data)

    def get_data(self) -> Dict[str, Dataset]:
        """Return the data from Sheet."""
        return self._data

    def get_html(self) -> str:
        """Return the HTML code from Sheet."""
        return self._top + '\n' + self._content + '\n' + self._bot

    def add_figure(self, figure: plt.Figure) -> None:
        """Given a plt Figure, convert it to HTML and
        add it to the Sheet.
        """
        html = mpld3.fig_to_html(figure)
        self._content = self._content + '\n' + html

    def add_annotation(self, tag: str, text: str) -> None:
        """Add some text-based HTML attribute given the tag (<h1>, for example) and the text.

        Preconditions:
            - len(tag) >= 3

        >>> sheet = Sheet(filtered_data)
        >>> sheet.add_annotation('<h1>', 'This is an annotation.')
        >>> '<h1>This is an annotation.</h1>' in sheet.get_html()
        True
        """
        closing_tag = tag[0] + '/' + tag[1:]
        self._content = self._content + '\n' + tag + text + closing_tag

    def generate_sheet(self, path: str) -> None:
        """Given a path, save the HTML file to that path."""
        with open(path, 'w+') as fp:
            fp.write(self.get_html())


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
    #         'matplotlib.pyplot',
    #         'mpld3',
    #         'python_ta.contracts'
    #     ],  # the names (strs) of imported modules
    #     'allowed-io': [
    #         'generate_sheet'
    #     ],  # the names (strs) of functions that call print/open/input
    #     'max-line-length': 150,
    #     'disable': ['R1705', 'C0200', 'W0611', 'C0103']
    # })
