# Copyright (c) 2016 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" Utilities for generating ASCII/ANSI terminal graphics.
"""

import os
import sys
from itertools import chain
from collections import defaultdict
from enum import IntEnum
from functools import partial
from typing import (
    Callable, Dict, Iterable, List, Optional, TextIO, Tuple, Union)
from typing_extensions import TypeAlias

# pylint: disable=wrong-spelling-in-docstring

TableFunction: TypeAlias = Callable[[Union[int, str]], str]
TableValue: TypeAlias = Union[int, str]
TableColumn: TypeAlias = Union[TableValue, Tuple[TableFunction, TableValue]]
TableRow: TypeAlias = Iterable[TableColumn]
TableType: TypeAlias = List[TableRow]


class ANSIDisplayAttributes(IntEnum):
    """ Code numbers of ANSI display attributes for use with `ESC[...m`\
        sequences.
    """
    # pylint: disable=invalid-name
    reset = 0
    bright = 1
    dim = 2
    underscore = 4
    blink = 5
    reverse = 7
    hidden = 8

    # foreground colours
    black = 30
    red = 31
    green = 32
    yellow = 33
    blue = 34
    magenta = 35
    cyan = 36
    white = 37

    # background colours
    bg_black = 40
    bg_red = 41
    bg_green = 42
    bg_yellow = 43
    bg_blue = 44
    bg_magenta = 45
    bg_cyan = 46
    bg_white = 47


class Terminal(object):
    """ ANSI terminal control shenanigans.

    Utilities for printing colourful output and re-printing the screen on ANSI
    terminals. When output is not directed to a TTY, or when running under
    Windows, no ANSI control characters are produced.

    Examples::

        t = Terminal()

        # Printing in colours
        print(t.red("I'm in red!"))

        # Updating a status line
        for num in range(100):
            print(t.update("Now at {}%".format(num)))
            time.sleep(0.05)

        # Combining style attributes
        print(t.bg_red_white_blink("Woah!"))

    This module was inspired by the 'blessings' module which I initially liked
    but proved to be just a little too buggy.

    Attributes
    ----------
    stream
        The IO stream which is being used.
    enabled :
        Is colour enabled?
    """

    def __init__(self, stream: Optional[TextIO] = None,
                 force: Optional[bool] = None):
        """
        Parameters
        ----------
        stream
            The IO stream being written to (by default sys.stdout).
        force :
            If a bool, forces styling to be enabled or disabled as specified.
            If None, checks whether the stream is a TTY (and that we're not o
            non-posix OS) before enabling colouring automatically.
        """
        self.stream = stream if stream is not None else sys.stdout

        if force is None:
            self.enabled = os.name == "posix" and self.stream.isatty()
        else:
            self.enabled = force

        self._location_saved = False

    def __call__(self, string: str) -> str:
        """ If enabled, passes through the given value, otherwise passes\
            through an empty string.

        :returns: The string if the terminal is enabled otherwise ""
        """
        return string if self.enabled else ""

    def clear_screen(self) -> str:
        """ Clear the screen and reset cursor to top-left corner.

        :return: terminal control sequences
        """
        return self("\033[2J\033[;H")

    def update(self, string: str = "", start_again: bool = False) -> str:
        """
        Print before a line and it will replace the previous line prefixed\
            with :py:meth:`.update`.

        :param string: The string to print (optional).
        :param start_again:
            If False, overwrites the last thing printed. If True, starts a new
            line.

        :return: String preceded with terminal control sequence(s)
        """
        if start_again:
            self._location_saved = False

        if not self._location_saved:
            # No previous line to update, just save the cursor.
            self._location_saved = True
            return "".join((self("\0337"), str(string)))
        # Restore to previous location and clear line.
        return "".join((self("\0338\033[K"), str(string)))

    def set_attrs(self, attrs: List) -> str:
        """
        :returns: An ANSI control sequence which sets the given attribute
            numbers.
        """
        if not attrs:
            return ""
        return self(f"\033[{';'.join(str(attr) for attr in attrs)}m")

    def wrap(self, string: Optional[str] = None,
             pre: str = "", post: str = "") -> str:
        """ Wrap a string in the suppled pre and post strings or just print\
            the pre string if no string given

        :returns: pre + string + post or just pre if string is None
        """
        if string is None:
            return pre
        return "".join((pre, str(string), post))

    def __getattr__(self, name: str) -> partial:
        """ Implements all the 'magic' style methods.
        """
        attrs = []
        while name:
            # pylint: disable=not-an-iterable
            for attr in ANSIDisplayAttributes:
                if name.startswith(attr.name):
                    attrs.append(int(attr))
                    name = name[len(attr.name):].lstrip("_")
                    break
            else:
                # No attr name matched! Fail!
                raise AttributeError(name)
        return partial(self.wrap,
                       pre=self.set_attrs(attrs),
                       post=self("\033[0m"))


def render_table(table: TableType, column_sep: str = "  ") -> str:
    """ Render an ASCII table with optional ANSI escape codes.

    An example table::

        Something   Another thing  Finally
        some value  woah              1234
        ace         duuued              -1
        magic       rather good       9001

    Parameters
    ----------
    table :
        A table to render. Each row contains an iterable of column values which
        may be either values or a tuples (f, value) where value is the string
        to print, or an integer to print right-aligned. If a column is a tuple,
        f is a formatting function which is applied to the string before the
        table is finally displayed. Columns are padded to have matching widths
        *before* any formatting functions are applied.
    column_sep :
        String inserted between each column.

    Returns
    -------
    str
        The formatted table.
    """
    # Determine maximum column widths
    column_widths: Dict[int, int] = defaultdict(lambda: 0)
    for row in table:
        for i, column in enumerate(row):
            if isinstance(column, str):
                string = column
            elif isinstance(column, int):
                string = str(column)
            else:
                string = str(column[1])
            column_widths[i] = max(len(str(string)), column_widths[i])

    # Render the table cells with padding [[str, ...], ...]
    out = []
    for row in table:
        rendered_row: List[str] = []
        out.append(rendered_row)
        f: TableFunction
        for i, column in enumerate(row):
            # Get string length and formatted string
            if isinstance(column, str):
                string = column
                length = len(string)
                right_align = False
            elif isinstance(column, int):
                string = str(column)
                length = len(string)
                right_align = True
            else:
                f = column[0]
                value = column[1]
                if isinstance(value, str):
                    length = len(value)
                    right_align = False
                else:
                    length = len(str(value))
                    right_align = True
                string = f(value)

            padding = " " * (column_widths[i] - length)
            if right_align:
                rendered_row.append(padding + string)
            else:
                rendered_row.append(string + padding)

    # Render the final table
    return "\n".join(column_sep.join(row).rstrip() for row in out)


def render_definitions(definitions: Dict, separator: str = ": ") -> str:
    """ Render a definition list.

    Such a list looks like this::

              Key: Value
        Something: Else
          Another: Thing,
                   but this time with
                   line
                   breaks!

    Parameters
    ----------
    definitions :
        The key/value set to display.
    separator :
        The separator inserted between keys and values.

    Returns
    -------
    The Dict formatted into rows and columns
    """
    # Special case since max would fail
    if not definitions:
        return ""

    col_width = max(map(len, definitions))
    return "\n".join("{:>{}s}{}{}".format(
        key, col_width, separator, str(value).replace(
            "\n", f"\n{' '*(col_width + len(separator))}"))
        for key, value in definitions.items())


def _board_to_cartesian(x: int, y: int, z: int) -> Tuple[int, int]:
    r""" Translate from logical board coordinates (x, y, z) into Cartesian
        coordinates for printing hexagons.

    Example coordinates::

         ___     ___
        /-15\___/1 5\___
        \___/0 4\___/3 4\
        /-13\___/1 3\___/
        \___/0 2\___/2 2\___
            \___/1 1\___/3 1\
            /0 0\___/2 0\___/
            \___/   \___/

    Parameters
    ----------
    x :
        The logical board's X coordinate.
    y :
        The logical board's Y coordinate.
    z :
        The logical board's Z coordinate.

    Returns
    -------
    x, y :
        Equivalent Cartesian coordinates.
    """  # noqa: W605
    cx = (2*x) - y + (1 if z == 1 else 0)
    cy = (3*y) + z

    return (cx, cy)


_LINK_TO_EDGE = {
    0: (+1, -1, 2),  # E
    1: (+1, +0, 1),  # NE
    2: (+0, +1, 0),  # N
    3: (+0, +0, 2),  # W
    4: (+0, -1, 1),  # SW
    5: (+0, -1, 0),  # S
}
r""" Mapping from link direction to board edge.

We define and number a board's link directions as::

         N 2
         ___
     3 W/   \ NE 1
    4 SW\___/ E 0
         S 5

Boards have the following *three* edge numbers::

      ___
    2/   \
    1\___/
       0

This means that, for example, the North 'edge' is actually represented as a
South edge on the board above. As a result this lookup table maps each link
direction to both a delta (in cartesian coordinates) and edge number.

{link: (dx, dy, edge), ...}
"""

_LINK_TO_DELTA = {
    0: (+1, -1),  # E
    1: (+1, +1),  # NE
    2: (+0, +2),  # N
    3: (-1, +1),  # W
    4: (-1, -1),  # SW
    5: (-0, -2),  # S
}
""" The Cartesian offsets of the immediate neighbouring boards.
"""


DEFAULT_BOARD_EDGES = ("___", "\\", "/")
""" The default board edge styles.
"""


def render_boards(
        board_groups: List[Tuple[List[Tuple[int, int, int]], str,
                           Tuple[str, str, str], Tuple[str, str, str]]],
        dead_links: List,
        dead_edge: Tuple[str, str, str] = ("XXX", "X", "X"),
        blank_label: str = "   ",
        blank_edge: Tuple[str, str, str] = ("   ", " ", " ")) -> str:
    r""" Render an ASCII art diagram of a set of boards with sets of boards.

    For example::

         ___     ___     ___
        / . \___/ . \___/ . \___
        \___/ . \___/ . \___/ . \
        / . \___/ . \___/ . \___/
        \___/   \___/   \___/

    Parameters
    ----------
    board_groups :
        Lists the groups of boards to display. Label is a 3-character string
        labelling the boards in the group, edge_inner and edge_outer are the
        characters to use to draw board edges as a tuple ("___", "\\", "/")
        which are to be used for the inner and outer board edges respectively.
        Board groups are drawn sequentially with later board groups obscuring
        earlier ones when their edges or boards overlap.
    dead_links :
        Enumeration of all dead links. These links are re-drawn in the style
        defined by the dead_edge argument after all board groups have been
        drawn.
    dead_edge :
        The strings to use to draw dead links.
        ("___", "\\", "/")
    blank_label :
        The 3-character string to use to label non-existent boards. (Blank by
        default)
    blank_edge :
        The characters to use to render non-existent board edges. (Blank by
        default)
        ("___", "\\", "/")

    Returns
    -------
    Boards as ASCII art diagram
    """
    # pylint: disable=too-many-locals

    # {(x, y): string_types, ...}
    board_labels = {}
    # {(x, y, edge): str, ...}
    board_edges = {}

    # The set of all boards defined (used to filter displaying of dead links to
    # non-existent boards
    all_boards = set()

    for _boards, label, edge_inner, edge_outer in board_groups:
        # Convert to Cartesian coordinates
        boards = set(_board_to_cartesian(x, y, z) for x, y, z in _boards)
        all_boards.update(boards)

        # Set board labels and basic edge style
        for x, y in boards:
            board_labels[(x, y)] = label

            for link in range(6):
                dx, dy = _LINK_TO_DELTA[link]
                x2 = x + dx
                y2 = y + dy

                edx, edy, edge = _LINK_TO_EDGE[link]
                if (x2, y2) in boards:
                    style = edge_inner[edge]
                else:
                    style = edge_outer[edge]
                ex = x + edx
                ey = y + edy
                board_edges[(ex, ey, edge)] = style

    # Mark dead links
    for x, y, z, link in dead_links:
        x, y = _board_to_cartesian(x, y, z)
        edx, edy, edge = _LINK_TO_EDGE[link]
        ex = x + edx
        ey = y + edy
        board_edges[(ex, ey, edge)] = dead_edge[edge]

    # Get the bounds of the size of diagram to render
    all_xy = tuple(chain(all_boards, ((x, y) for x, y, edge in board_edges)))
    if not all_xy:
        return ""  # Special case since min/max will fail otherwise
    x_min, y_min = map(min, zip(*all_xy))
    x_max, y_max = map(max, zip(*all_xy))

    # Render row-by-row
    #   ___     ___            6 Even
    #  /-15\___/1 5\___        5 Odd
    #  \___/0 4\___/3 4\       4 Even
    #  /-13\___/1 3\___/       3 Odd
    #  \___/0 2\___/2 2\___    2 Even
    #  .   \___/1 1\___/3 1\   1 Odd
    #  .   /0 0\___/2 0\___/   0 Even
    #  .   \___/   \___/      -1 Odd
    # -1   0   1   2   3   4
    #  Odd Even Odd Even Odd Even
    out = []
    for y in range(y_max, y_min - 1, -1):
        even_row = (y % 2) == 0
        row = ""
        for x in range(x_min, x_max + 1):
            even_col = (x % 2) == 0
            if even_row == even_col:
                row += board_edges.get((x, y, 2), blank_edge[2])
                row += board_labels.get((x, y), blank_label)
            else:
                row += board_edges.get((x, y, 1), blank_edge[1])
                row += board_edges.get((x, y, 0), blank_edge[0])
        out.append(row)

    return "\n".join(filter(None, map(str.rstrip, out)))


def render_cells(cells: List[Tuple[int, str]], width: int = 80,
                 col_spacing: int = 2) -> str:
    """ Given a list of short (~10 char) strings, display these aligned in\
        columns.

    Example output::

        Something  like       this       can        be
        used       to         neatly     arrange    long
        sequences  of         values     in         a
        compact    format.

    Parameters
    ----------
    cells :
        Gives the cells to print as tuples giving the strings length in visible
        characters and the string to display.
    width :
        The width of the terminal.
    col_spacing :
        Size of the gap to leave between columns.

    Returns
    -------
    Cells in formatted columns
    """
    # Special case (since max below will fail)
    if not cells:
        return ""

    # Columns should be at least as large as the largest cell with padding
    # between columns
    col_width = max(strlen for strlen, s in cells) + col_spacing

    lines = [""]
    cur_length = 0
    for strlen, s in cells:
        # Once line is full, move to the next
        if cur_length + strlen > width:
            lines.append("")
            cur_length = 0

        # Add the current cell (with spacing)
        lines[-1] += s + (" "*(col_width - strlen))
        cur_length += col_width

    return "\n".join(map(str.rstrip, lines))
