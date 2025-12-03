# Copyright (c) 2016-2019 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from collections import OrderedDict
import pytest
from spalloc.term import (
    Terminal, render_table, render_definitions, render_boards, render_cells,
    DEFAULT_BOARD_EDGES)


@pytest.mark.parametrize("force", [True, False])
def test_force(force):
    t = Terminal(force=force)
    assert t.enabled is force


def test_call():
    t = Terminal()
    t.enabled = False
    assert t("foo") == ""
    t.enabled = True
    assert t("foo") == "foo"


def test_clear_screen():
    t = Terminal(force=True)
    assert t.clear_screen() == "\033[2J\033[;H"
    t.enabled = False
    assert t.clear_screen() == ""


def test_update():
    t = Terminal(force=True)

    # First time just save the cursor
    assert t.update() == "\0337"

    # Subsequent times restore the cursor and clear the line
    assert t.update() == "\0338\033[K"
    assert t.update() == "\0338\033[K"
    assert t.update() == "\0338\033[K"

    # Start again
    assert t.update(start_again=True) == "\0337"
    assert t.update() == "\0338\033[K"
    assert t.update() == "\0338\033[K"

    # Wrap a string
    assert t.update("foo", start_again=True) == "\0337foo"
    assert t.update("bar") == "\0338\033[Kbar"

    # Cast to string
    assert t.update(123) == "\0338\033[K123"

    # Disable
    t.enabled = False
    assert t.update(start_again=True) == ""
    assert t.update() == ""
    assert t.update() == ""

    assert t.update("foo", start_again=True) == "foo"
    assert t.update("bar") == "bar"
    assert t.update(123) == "123"


def test_set_attr():
    t = Terminal(force=True)

    # Empty list
    assert t.set_attrs() == ""

    # Single item
    assert t.set_attrs([1]) == "\033[1m"

    # Many items
    assert t.set_attrs([1, 2, 3]) == "\033[1;2;3m"

    # When disabled should do nothing
    t.enabled = False
    assert t.set_attrs() == ""
    assert t.set_attrs([1]) == ""
    assert t.set_attrs([1, 2, 3]) == ""


def test_wrap():
    t = Terminal()

    # Default do nothing
    assert t.wrap("foo") == "foo"

    # Cast to string
    assert t.wrap(123) == "123"

    # Pre and post
    assert t.wrap(123, pre="<") == "<123"
    assert t.wrap(123, post=">") == "123>"
    assert t.wrap(123, pre="<", post=">") == "<123>"

    # Without string should just get pre
    assert t.wrap(pre="<") == "<"
    assert t.wrap(post=">") == ""
    assert t.wrap(pre="<", post=">") == "<"


def test_getattr():
    t = Terminal(force=True)

    # Single things should work
    assert t.reset() == "\033[0m"
    assert t.red() == "\033[31m"
    assert t.bg_red() == "\033[41m"

    # Multiple things should too
    assert t.red_bg_blue_bright() == "\033[31;44;1m"

    # Should wrap strings
    assert t.green("I'm a tree") == "\033[32mI'm a tree\033[0m"

    # When disabled should do nothing but passthrough
    t.enabled = False
    assert t.red_bg_blue_bright() == ""
    assert t.red_bg_blue_bright("foo") == "foo"

    # Should fail when unrecognised things appear
    with pytest.raises(AttributeError):
        t.bad
    with pytest.raises(AttributeError):
        t.red_bad
    with pytest.raises(AttributeError):
        t.bad_red


def test_render_table():
    t = Terminal(force=True)

    # Empty
    assert render_table([]) == ""

    # Singleton
    assert render_table([["a"]]) == "a"

    # Single row
    assert render_table([["a", "bc", "def"]]) == "a  bc  def"

    # Column padding
    assert render_table([
        ("a", "bc", "def"),
        ("def", "bc", "a"),
    ]) == ("a    bc  def\n"
           "def  bc  a")

    # Column padding and formatting
    assert render_table([
        ("a", "bc", "def"),
        ((t.red, "def"), "bc", "a"),
    ]) == ("a    bc  def\n"
           "\033[31mdef\033[0m  bc  a")

    # Casting from ints and right-aligning them
    assert render_table([
        ("a", "bc", "integers"),
        ("def", "bc", 1234),
    ]) == ("a    bc  integers\n"
           "def  bc      1234")

    # Formatted ints
    assert render_table([
        ("a", "bc", "integers"),
        ("def", "bc", (t.red, 1234)),
    ]) == ("a    bc  integers\n"
           "def  bc      \033[31m1234\033[0m")


def test_render_definitions():
    # Empty case
    assert render_definitions(OrderedDict()) == ""

    # Singleton
    assert render_definitions(OrderedDict([("foo", "bar")])) == "foo: bar"

    # Ragged
    assert render_definitions(OrderedDict([
        ("Key", "Value"),
        ("Something", "Else"),
        ("Another", "Thing"),
    ])) == ("      Key: Value\n"
            "Something: Else\n"
            "  Another: Thing")

    # Alternative seperator
    assert render_definitions(OrderedDict([("foo", "bar")]),
                              seperator="=") == "foo=bar"
    # Linebreaks
    assert render_definitions({"Key": "Lines\nBroken\nUp."}) == (
        "Key: Lines\n"
        "     Broken\n"
        "     Up."
    )


INNER_BOARD_EDGES = ("===", "`", ",")
OUTER_BOARD_EDGES = DEFAULT_BOARD_EDGES


class TestRenderBoards(object):

    def test_empty(self):
        assert render_boards([]) == ""

    def test_single(self):
        out = render_boards([
            ([(0, 0, 0)], "ABC", INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
        ])
        assert out == (r" ___."
                       r"/ABC\."
                       r"\___/".replace(".", "\n"))

    def test_three_boards(self):
        out = render_boards([
            ([(0, 0, z) for z in range(3)], "ABC",
             INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
        ])
        assert out == (r" ___."
                       r"/ABC\___."
                       r"\===,ABC\."
                       r"/ABC`___/."
                       r"\___/"
                       ).replace(".", "\n")

    def test_many_boards(self):
        out = render_boards([
            ([(x, y, z)
              for x in range(2)
              for y in range(2)
              for z in range(3)], "ABC",
             INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
        ])
        assert out == (r" ___     ___."
                       r"/ABC\___/ABC\___."
                       r"\===,ABC`===,ABC\."
                       r"/ABC`===,ABC`===/."
                       r"\___,ABC`===,ABC\___."
                       r"    \===,ABC`===,ABC\."
                       r"    /ABC`___,ABC`___/."
                       r"    \___/   \___/"
                       ).replace(".", "\n")

    def test_dead_links(self):
        out = render_boards([
            ([(x, y, z)
              for x in range(2)
              for y in range(2)
              for z in range(3)], "ABC",
             INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
        ], dead_links=set([
            (0, 0, 0, 0),  # 0, 0, East
            (0, 0, 0, 2),  # 0, 0, North
            (0, 0, 0, 1),  # 0, 0, North East
            (0, 0, 1, 4),  # 1, 1, South West
        ]))
        assert out == (r" ___     ___."
                       r"/ABC\___/ABC\___."
                       r"\===,ABC`===,ABC\."
                       r"/ABC`===,ABC`===/."
                       r"\___,ABC`===,ABC\___."
                       r"    \XXX,ABC`===,ABC\."
                       r"    /ABCX___,ABC`___/."
                       r"    \___X   \___/"
                       ).replace(".", "\n")

    def test_multiple_board_groups(self):
        out = render_boards([
            ([(0, 0, z) for z in range(3)], " # ",
             INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
            ([(x, y, z)
              for x in range(2) for y in range(2) for z in range(3)
              if (x, y) != (0, 0)], "ABC",
             INNER_BOARD_EDGES, OUTER_BOARD_EDGES),
        ])
        assert out == (r" ___     ___."
                       r"/ABC\___/ABC\___."
                       r"\===,ABC`===,ABC\."
                       r"/ABC`___,ABC`===/."
                       r"\___/ # \___,ABC\___."
                       r"    \===, # \===,ABC\."
                       r"    / # `___/ABC`___/."
                       r"    \___/   \___/"
                       ).replace(".", "\n")


class TestRenderCells(object):

    def test_empty(self):
        assert render_cells([]) == ""

    def test_single(self):
        assert render_cells([(3, "foo")]) == "foo"

    def test_even_spacing(self):
        assert render_cells([
            (3, "foo"),
            (4, "food"),
            (5, "food!"),
        ]) == "foo    food   food!"

    @pytest.mark.parametrize("width", [5, 10])
    def test_full_line_breaking(self, width):
        assert render_cells([
            (3, "foo"),
            (4, "food"),
            (5, "food!"),
        ], width) == ("foo\n"
                      "food\n"
                      "food!")

    def test_not_quite_breaking(self):
        assert render_cells([
            (3, "foo"),
            (4, "food"),
            (5, "food!"),
        ], 11) == ("foo    food\n"
                   "food!")

    def test_column_alignment(self):
        assert render_cells([
            (3, "foo"),
            (4, "food"),
            (5, "food!"),
            (1, "!"),
        ], 11) == ("foo    food\n"
                   "food!  !")
