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

import tempfile
import shutil
import os.path
import pytest
from spalloc.config import read_config


@pytest.yield_fixture
def tempdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def filename(tempdir):
    filename = os.path.join(tempdir, "f1")
    return filename


def test_priority(tempdir):
    f1 = os.path.join(tempdir, "f1")
    f2 = os.path.join(tempdir, "f2")

    with open(f1, "w") as f:
        f.write("[spalloc]\nport=123\nhostname=bar")
    with open(f2, "w") as f:
        f.write("[spalloc]\nport=321\ntags=qux")

    cfg = read_config([f1, f2])

    assert cfg["port"] == 321
    assert cfg["reconnect_delay"] == 5.0
    assert cfg["hostname"] == "bar"
    assert cfg["tags"] == ["qux"]


@pytest.mark.parametrize(
    "option_name,config_value,value",
    [("hostname", None, None),
     ("hostname", "foo", "foo"),
     ("port", None, 22244),
     ("port", "1234", 1234),
     ("owner", None, None),
     ("owner", "foo", "foo"),
     ("keepalive", None, 60.0),
     ("keepalive", "None", None),
     ("keepalive", "3.0", 3.0),
     ("reconnect_delay", None, 5.0),
     ("reconnect_delay", "3.0", 3.0),
     ("timeout", None, 5.0),
     ("timeout", "None", None),
     ("timeout", "3.0", 3.0),
     ("machine", None, None),
     ("machine", "None", None),
     ("machine", "foo", "foo"),
     ("tags", None, None),
     ("tags", "None", None),
     ("tags", "foo", ["foo"]),
     ("tags", "one, two , three", ["one", "two", "three"]),
     ("min_ratio", None, 0.333),
     ("min_ratio", "1.0", 1.0),
     ("max_dead_boards", None, 0),
     ("max_dead_boards", "None", None),
     ("max_dead_boards", "3", 3),
     ("max_dead_links", None, None),
     ("max_dead_links", "None", None),
     ("max_dead_links", "3", 3),
     ("require_torus", None, False),
     ("require_torus", "False", False),
     ("require_torus", "True", True)])
def test_options(filename, option_name, config_value, value):
    # Test all config options.

    # Write config file (omitting the config value if None, e.g. to test
    # default value)
    with open(filename, "w") as f:
        f.write("[spalloc]\n")
        if config_value is not None:
            f.write("{}={}".format(option_name, config_value))

    cfg = read_config([filename])

    assert option_name in cfg
    assert cfg[option_name] == value
