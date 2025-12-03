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

import os
import sys
import threading
import tempfile
import pytest
from mock import Mock
from spalloc import ProtocolClient
from spalloc.config import SEARCH_PATH
from .common import MockServer


@pytest.yield_fixture
def no_config_files(monkeypatch):
    # Prevent discovery of config files during test
    before = SEARCH_PATH[:]
    del SEARCH_PATH[:]
    yield
    SEARCH_PATH.extend(before)


@pytest.yield_fixture
def s():
    # A mock server
    s = MockServer()
    yield s
    s.close()


@pytest.yield_fixture
def c():
    c = ProtocolClient("localhost")
    yield c
    c.close()


@pytest.yield_fixture
def bg_accept(s):
    # Accept the first conncetion in the background
    started = threading.Event()

    def accept_and_listen():
        s.listen()
        started.set()
        s.connect()
    t = threading.Thread(target=accept_and_listen)
    t.start()
    started.wait()
    yield t
    s.close()
    t.join()


@pytest.yield_fixture
def basic_config_file(monkeypatch):
    # Sets up a basic config file with known and non-default values for all
    # fields
    fd, filename = tempfile.mkstemp()
    with os.fdopen(fd, "w") as f:
        f.write("[spalloc]\n"
                "hostname=localhost\n"
                "port=22244\n"
                "owner=me\n"
                "keepalive=1.0\n"
                "reconnect_delay=2.0\n"
                "timeout=3.0\n"
                "machine=m\n"
                "tags=foo, bar\n"
                "min_ratio=4.0\n"
                "max_dead_boards=5\n"
                "max_dead_links=6\n"
                "require_torus=True\n")
    before = SEARCH_PATH[:]
    del SEARCH_PATH[:]
    SEARCH_PATH.append(filename)
    yield
    del SEARCH_PATH[:]
    SEARCH_PATH.extend(before)
    os.remove(filename)


@pytest.fixture
def basic_job_kwargs():
    # The kwargs set by the basic_config_file fixture
    return dict(hostname="localhost",
                port=22244,
                reconnect_delay=2.0,
                timeout=3.0,
                owner="me",
                keepalive=1.0,
                machine="m",
                tags=None,  # As machine is not None
                min_ratio=4.0,
                max_dead_boards=5,
                max_dead_links=6,
                require_torus=True)


@pytest.fixture
def no_colour(monkeypatch):
    isatty = Mock(return_value=False)
    monkeypatch.setattr(sys, "stdout",
                        Mock(write=sys.stdout.write, isatty=isatty))
    monkeypatch.setattr(sys, "stderr",
                        Mock(write=sys.stdout.write, isatty=isatty))
