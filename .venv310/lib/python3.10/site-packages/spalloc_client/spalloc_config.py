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

# pylint: disable=wrong-spelling-in-docstring
""" The spalloc command-line tool and Python library determine their default
configuration options from a spalloc configuration file if present.

.. note::

    Use of spalloc's configuration files is entirely optional as all
    configuration options may be presented as arguments to commands/methods at
    runtime.

By default, configuration files are read (in ascending order of priority) from
a system-wide configuration directory (e.g. ``/etc/xdg/spalloc``), user
configuration file (e.g. ``$HOME/.config/spalloc``) and finally the current
working directory (in a file named ``.spalloc``). The default search paths on
your system can be discovered by running::

    $ python -m spalloc.config

Config files use the Python :py:mod:`configparser` INI-format with a single
section, ``spalloc``, like so::

    [spalloc]
    hostname = localhost
    owner = jdh@cs.man.ac.uk

Though most users will only wish to specify the ``hostname`` and ``owner``
options (as in the example above), the following enumerates the complete set of
options available (and the default value).

``hostname``
    The hostname or IP address of the spalloc-server to connect to.
``owner``
    The name of the owner of created jobs. By convention the user's email
    address.
``port``
    The port used by the spalloc-server. (Default: 22244)
``keepalive``
    The keepalive interval, in seconds, to use when creating jobs. If the
    spalloc-server does not receive a keepalive command for this interval the
    job is automatically destroyed. May be set to None to disable this feature.
    (Default: 60.0)
``reconnect_delay``
    The time, in seconds, to wait between reconnection attempts to the server
    if disconnected. (Default 5.0)
``timeout``
    The time, in seconds, to wait before giving up waiting for a response from
    the server or None to wait forever. (Default 5.0)
``machine``
    The name of a specific machine on which to run all jobs or None to use any
    available machine. (Default: None)
``tags``
    The set of tags, comma separated, to require a machine to have when
    allocating jobs. (Default: default)
``min_ratio``
    Require that when allocating a number of boards the allocation is at least
    as square as this aspect ratio. (Default: 0.333)
``max_dead_boards``
    The maximum number of dead boards which may be present in an allocated set
    of boards or None to allow any number of dead boards. (Default: 0)
``max_dead_links``
    The maximum number of dead links which may be present in an allocated set
    of boards or None to allow any number of dead links. (Default: None)
``require_torus``
    If True, require that an allocation have wrap-around links. This typically
    requires the allocation of a whole machine. If False, wrap-around links may
    or may-not be present in allocated machines. (Default: False)
"""
import configparser
import os.path
from typing import List, Optional

import appdirs

# The application name to use in config file names
_name = "spalloc"

# Standard config file names/locations
SYSTEM_CONFIG_FILE = appdirs.site_config_dir(_name)
USER_CONFIG_FILE = appdirs.user_config_dir(_name)
CWD_CONFIG_FILE = os.path.join(os.curdir, f".{_name}")

# Search path for config files (lowest to highest priority)
SEARCH_PATH = [
    SYSTEM_CONFIG_FILE,
    USER_CONFIG_FILE,
    CWD_CONFIG_FILE,
]

TIMEOUT = 10.0

SECTION = "spalloc"
DEFAULT_CONFIG = {
    "port": "22244",
    "keepalive": "60.0",
    "reconnect_delay": "5.0",
    "timeout": str(TIMEOUT),
    "machine": "None",
    "tags": "None",
    "min_ratio": "0.333",
    "max_dead_boards": "0",
    "max_dead_links": "None",
    "require_torus": "False",
    "ignore_version": "False"}


def _read_none_or_float(
        parser: configparser.ConfigParser, option: str) -> Optional[float]:
    if parser.get(SECTION, option) == "None":
        return None
    return parser.getfloat(SECTION, option)


def _read_none_or_int(
        parser: configparser.ConfigParser, option: str) -> Optional[int]:
    if parser.get(SECTION, option) == "None":
        return None
    return parser.getint(SECTION, option)


def _read_any_str(
        parser: configparser.ConfigParser, option: str) -> Optional[str]:
    try:
        return parser.get(SECTION, option)
    except configparser.NoOptionError:
        return None


def _read_none_or_str(
        parser: configparser.ConfigParser, option: str) -> Optional[str]:
    if parser.get(SECTION, option) == "None":
        return None
    return parser.get(SECTION, option)


class SpallocConfig(object):
    """ Typed configs """

    __slots__ = ("_hostname", "_ignore_version", "_keepalive", "_machine",
                 "_max_dead_boards", "_max_dead_links", "_min_ratio",
                 "_owner", "_port", "_reconnect_delay", "_require_torus",
                 "_tags", "_timeout")

    def __init__(self, filenames: Optional[List[str]] = None):
        """ Attempt to read local configuration files
        to determine spalloc client settings.

        Parameters
        ----------
        filenames :
            Filenames to attempt to read.
            Later config file have higher priority.

        """
        if filenames is None:  # pragma: no cover
            filenames = SEARCH_PATH
        parser = configparser.ConfigParser()

        # Set default config values (NB: No read_dict in Python 2.7)
        parser.add_section(SECTION)
        for key, value in DEFAULT_CONFIG.items():
            parser.set(SECTION, key, value)

        # Attempt to read from each possible file location in turn
        for filename in filenames:
            try:
                with open(filename, "r", encoding="utf-8") as f:
                    parser.read_file(f, filename)
            except (IOError, OSError):
                # File did not exist, keep trying
                pass

        self._hostname = _read_any_str(parser, "hostname")
        self._owner = _read_any_str(parser, "owner")
        self._port = parser.getint(SECTION, "port")
        self._keepalive = _read_none_or_float(parser, "keepalive")
        self._reconnect_delay = parser.getfloat(SECTION, "reconnect_delay")
        self._timeout = _read_none_or_float(parser, "timeout")
        self._machine = _read_none_or_str(parser, "machine")
        self._min_ratio = parser.getfloat(SECTION, "min_ratio")
        self._max_dead_boards = _read_none_or_int(parser, "max_dead_boards")
        self._max_dead_links = _read_none_or_int(parser, "max_dead_links")
        self._require_torus = parser.getboolean(SECTION, "require_torus")
        self._ignore_version = parser.getboolean(SECTION, "ignore_version")

        tags = _read_none_or_str(parser, "tags")
        self._tags = None if tags is None else list(
            map(str.strip, tags.split(",")))

    @property
    def hostname(self) -> Optional[str]:
        """ Name of the spalloc server if specified """
        return self._hostname

    @property
    def ignore_version(self) -> bool:
        """ Flag to say version can be ignored """
        return self._ignore_version

    @property
    def keepalive(self) -> Optional[float]:
        """ Time to keep job allive """
        return self._keepalive

    @property
    def machine(self) -> Optional[str]:
        """ Name of the spalloc machine to use"""
        return self._machine

    @property
    def max_dead_boards(self) -> Optional[int]:
        """ How many dead boards are allowed in the job"""
        return self._max_dead_boards

    @property
    def max_dead_links(self) -> Optional[int]:
        """ How many dead links are allowed in the Job"""
        return self._max_dead_links

    @property
    def min_ratio(self) -> float:
        """ Min ratio"""
        return self._min_ratio

    @property
    def owner(self) -> Optional[str]:
        """ Owner to assign job to """
        return self._owner

    @property
    def port(self) -> int:
        """ Spalloc server port"""
        return self._port

    @property
    def reconnect_delay(self) -> float:
        """ Reconnect delay """
        return self._reconnect_delay

    @property
    def require_torus(self) -> bool:
        """ Flag to say a torus is required """
        return self._require_torus

    @property
    def tags(self) -> Optional[List[str]]:
        """ List of tags """
        return self._tags

    @property
    def timeout(self) -> Optional[float]:
        """ Time before a command should timeout """
        return self._timeout


if __name__ == "__main__":  # pragma: no cover
    print("Default search path (lowest-priority first):")
    print("\n".join(SEARCH_PATH))
