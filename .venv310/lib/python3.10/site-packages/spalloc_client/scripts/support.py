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

from argparse import ArgumentParser, Namespace
import sys
from typing import Any, Optional
from spinn_utilities.abstract_base import AbstractBase, abstractmethod
from spalloc_client import (
    ProtocolClient, ProtocolError, ProtocolTimeoutError,
    SpallocServerException)
from spalloc_client.spalloc_config import SpallocConfig

# The acceptable range of server version numbers
VERSION_RANGE_START = (0, 1, 0)
# Note that v6 of the server is the Java-based one
VERSION_RANGE_STOP = (7, 0, 0)


class Terminate(Exception):
    """ Exception that can be used to exit the code """
    def __init__(self, code: int, message: Optional[str] = None):
        """
        :param code: error code
        :param message: error message
        """
        super().__init__()
        self._code = code
        self._msg = message

    def exit(self) -> None:
        """ Exit the program after printing an error msg. """
        if self._msg is not None:
            sys.stderr.write(self._msg + "\n")
        sys.exit(self._code)


def version_verify(client: ProtocolClient, timeout: Optional[float]) -> None:
    """
    Verify that the current version of the client is compatible
    """
    version = tuple(map(int, client.version(timeout=timeout).split(".")))
    if not (VERSION_RANGE_START <= version < VERSION_RANGE_STOP):
        raise Terminate(
            2, f"Incompatible server version ({'.'.join(map(str, version))})")


class Script(object, metaclass=AbstractBase):
    """ Base class of various Script Objects. """
    def __init__(self) -> None:
        self.client_factory = ProtocolClient

    def get_parser(self, cfg: SpallocConfig) -> ArgumentParser:
        """ Return a set-up instance of :py:class:`argparse.ArgumentParser`

        :param cfg: configuration object
        :returns: Arguments read from user call
        """
        raise NotImplementedError

    @abstractmethod
    def verify_arguments(self, args: Namespace) -> None:
        """ Check the arguments for sanity and do any second-stage parsing\
            required.
        """

    @abstractmethod
    def body(self, client: ProtocolClient, args: Namespace) -> int:
        """ How to do the processing of the script once a client has been\
            obtained and verified to be compatible.

        :returns: Success code (0 is OK)
        """
        raise NotImplementedError

    def build_server_arg_group(self, server_args: Any,
                               cfg: SpallocConfig) -> None:
        """
        Adds a few more arguments

        :param server_args:
        :param cfg: the configs
        """
        server_args.add_argument(
            "--hostname", "-H", default=cfg.hostname,
            help="hostname or IP of the spalloc server (default: %(default)s)")
        server_args.add_argument(
            "--port", "-P", default=cfg.port, type=int,
            help="port number of the spalloc server (default: %(default)s)")
        server_args.add_argument(
            "--timeout", default=cfg.timeout, type=float, metavar="SECONDS",
            help="seconds to wait for a response from the server (default: "
            "%(default)s)")
        server_args.add_argument(
            "--ignore_version", default=cfg.ignore_version, type=bool,
            help="Ignore the server version (WARNING: could result in errors) "
                 "default: %(default)s)")

    def __call__(self, argv: Optional[str] = None) -> int:
        cfg = SpallocConfig()
        parser = self.get_parser(cfg)
        server_args = parser.add_argument_group("spalloc server arguments")
        self.build_server_arg_group(server_args, cfg)
        args = parser.parse_args(argv)

        # Fail if server not specified
        if args.hostname is None:
            parser.error("--hostname of spalloc server must be specified")
        self.verify_arguments(args)

        try:
            with self.client_factory(args.hostname, args.port) as client:
                if not args.ignore_version:
                    version_verify(client, args.timeout)
                self.body(client, args)
                return 0
        except (IOError, OSError, ProtocolError, ProtocolTimeoutError) as e:
            sys.stderr.write(f"Error communicating with server: {e}\n")
            return 1
        except SpallocServerException as srv_exn:
            sys.stderr.write(f"Error from server: {srv_exn}\n")
            return 1
        except Terminate as t:
            t.exit()
            return 1
