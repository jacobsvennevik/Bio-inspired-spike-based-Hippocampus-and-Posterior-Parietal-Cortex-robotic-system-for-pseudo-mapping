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

""" A simple blocking spalloc_server protocol implementation.
"""
from collections import deque
import errno
import json
import socket
from types import TracebackType
from typing import Any, cast, Dict, List, Literal, Optional, Type, Union
from threading import current_thread, RLock, local, Thread

from typing_extensions import Self

from spinn_utilities.typing.json import (
    JsonObject, JsonObjectArray, JsonValue)

from spalloc_client._utils import time_left, timed_out, make_timeout


class ProtocolError(Exception):
    """ Thrown when a network-level problem occurs during protocol handling.
    """


class ProtocolTimeoutError(Exception):
    """ Thrown upon a protocol-level timeout.
    """


class SpallocServerException(Exception):
    """ Thrown when something went wrong on the server side that caused us to\
        be sent a message.
    """


class _ProtocolThreadLocal(local):
    """ Subclass of threading.local to ensure that we get sane initialisation\
        of our state in each thread.
    """
    # See https://github.com/SpiNNakerManchester/spalloc/issues/12
    def __init__(self) -> None:
        super().__init__()
        self.buffer = b""
        self.sock: Optional[socket.socket] = None


class ProtocolClient(object):
    """ A simple (blocking) client implementation of the `spalloc-server
    <https://github.com/SpiNNakerManchester/spalloc_server>`_ protocol.

    This minimal implementation is intended to serve both simple applications
    and as an example implementation of the protocol for other applications.
    This implementation simply implements the protocol,
    presenting a Remote procedure call-like interface to the server.
    For a higher-level interface built on top of this
    client, see :py:class:`spalloc.Job`.

    Usage examples::

        # Connect to a spalloc_server
        with ProtocolClient("hostname") as c:
            # Call commands by name
            print(c.call("version"))  # '0.1.0'

            # Call commands as if they were methods
            print(c.version())  # '0.1.0'

            # Wait an event to be received
            print(c.wait_for_notification())  # {"jobs_changed": [1, 3]}

        # Done!
    """

    def __init__(self, hostname: str, port: int = 22244,
                 timeout: Optional[float] = None):
        """ Define a new connection.

        .. note::

            Does not connect to the server until :py:meth:`.connect` is called.

        Parameters
        ----------
        hostname :
            The hostname of the server.
        port :
            The port to use (default: 22244).
        """
        self._hostname = hostname
        self._port = port
        # Mapping from threads to sockets. Kept because we need to have way to
        # shut down all sockets at once.
        self._socks: Dict[Thread, socket.socket] = dict()
        # Thread local variables
        self._local = _ProtocolThreadLocal()
        # A queue of unprocessed notifications
        self._notifications: deque = deque()
        self._dead = True
        self._socks_lock = RLock()
        self._notifications_lock = RLock()
        self._default_timeout = timeout

    def __enter__(self) -> Self:
        self.connect(self._default_timeout)
        return self

    def __exit__(self, exc_type: Optional[Type],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> Literal[False]:
        self.close()
        return False

    def _get_connection(self, timeout: Optional[float]) -> socket.socket:
        if self._dead:
            raise OSError(errno.ENOTCONN, "not connected")
        connect_needed = False
        key = current_thread()
        with self._socks_lock:
            sock = self._socks.get(key, None)
            if sock is None:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # The socket connected to the server or None if disconnected.
                self._socks[key] = sock
                connect_needed = True

        if connect_needed:
            # A buffer for incoming, but incomplete, lines of data
            self._local.buffer = b""
            self._local.sock = sock
            sock.settimeout(timeout)
            if not self._do_connect(sock):  # pragma: no cover
                self._close(key)
                return self._get_connection(timeout)

        sock.settimeout(timeout)
        return sock

    def _do_connect(self, sock: socket.socket) -> bool:
        success = False
        try:
            sock.connect((self._hostname, self._port))
            success = True
        except OSError as e:
            if e.errno != errno.EISCONN:
                raise
        return success

    def _has_open_socket(self) -> bool:
        return self._local.sock is not None

    def connect(self, timeout: Optional[float] = None) -> None:
        """(Re)connect to the server.

        Raises
        ------
        OSError, IOError
            If a connection failure occurs.
        """
        # Close any existing connection
        if self._has_open_socket():
            self._close()
        self._dead = False
        self._connect(timeout)

    def _connect(self, timeout: Optional[float]) -> socket.socket:
        """ Try to (re)connect to the server.
        """
        try:
            return self._get_connection(timeout)
        except (IOError, OSError):
            # Failure, try again...
            self._close()
            # Pass on the exception
            raise

    def _close(self, key: Optional[Thread] = None) -> None:
        if key is None:
            key = current_thread()
        with self._socks_lock:
            sock = self._socks.get(key, None)
            if sock is None:  # pragma: no cover
                return
            del self._socks[key]
        if key == current_thread():
            self._local.sock = None
            self._local.buffer = b""
        sock.close()

    def close(self) -> None:
        """ Disconnect from the server.
        """
        self._dead = True
        with self._socks_lock:
            keys = list(self._socks.keys())
        for key in keys:
            self._close(key)
        self._local = _ProtocolThreadLocal()

    def _recv_json(self, timeout: Optional[float] = None) -> JsonObject:
        """ Receive a line of JSON from the server.

        Parameters
        ----------
        timeout :
            The number of seconds to wait before timing out or None if this
            function should try again forever.

        Returns
        -------
        object or None
            The unpacked JSON line received.

        Raises
        ------
        ProtocolTimeoutError
            If a timeout occurs.
        OSError
            If the socket is unusable or becomes disconnected.
        """
        sock = self._get_connection(timeout)

        # Wait for some data to arrive
        while b"\n" not in self._local.buffer:
            try:
                data = sock.recv(1024)
            except socket.timeout as e:
                raise ProtocolTimeoutError("recv timed out.") from e

            # Has socket closed?
            if not data:
                raise OSError("Connection closed.")

            self._local.buffer += data

        # Unpack and return the JSON
        line, _, self._local.buffer = self._local.buffer.partition(b"\n")
        return json.loads(line.decode("utf-8"))

    def _send_json(
            self, obj: JsonObject, timeout: Optional[float] = None) -> None:
        """ Attempt to send a line of JSON to the server.

        Parameters
        ----------
        obj :
            The object to serialise.
        timeout :
            The number of seconds to wait before timing out or None if this
            function should try again forever.

        Raises
        ------
        ProtocolTimeoutError
            If a timeout occurs.
        OSError
            If the socket is unusable or becomes disconnected.
        """
        sock = self._get_connection(timeout)

        # Send the line
        data = json.dumps(obj).encode("utf-8") + b"\n"
        try:
            if sock.send(data) != len(data):
                # If can't send whole command at once, just fail
                raise OSError("Could not send whole command.")
        except socket.timeout as e:
            raise ProtocolTimeoutError("send timed out.") from e

    def call(self, name: str, timeout: Optional[float],
             *args: Union[int, str, None],
             **kwargs: Any) -> JsonValue:
        """
        Send a command to the server and return the reply.

        :param name:  The name of the command to send.
        :param timeout:
            The number of seconds to wait before timing out or None if this
            function should wait forever. (Default: None)
        :param args:
        :param kwargs:
        :returns: The object returned by the server.
        :raises ProtocolTimeoutError: If a timeout occurs.
        :raises ProtocolError:
            If the connection is unavailable or is closed.
        """
        try:
            finish_time = make_timeout(timeout)

            # Construct the command message
            command: JsonObject = {}
            command["command"] = name
            command["args"] = list(args)
            command["kwargs"] = kwargs
            self._send_json(command, timeout=timeout)

            # Command sent! Attempt to receive the response...
            while not timed_out(finish_time):
                obj = self._recv_json(timeout=time_left(finish_time))
                if "return" in obj:
                    # Success!
                    return obj["return"]
                if "exception" in obj:
                    raise SpallocServerException(obj["exception"])
                # Got a notification, keep trying...
                with self._notifications_lock:
                    self._notifications.append(obj)

            raise ProtocolTimeoutError(f"{timeout=} passed!")
        except (IOError, OSError) as e:
            raise ProtocolError(str(e)) from e

    def wait_for_notification(
            self, timeout: Optional[float] = None) -> Optional[JsonObject]:
        """ Return the next notification to arrive.

        Parameters
        ----------
        timeout :
            The number of seconds to wait before timing out or None if this
            function should try again forever.

            If negative only responses already-received will be returned. If no
            responses are available, in this case the function does not raise a
            ProtocolTimeoutError but returns None instead.

        Returns
        -------
        object
            The notification sent by the server.

        Raises
        ------
        ProtocolTimeoutError
            If a timeout occurs.
        ProtocolError
            If the socket is unusable or becomes disconnected.
        """
        # If we already have a notification, return it
        with self._notifications_lock:
            if self._notifications:
                return self._notifications.popleft()

        # Check for a duff timeout
        if timeout is not None and timeout < 0.0:
            return None

        # Otherwise, wait for a notification to arrive
        try:
            return self._recv_json(timeout)
        except (IOError, OSError) as e:  # pragma: no cover
            raise ProtocolError(str(e)) from e

    # The bindings of the Spalloc protocol methods themselves; simplifies use
    # from IDEs.

    def version(self, timeout: Optional[float] = None) -> str:
        """ Ask what version of spalloc is running.

        :returns: The string returned by the call to the server.
        """
        return cast(str, self.call("version", timeout))

    def create_job(self, timeout: Optional[float], *args: int,
                   owner: Optional[str] = None,
                   keepalive: Optional[float] = None,
                   machine: Optional[str] = None,
                   tags: Optional[List[str]] = None,
                   min_ratio: Optional[float] = None,
                   max_dead_boards: Optional[int] = None,
                   max_dead_links: Optional[int] = None,
                   require_torus: Optional[bool] = None) -> int:
        """
        Start a new job

        :returns: The ID returned by the call to the server.
        """
        # If no owner, don't bother with the call
        if owner is None:
            raise SpallocServerException(
                "owner must be specified for all jobs.")
        if tags is not None and machine is not None:
            raise SpallocServerException(
                f"Unexpected {tags=} and {machine=} are both not None")
        return cast(int, self.call(
            "create_job", timeout, *args, owner=owner,
            keepalive=keepalive, machine=machine, tags=tags,
            min_ratio=min_ratio, max_dead_boards=max_dead_boards,
            max_dead_links=max_dead_links, require_torus=require_torus))

    def job_keepalive(self, job_id: int,
                      timeout: Optional[float] = None) -> JsonObject:
        """
        Send s message to keep the job alive.

        Without these the job will be killed after a while.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("job_keepalive", timeout, job_id))

    def get_job_state(self, job_id: int,
                      timeout: Optional[float] = None) -> JsonObject:
        """Get the state for this job

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("get_job_state", timeout, job_id))

    def get_job_machine_info(self, job_id: int,
                             timeout: Optional[float] = None) -> JsonObject:
        """ Get info for this job.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("get_job_machine_info", timeout, job_id))

    def power_on_job_boards(self, job_id: int,
                            timeout: Optional[float] = None) -> JsonObject:
        """ Turn on the power on the jobs boards.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("power_on_job_boards", timeout, job_id))

    def power_off_job_boards(self, job_id: int,
                             timeout: Optional[float] = None) -> JsonObject:
        """ Turn off the power on the jobs boards.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("power_off_job_boards", timeout, job_id))

    def destroy_job(self, job_id: int, reason: Optional[str] = None,
                    timeout: Optional[float] = None) -> JsonObject:
        """ Destroy the job

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("destroy_job", timeout,
                                    job_id, reason=reason))

    def notify_job(self, job_id: Optional[int] = None,
                   timeout: Optional[float] = None) -> JsonObject:
        """ Turn on notification of job status changes.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("notify_job", timeout, job_id))

    def no_notify_job(self, job_id: Optional[int] = None,
                      timeout: Optional[float] = None) -> JsonObject:
        """ Turn off notification of job status changes.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("no_notify_job", timeout,
                                    job_id))

    def notify_machine(self, machine_name: Optional[str] = None,
                       timeout: Optional[float] = None) -> JsonObject:
        """ Turn on notification of machine status changes.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("notify_machine", timeout,
                                    machine_name))

    def no_notify_machine(self, machine_name: Optional[str] = None,
                          timeout: Optional[float] = None) -> JsonObject:
        """ Turn off notification of machine status changes.

        :returns: The dict returned by the call to the server.
        """
        return cast(dict, self.call("no_notify_machine", timeout,
                                    machine_name))

    def list_jobs(self, timeout: Optional[float] = None) -> JsonObjectArray:
        """ Obtains a list of jobs currently running.

        :returns: The list returned by the call to the server.
        """
        return cast(list, self.call("list_jobs", timeout))

    def list_machines(self,
                      timeout: Optional[float] = None) -> JsonObjectArray:
        """ Obtains a list of currently supported machines.

        :returns: The list returned by the call to the server.
        """
        return cast(list, self.call("list_machines", timeout))

    def get_board_position(
            self, machine_name: str, x: int, y: int, z: int,
            timeout: Optional[float] = None) -> JsonObject:  # pragma: no cover
        """ Gets the position of board x, y, z on the given machine.

        :returns: The dict returned by the call to the server.
        """
        # pylint: disable=too-many-arguments
        return cast(dict, self.call("get_board_position", timeout,
                                    machine_name, x, y, z))

    def get_board_at_position(self, machine_name: str, x: int, y: int, z: int,
                              timeout: Optional[float] = None
                              ) -> JsonObject:  # pragma: no cover
        """ Gets the board x, y, z on the requested machine.

        :returns: The dict returned by the call to the server.
        """
        # pylint: disable=too-many-arguments
        return cast(dict, self.call("get_board_at_position", timeout,
                                    machine_name, x, y, z))

    _acceptable_kwargs_for_where_is = frozenset([
        frozenset("machine x y z".split()),
        frozenset("machine cabinet frame board".split()),
        frozenset("machine chip_x chip_y".split()),
        frozenset("job_id chip_x chip_y".split())])

    def where_is(self, job_id: int, chip_x: int, chip_y: int,
                 timeout: Optional[float] = None) -> JsonObject:
        """ Reports where on the Machine a job is running

        :returns: The dict returned by the call to the server.
        """
        # Test for whether sane arguments are passed.
        return cast(dict, self.call("where_is", timeout, job_id=job_id,
                                    chip_x=chip_x, chip_y=chip_y))
