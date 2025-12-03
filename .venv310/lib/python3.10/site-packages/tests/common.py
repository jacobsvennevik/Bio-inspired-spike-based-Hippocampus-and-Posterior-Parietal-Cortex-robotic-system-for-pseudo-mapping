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

import socket
import json
import threading


class MockServer(object):
    """A mock JSON line sender/receiver server."""

    def __init__(self):
        self._server_socket = socket.socket(socket.AF_INET,
                                            socket.SOCK_STREAM)
        self._server_socket.setsockopt(socket.SOL_SOCKET,
                                       socket.SO_REUSEADDR, 1)

        self._sock = None
        self._buf = b""
        self._started = threading.Event()

    def listen(self):
        """Start listening to the server port."""
        self._server_socket.bind(("", 22244))
        self._server_socket.listen(1)

    def connect(self):
        """Wait for a client to connect."""
        self._started.set()
        self._sock, _addr = self._server_socket.accept()

    def close(self):
        if self._sock is not None:
            self._sock.close()
        self._sock = None
        self._buf = b""
        self._server_socket.close()

    def send(self, obj):
        """Send a JSON object to the client."""
        self._sock.send(json.dumps(obj).encode("utf-8") + b"\n")

    def recv(self):
        """Recieve a JSON object."""
        while b"\n" not in self._buf:
            data = self._sock.recv(1024)
            if len(data) == 0:  # pragma: no cover
                raise OSError("Socket closed!")
            self._buf += data

        line, _, self._buf = self._buf.partition(b"\n")
        return json.loads(line.decode("utf-8"))
