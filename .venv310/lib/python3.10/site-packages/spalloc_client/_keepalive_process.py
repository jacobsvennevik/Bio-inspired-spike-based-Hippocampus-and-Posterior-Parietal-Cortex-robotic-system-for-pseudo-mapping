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

""" A script for keeping Spalloc jobs alive, intended to only ever be run\
    from the Spalloc client itself.
"""
import sys
import threading
from typing import List
from spalloc_client.protocol_client import ProtocolClient, ProtocolTimeoutError


def wait_for_exit(stop_event: threading.Event) -> None:
    """ Listens to stdin for a line equal to 'exit' or end-of-file and then\
        notifies the given event (that it is time to stop keeping the Spalloc\
        job alive).

    :param stop_event: Used to notify another thread that is time to stop.
    """
    for line in sys.stdin:
        if line.strip() == "exit":
            break
    stop_event.set()


def keep_job_alive(
        hostname: str, port: int, job_id: int, keepalive_period: float,
        timeout: float, reconnect_delay: float,
        stop_event: threading.Event) -> None:
    """ Keeps a Spalloc job alive. Run as a separate process to the main\
        Spalloc client.

    :param hostname: The address of the Spalloc server.
    :param port: The port of the Spalloc server.
    :param job_id: The ID of the Spalloc job to keep alive.
    :param keepalive_period: \
        How long will the job live without a keep-alive message being sent.
    :param timeout: The communication timeout.
    :param reconnect_delay: \
        The delay before reconnecting on communication failure.
    :param stop_event: Used to notify this function that it is time to stop \
        keeping the job alive.
    """
    client = ProtocolClient(hostname, port)
    client.connect(timeout)

    # Send the keepalive packet twice as often as required
    if keepalive_period is not None:
        keepalive_period /= 2.0
    while not stop_event.wait(keepalive_period):

        # Keep trying to send the keep-alive packet, if this fails,
        # keep trying to reconnect until it succeeds.
        while not stop_event.is_set():
            try:
                client.job_keepalive(job_id, timeout=timeout)
                break
            except (ProtocolTimeoutError, IOError, OSError):
                # Something went wrong, reconnect, after a delay which
                # may be interrupted by the thread being stopped

                # pylint: disable=protected-access
                client._close()
                if not stop_event.wait(reconnect_delay):
                    try:
                        client.connect(timeout)
                    except (IOError, OSError):
                        client.close()


def _run(argv: List[str]) -> None:
    print("KEEPALIVE")
    sys.stdout.flush()
    hostname = argv[1]
    port = int(argv[2])
    job_id = int(argv[3])
    keepalive = float(argv[4])
    timeout = float(argv[5])
    reconnect_delay = float(argv[6])

    # Set things up so that we can detect when to stop
    stop_event = threading.Event()
    stdin_watcher = threading.Thread(target=wait_for_exit, args=(stop_event,))
    stdin_watcher.daemon = True
    stdin_watcher.start()

    # Start keeping the job alive
    keep_job_alive(hostname, port, job_id, keepalive, timeout,
                   reconnect_delay, stop_event)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        sys.stderr.write(
            "wrong # args: should be '" + sys.argv[0] + " hostname port "
            "job_id keepalive_delay comms_timeout reconnect_delay'\n")
        sys.exit(1)
    _run(sys.argv)
