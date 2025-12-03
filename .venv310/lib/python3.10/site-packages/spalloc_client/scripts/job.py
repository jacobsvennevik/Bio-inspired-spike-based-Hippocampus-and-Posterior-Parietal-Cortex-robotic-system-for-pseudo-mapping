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
""" Command-line administrative job management interface.

``spalloc-job`` may be called with a job ID, or if no arguments supplied your
currently running job is shown by default. Various actions may be taken and
each is described below.

Displaying job information
--------------------------

By default, the command displays all known information about a job.

The ``--watch`` option may be added which will cause the output to be updated
in real-time as a job's state changes. For example::

    $ spalloc-job --watch

.. image:: _static/spalloc_job.gif
    :alt: spalloc-job displaying job information.

Controlling board power
-----------------------

The boards allocated to a job may be reset or powered on/off on demand (by
anybody, at any time) by adding the ``--power-on``, ``--power-off`` or
``--reset`` options. For example::

    $ spalloc-job --reset

.. note::

    This command blocks until the action is completed.

Listing board IP addresses
--------------------------

The hostnames of Ethernet-attached chips can be listed in CSV format by adding
the --ethernet-ips argument::

    $ spalloc-job --ethernet-ips
    x,y,hostname
    0,0,192.168.1.97
    0,12,192.168.1.105
    4,8,192.168.1.129
    4,20,192.168.1.137
    8,4,192.168.1.161
    8,16,192.168.1.169

Destroying/Cancelling Jobs
--------------------------

Jobs can be destroyed (by anybody, at any time) using the ``--destroy`` option
which optionally accepts a human-readable explanation::

    $ spalloc-job --destroy "Your job is taking too long..."

.. warning::

    That this "super power" should be used carefully since the user may not be
    notified that their job was destroyed and the first sign of this will be
    their boards being powered down and re-partitioned ready for another user.
"""
import argparse
import sys
from typing import Any, cast, Dict, Optional

from spinn_utilities.overrides import overrides
from spinn_utilities.typing.json import JsonObject

from spalloc_client import __version__, JobState
from spalloc_client.term import (
    Terminal, render_definitions, render_boards, DEFAULT_BOARD_EDGES)
from spalloc_client import ProtocolClient
from spalloc_client._utils import render_timestamp
from spalloc_client.spalloc_config import SpallocConfig
from spalloc_client.scripts.support import Terminate, Script


def _state_name(mapping: JsonObject) -> str:
    state = JobState(cast(int, mapping["state"]))
    return state.name  # pylint: disable=no-member


def show_job_info(t: Terminal, client: ProtocolClient,
                  timeout: Optional[float], job_id: int) -> None:
    """ Print a human-readable overview of a Job's attributes.

    Parameters
    ----------
    t :
        An output styling object for stdout.
    client :
        A connection to the server.
    timeout :
        The timeout for server responses.
    job_id :
        The job ID of interest.
    """
    # Get the complete job information (if the job is alive)
    job_list = client.list_jobs(timeout=timeout)
    jobs = [job for job in job_list if job["job_id"] == job_id]
    info: Dict[str, Any] = dict()
    info["Job ID"] = job_id

    if not jobs:
        # Job no longer exists, just print basic info
        job = cast(dict, client.get_job_state(job_id, timeout=timeout))
        info["State"] = _state_name(job)
        if job["reason"] is not None:
            info["Reason"] = job["reason"]
    else:
        # Job is enqueued, show all info
        machine_info = client.get_job_machine_info(job_id, timeout=timeout)
        job = cast(dict, jobs[0])

        info["Owner"] = job["owner"]
        info["State"] = _state_name(job)
        if job["start_time"] is not None:
            info["Start time"] = render_timestamp(job["start_time"])
        info["Keepalive"] = job["keepalive"]
        if "keepalivehost" in job and job["keepalivehost"] is not None:
            info["Owner host"] = job["keepalivehost"]

        args = cast(list, job["args"])
        kwargs = cast(dict, job["kwargs"])
        info["Request"] = "Job({}{}{})".format(
            ", ".join(map(str, args)),
            ",\n    " if args and kwargs else "",
            ",\n    ".join(f"{k}={v!r}" for k, v in sorted(kwargs.items()))
        )

        if job["boards"] is not None:
            info["Allocation"] = render_boards([(
                job["boards"],
                t.dim(" . "),
                tuple(map(t.dim, DEFAULT_BOARD_EDGES)),
                tuple(map(t.bright, DEFAULT_BOARD_EDGES)),
            )], [])

        if machine_info["connections"] is not None:
            connections = cast(list, machine_info["connections"])
            connections.sort()
            info["Hostname"] = connections[0][1]
        if machine_info["width"] is not None:
            info["Width"] = machine_info["width"]
        if machine_info["height"] is not None:
            info["Height"] = machine_info["height"]
        if job["boards"] is not None:
            info["Num boards"] = len(job["boards"])
        if job["power"] is not None:
            info["Board power"] = "on" if job["power"] else "off"
        if job["allocated_machine_name"] is not None:
            info["Running on"] = job["allocated_machine_name"]

    print(render_definitions(info))


def watch_job(t: Terminal, client: ProtocolClient, timeout: Optional[float],
              job_id: int) -> int:
    """ Re-print a job's information whenever the job changes.

    Parameters
    ----------
    t :
        An output styling object for stdout.
    client :
        A connection to the server.
    timeout :
        The timeout for server responses.
    job_id :
        The job ID of interest.

    Returns
    -------
    int
        An error code, 0 for success.
    """
    client.notify_job(job_id, timeout=timeout)
    while True:
        t.stream.write(t.clear_screen())
        show_job_info(t, client, timeout, job_id)

        try:
            client.wait_for_notification()
        except KeyboardInterrupt:
            # Gracefully exit
            return 0
        finally:
            print("")


def power_job(client: ProtocolClient, timeout: Optional[float],
              job_id: int, power: bool) -> None:
    """ Power a job's boards on/off and wait for the action to complete.

    Parameters
    ----------
    client :
        A connection to the server.
    timeout :
        The timeout for server responses.
    job_id :
        The job ID of interest.
    power :
        True = turn on/reset, False = turn off.

    """
    if power:
        client.power_on_job_boards(job_id, timeout=timeout)
    else:
        client.power_off_job_boards(job_id, timeout=timeout)

    # Wait for power command to complete...
    while True:
        client.notify_job(job_id, timeout=timeout)
        state = client.get_job_state(job_id, timeout=timeout)
        if state["state"] == JobState.ready:
            # Power command completed
            return

        if state["state"] == JobState.power:
            # Wait for change...
            try:
                client.wait_for_notification()
            except KeyboardInterrupt as exc:
                # If interrupted, quietly return an error state
                raise Terminate(7) from exc
        else:
            # In an unknown state, perhaps the job was queued etc.
            raise Terminate(
                8, (f"Error: Cannot power {'on' if power else 'off'} "
                    f"job {job_id} in state {_state_name(state)}"))


def list_ips(client: ProtocolClient, timeout: Optional[float],
             job_id: int) -> None:
    """ Print a CSV of board hostnames for all boards allocated to a job.

    Parameters
    ----------
    client :
        A connection to the server.
    timeout :
        The timeout for server responses.
    job_id :
        The job ID of interest.
    """
    info = client.get_job_machine_info(job_id, timeout=timeout)
    connections = cast(list, info["connections"])
    if connections is None:
        raise Terminate(9, f"Job {job_id} is queued or does not exist")
    print("x,y,hostname")
    connections.sort()
    for connection in connections:
        assert isinstance(connection, list)
        (xy, hostname) = connection
        assert isinstance(xy, list)
        (x, y) = xy
        print(f"{x},{y},{hostname}")


def destroy_job(client: ProtocolClient, timeout: Optional[float],
                job_id: int, reason: Optional[str] = None) -> None:
    """ Destroy a running job.

    Parameters
    ----------
    client :
        A connection to the server.
    timeout :
        The timeout for server responses.
    job_id :
        The job ID of interest.
    reason :
        The human-readable reason for destroying the job.
    """
    client.destroy_job(job_id, reason, timeout=timeout)


class ManageJobScript(Script):
    """
    A tool for running Job scripts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parser: Optional[argparse.ArgumentParser] = None

    def get_job_id(self, client: ProtocolClient,
                   args: argparse.Namespace) -> int:
        """
        :returns: ID for a job for the owner named in the args
        """
        if args.job_id is not None:
            return args.job_id
        # No Job ID specified, attempt to discover one
        jobs = client.list_jobs(timeout=args.timeout)
        job_ids = [job["job_id"] for job in jobs if job["owner"] == args.owner]
        if not job_ids:
            raise Terminate(3, f"Owner {args.owner} has no live jobs")
        elif len(job_ids) > 1:
            msg = (f"Ambiguous: {args.owner} has {len(job_ids)} live jobs: "
                   f"{', '.join(map(str, job_ids))}")
            raise Terminate(3, msg)
        return cast(int, job_ids[0])

    @overrides(Script.get_parser)
    def get_parser(self, cfg: SpallocConfig) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Manage running jobs.")
        parser.add_argument(
            "--version", "-V", action="version", version=__version__)
        parser.add_argument(
            "job_id", type=int, nargs="?",
            help="the job ID of interest, optional if the current owner only "
            "has one job")
        parser.add_argument(
            "--owner", "-o", default=cfg.owner,
            help="if no job ID is provided and this owner has only one job, "
            "this job is assumed (default: %(default)s)")
        control_args = parser.add_mutually_exclusive_group()
        control_args.add_argument(
            "--info", "-i", action="store_true",
            help="Show basic job information (the default)")
        control_args.add_argument(
            "--watch", "-w", action="store_true",
            help="watch this job for state changes")
        control_args.add_argument(
            "--power-on", "--reset", "-p", "-r", action="store_true",
            help="power-on or reset the job's boards")
        control_args.add_argument(
            "--power-off", action="store_true",
            help="power-off the job's boards")
        control_args.add_argument(
            "--ethernet-ips", "-e", action="store_true",
            help="output the IPs of all Ethernet connected chips as a CSV")
        control_args.add_argument(
            "--destroy", "-D", nargs="?", metavar="REASON", const="",
            help="destroy a queued or running job")
        self.parser = parser
        return parser

    @overrides(Script.verify_arguments)
    def verify_arguments(self, args: argparse.Namespace) -> None:
        if args.job_id is None and args.owner is None:
            assert self.parser is not None
            self.parser.error("job ID (or --owner) not specified")

    @overrides(Script.body)
    def body(self, client: ProtocolClient, args:  argparse.Namespace) -> int:
        jid = self.get_job_id(client, args)

        # Do as the user asked
        if args.watch:
            watch_job(Terminal(), client, args.timeout, jid)
        elif args.power_on:
            power_job(client, args.timeout, jid, True)
        elif args.power_off:
            power_job(client, args.timeout, jid, False)
        elif args.ethernet_ips:
            list_ips(client, args.timeout, jid)
        elif args.destroy is not None:
            # Set default destruction message
            if args.destroy == "" and args.owner:
                args.destroy = f"Destroyed by {args.owner}"
            destroy_job(client, args.timeout, jid, args.destroy)
        else:
            show_job_info(Terminal(), client, args.timeout, jid)
        return 0


main = ManageJobScript()
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
