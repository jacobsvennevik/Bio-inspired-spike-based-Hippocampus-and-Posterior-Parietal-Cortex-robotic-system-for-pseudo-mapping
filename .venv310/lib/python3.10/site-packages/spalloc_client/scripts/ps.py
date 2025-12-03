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

""" An administrative command-line process listing utility.

By default, the ``spalloc-ps`` command lists all running and queued jobs.  For
a real-time monitor of queued and running jobs, the ``--watch`` option may be
added.

.. image:: _static/spalloc_ps.png
    :alt: Jobs being listed by spalloc- process scripts

This list may be filtered by owner or machine with the ``--owner`` and
``--machine`` arguments.
"""
import argparse
from collections.abc import Sized
import sys
from typing import cast, Union

from spinn_utilities.overrides import overrides
from spinn_utilities.typing.json import JsonObjectArray

from spalloc_client import __version__, JobState, ProtocolClient
from spalloc_client.spalloc_config import SpallocConfig
from spalloc_client.term import Terminal, render_table, TableColumn, TableType
from spalloc_client._utils import render_timestamp
from .support import Script


def render_job_list(t: Terminal, jobs: JsonObjectArray,
                    args: argparse.Namespace) -> str:
    """
    :param t: The terminal to which the output will be sent.
    :param jobs: The list of jobs returned by the server.
    :param args:
    :return: A human-readable process listing.
    """
    table: TableType = []

    # Add headings
    table.append(((t.underscore_bright, "ID"),
                  (t.underscore_bright, "State"),
                  (t.underscore_bright, "Power"),
                  (t.underscore_bright, "Boards"),
                  (t.underscore_bright, "Machine"),
                  (t.underscore_bright, "Created at"),
                  (t.underscore_bright, "Keepalive"),
                  (t.underscore_bright, "Owner (Host)")))

    for job in jobs:
        # Filter jobs
        if args.machine is not None and \
                job["allocated_machine_name"] != args.machine:
            continue
        if args.owner is not None and job["owner"] != args.owner:
            continue

        # Colourise job states
        job_state: TableColumn
        if job["state"] == JobState.queued:
            job_state = (t.blue, "queue")
        elif job["state"] == JobState.power:
            job_state = (t.yellow, "power")
        elif job["state"] == JobState.ready:
            job_state = (t.green, "ready")
        else:
            job_state = str(job["state"])

        # Colourise power states
        power_state: TableColumn
        if job["power"] is not None:
            power_state = (t.green, "on") if job["power"] else (t.red, "off")
            if job["state"] == JobState.power:
                power_state = (t.yellow, power_state[1])
        else:
            power_state = ""

        num_boards: Union[int, str]
        if isinstance(job["boards"],  Sized):
            num_boards = len(job["boards"])
        else:
            num_boards = ""
        # Format start time
        timestamp = render_timestamp(cast(float, job["start_time"]))

        if job["allocated_machine_name"] is not None:
            machine_name = str(job["allocated_machine_name"])
        else:
            machine_name = ""

        owner = str(job["owner"])
        if "keepalivehost" in job and job["keepalivehost"] is not None:
            owner += f" ({job['keepalivehost']})"

        table.append((
            cast(int, job["job_id"]),
            job_state,
            power_state,
            num_boards,
            machine_name,
            timestamp,
            str(job["keepalive"]),
            owner,
        ))
    return render_table(table)


class ProcessListScript(Script):
    """
    An object form Job scripts.
    """
    @overrides(Script.get_parser)
    def get_parser(self, cfg: SpallocConfig) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="List all active jobs.")
        parser.add_argument(
            "--version", "-V", action="version", version=__version__)
        parser.add_argument(
            "--watch", "-w", action="store_true", default=False,
            help="watch the list of live jobs in real time")
        filter_args = parser.add_argument_group("filtering arguments")
        filter_args.add_argument(
            "--machine", "-m", help="list only jobs on the specified machine")
        filter_args.add_argument(
            "--owner", "-o",
            help="list only jobs belonging to a particular owner")
        return parser

    def one_shot(self, client: ProtocolClient,
                 args: argparse.Namespace) -> None:
        """ Gets info on the job list once. """
        t = Terminal(stream=sys.stderr)
        jobs = client.list_jobs(timeout=args.timeout)
        print(render_job_list(t, jobs, args))

    def recurring(
            self, client: ProtocolClient, args: argparse.Namespace) -> None:
        """ Repeatedly gets info on the job list. """
        client.notify_job(timeout=args.timeout)
        t = Terminal(stream=sys.stderr)
        while True:
            jobs = client.list_jobs(timeout=args.timeout)
            # Clear the screen before reprinting the table
            sys.stdout.write(t.clear_screen())
            print(render_job_list(t, jobs, args))
            # Wait for state change
            try:
                client.wait_for_notification()
            except KeyboardInterrupt:
                # Gracefully exit
                return
            finally:
                print("")

    @overrides(Script.body)
    def body(self, client: ProtocolClient, args: argparse.Namespace) -> int:
        if args.watch:
            self.recurring(client, args)
        else:
            self.one_shot(client, args)
        return 0

    def verify_arguments(self, args: argparse.Namespace) -> None:
        pass


main = ProcessListScript()
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
