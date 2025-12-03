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

""" Command-line administrative machine management interface.

When called with no arguments the ``spalloc-machine`` command lists all
available machines and a summary of their current load.

If a specific machine is given as an argument, the current allocation of jobs
to machines is displayed:

.. image:: _static/spalloc_machine.png
    :alt: spalloc-machine showing jobs allocated on a machine.

Adding the ``--detailed`` option displays additional information about jobs
running on a machine.

If the ``--watch`` option is given, the information displayed is updated in
real-time.
"""
from collections import defaultdict
import argparse
import sys
from typing import Any, cast, Dict, Iterator, List, Optional

from spinn_utilities.overrides import overrides
from spinn_utilities.typing.json import JsonObject, JsonObjectArray

from spalloc_client import __version__, ProtocolClient
from spalloc_client.spalloc_config import SpallocConfig
from spalloc_client.term import (
    Terminal, render_table, render_definitions, render_boards, render_cells,
    DEFAULT_BOARD_EDGES, TableRow, TableType)
from spalloc_client.scripts.support import Terminate, Script


def generate_keys(alphabet: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ") -> Iterator:
    """ Generate ascending values in spreadsheet-column-name style.

    For example, A, B, C, ..., Y, Z, AA, AB, AC...

    :returns:
        Iterator yielding ascending values in spreadsheet-column-name style.
    """
    for symbol in alphabet:
        yield symbol

    for prefix in generate_keys(alphabet):
        for symbol in alphabet:
            yield prefix + symbol


def list_machines(t: Terminal, machines: JsonObjectArray,
                  jobs: JsonObjectArray) -> None:
    """ Display a table summarising the available machines and their load.

    Parameters
    ----------
    t :
        An output styling object for stdout.
    machines :
        The list of machines and their properties returned from the server.
    jobs :
        The list of jobs and their properties returned from the server.

    Returns
    -------
        An error code: 0 on success.
    """
    machine_jobs = defaultdict(list)
    for job in jobs:
        machine_jobs[job["allocated_machine_name"]].append(job)

    table: TableType = [[
        (t.underscore_bright, "Name"),
        (t.underscore_bright, "Num boards"),
        (t.underscore_bright, "In-use"),
        (t.underscore_bright, "Jobs"),
        (t.underscore_bright, "Tags"),
    ]]

    for machine in machines:
        name = cast(str, machine["name"])
        boards = (((cast(int, machine["width"])) *
                   cast(int, machine["height"]) * 3) -
                  len(cast(list, machine["dead_boards"])))
        in_use = sum(len(cast(list, job["boards"]))
                     for job in cast(dict, machine_jobs[machine["name"]]))
        the_jobs = len(machine_jobs[machine["name"]])
        tags = ", ".join(cast(list, machine["tags"]))
        table.append([
            name, boards, in_use, the_jobs, tags])

    print(render_table(table))


def _get_machine(machines: JsonObjectArray, machine_name: str) -> JsonObject:
    for machine in machines:
        if machine["name"] == machine_name:
            return machine
    # No matching machine
    raise Terminate(6, f"No machine '{machine_name}' was found")


def show_machine(t: Terminal, machines: JsonObjectArray, jobs: JsonObjectArray,
                 machine_name: str, compact: bool = False) -> None:
    """ Display a more detailed overview of an individual machine.

    Parameters
    ----------
    t :
        An output styling object for stdout.
    machines :
        The list of machines and their properties returned from the server.
    jobs :
        The list of jobs and their properties returned from the server.
    machine_name :
        The machine of interest.
    compact :
        If True, display the listing of jobs on the machine in a more compact
        format.

    Returns
    -------
        An error code: 0 on success.
    """
    # pylint: disable=too-many-locals

    # Find the machine requested
    machine = _get_machine(machines, machine_name)

    # Extract list of jobs running on the machine
    displayed_jobs: List[Dict[str, Any]] = []
    job_key_generator = iter(generate_keys())
    job_colours = [
        t.green, t.blue, t.magenta, t.yellow, t.cyan,
        t.dim_green, t.dim_blue, t.dim_magenta, t.dim_yellow, t.dim_cyan,
        t.bright_green, t.bright_blue, t.bright_magenta, t.bright_yellow,
        t.bright_cyan,
    ]
    for job in jobs:
        if job["allocated_machine_name"] == machine_name:
            displayed_jobs.append(job)
            job["key"] = next(job_key_generator)

    # Calculate machine stats
    num_boards = ((cast(int, machine["width"]) *
                   cast(int, machine["height"]) * 3) -
                  len(cast(list, machine["dead_boards"])))
    num_in_use = sum(map(len, (cast(list, job["boards"])
                               for job in displayed_jobs)))

    # Show general machine information
    info = dict()
    info["Name"] = machine["name"]
    info["Tags"] = ", ".join(cast(list, machine["tags"]))
    info["In-use"] = f"{num_in_use} of {num_boards}"
    info["Jobs"] = len(displayed_jobs)
    print(render_definitions(info))

    # Draw diagram of machine
    dead_boards = set((x, y, z) for x, y, z in cast(
        list, machine["dead_boards"]))
    board_groups = [(list([(x, y, z)
                          for x in range(cast(int, machine["width"]))
                          for y in range(cast(int, machine["height"]))
                          for z in range(3)
                          if (x, y, z) not in dead_boards]),
                     t.dim(" . "),  # Label
                     tuple(map(t.dim, DEFAULT_BOARD_EDGES)),  # Inner
                     tuple(map(t.dim, DEFAULT_BOARD_EDGES)))]  # Outer
    for job in displayed_jobs:
        boards_list = job["boards"]
        assert isinstance(boards_list, list)
        boards = []
        for board in boards_list:
            assert isinstance(board, list)
            (x, y, z) = board
            boards.append((cast(int, x), cast(int, y), cast(int, z)))
        colour_func = job_colours[
                cast(int, job["job_id"]) % len(job_colours)]
        board_groups.append((
            boards,
            colour_func(cast(str, job["key"]).center(3)),  # Label
            tuple(map(colour_func, DEFAULT_BOARD_EDGES)),  # Inner
            tuple(map(t.bright, DEFAULT_BOARD_EDGES))  # Outer
        ))
    print("")
    print(render_boards(board_groups, cast(list, machine["dead_links"]),
                        tuple(map(t.red, DEFAULT_BOARD_EDGES))))
    # Produce table showing jobs on machine
    if compact:
        # In compact mode, produce column-aligned cells
        cells = []
        for job in displayed_jobs:
            key = cast(str, job["key"])
            job_id = str(job["job_id"])
            colour_func = job_colours[
                cast(int, job["job_id"]) % len(job_colours)]
            cells.append((len(key) + len(job_id) + 1,
                         f"{colour_func(key)}:{job_id}"))
        print("")
        print(render_cells(cells))
    else:
        # In non-compact mode, produce a full table of job information
        job_table: TableType = [[
            (t.underscore_bright, "Key"),
            (t.underscore_bright, "Job ID"),
            (t.underscore_bright, "Num boards"),
            (t.underscore_bright, "Owner (Host)"),
        ]]
        for job in displayed_jobs:
            owner = str(job["owner"])
            if "keepalivehost" in job and job["keepalivehost"] is not None:
                owner += f" {job['keepalivehost']}"
            colour_func = job_colours[
                cast(int, job["job_id"]) % len(job_colours)]
            table_row: TableRow = [
                (colour_func, cast(str, job["key"])),
                cast(int, job["job_id"]),
                len(cast(list, job["boards"])),
                owner,
            ]
            job_table.append(table_row)
        print("")
        print(render_table(job_table))


class ListMachinesScript(Script):
    """
    A Script object to get information from a spalloc machine.
    """

    def __init__(self) -> None:
        super().__init__()
        self.parser: Optional[argparse.ArgumentParser] = None

    def get_and_display_machine_info(
            self, client: ProtocolClient,
            args: argparse.Namespace, t: Terminal) -> None:
        """ Gets and displays info for the machine(s) """
        # Get all information
        machines = client.list_machines(timeout=args.timeout)
        jobs = client.list_jobs(timeout=args.timeout)

        # Display accordingly
        if args.machine is None:
            list_machines(t, machines, jobs)
        else:
            show_machine(t, machines, jobs, args.machine, not args.detailed)

    @overrides(Script.get_parser)
    def get_parser(self, cfg: SpallocConfig) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Get the state of individual machines.")
        parser.add_argument(
            "--version", "-V", action="version", version=__version__)
        parser.add_argument(
            "machine", nargs="?",
            help="if given, specifies the machine to inspect")
        parser.add_argument(
            "--watch", "-w", action="store_true", default=False,
            help="update the output when things change.")
        parser.add_argument(
            "--detailed", "-d", action="store_true", default=False,
            help="list detailed job information")
        self.parser = parser
        return parser

    @overrides(Script.verify_arguments)
    def verify_arguments(self, args: argparse.Namespace) -> None:
        # Fail if --detailed used without specifying machine
        if args.machine is None and args.detailed:
            assert self.parser is not None
            self.parser.error(
                "--detailed only works when a specific machine is specified")

    def one_shot(self,  client: ProtocolClient,
                 args: argparse.Namespace) -> None:
        """
        Display the machine info once
        """
        t = Terminal()
        # Get all information and display accordingly
        self.get_and_display_machine_info(client, args, t)

    def recurring(self, client: ProtocolClient,
                  args: argparse.Namespace) -> None:
        """
        Repeatedly display the machine info
        """
        t = Terminal()
        while True:
            client.notify_machine(args.machine, timeout=args.timeout)
            t.stream.write(t.clear_screen())
            # Prevent errors on stderr being cleared away due to clear being
            # buffered
            t.stream.flush()

            # Get all information and display accordingly
            self.get_and_display_machine_info(client, args, t)

            # Wait for changes
            try:
                client.wait_for_notification()
            except KeyboardInterrupt:
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


main = ListMachinesScript()
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
