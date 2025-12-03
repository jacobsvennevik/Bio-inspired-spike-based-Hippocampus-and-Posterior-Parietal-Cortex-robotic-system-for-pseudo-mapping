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

""" Command-line tool to find out where a particular chip or board resides.

The ``spalloc-where-is`` command allows you to query boards by coordinate, by
physical location, by chip or by job. In response to a query, a standard set of
information is displayed as shown in the example below::

    $ spalloc-where-is --job-chip 24 14, 3
                     Machine: my-machine
           Physical Location: Cabinet 2, Frame 4, Board 7
            Board Coordinate: (3, 4, 0)
    Machine Chip Coordinates: (38, 51)
    Coordinates within board: (2, 3)
             Job using board: 24
      Coordinates within job: (14, 3)

In this example we ask, 'where is chip (14, 3) in job 24'? We discover that:

* The chip is the machine named 'my-machine' on the board in cabinet 2, frame
  4, board 7.
* This board's logical board coordinates are (3, 4, 0). These logical
  coordinates may be used to specifically request this board from Spalloc in
  the future.
* If 'my-machine' were booted as a single large machine, the chip we queried
  would be chip (38, 51). This may be useful for cross-referencing with
  diagrams produced by SpiNNer_.
* The chip in question is chip (2, 3) its board. This may be useful when
  reporting faulty chips/replacing boards..
* The job currently running on the board has ID 24. Obviously in this example
  we already knew this but this may be useful when querying by board.
* Finally, we're told that the queried chip has the coordinates (14, 3) in the
  machine allocated to job 24. Again, this information may be more useful when
  querying by board.

.. _SpiNNer: https://github.com/SpiNNakerManchester/SpiNNer

To query by logical board coordinate::

    spalloc-where-is --board MACHINE X Y Z

To query by physical board location::

    spalloc-where-is --physical MACHINE CABINET FRAME BOARD

To query by chip coordinate (as if the machine were booted as one large
machine)::

    spalloc-where-is --chip MACHINE X Y

To query by chip coordinate of chips allocated to a job::

    spalloc-where-is --job-chip JOB_ID X Y
"""
import argparse
import sys
from typing import Any, cast, Dict, Optional

from spinn_utilities.overrides import overrides

from spalloc_client import __version__, ProtocolClient
from spalloc_client.term import render_definitions
from spalloc_client.scripts.support import Terminate, Script
from spalloc_client.spalloc_config import SpallocConfig


class WhereIsScript(Script):
    """
    An script object to find where a board is
    """

    def __init__(self) -> None:
        super().__init__()
        self.parser: Optional[argparse.ArgumentParser] = None
        self.where_is_kwargs: Optional[dict] = None
        self.show_board_chip = False

    @overrides(Script.get_parser)
    def get_parser(self, cfg: SpallocConfig) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Find out the location (physical or logical) of a "
                        "chip or board.")
        parser.add_argument(
            "--version", "-V", action="version", version=__version__)
        control_args = parser.add_mutually_exclusive_group(required=True)
        control_args.add_argument(
            "--board", "-b", "--logical", "-l", nargs=4,
            metavar=("MACHINE", "X", "Y", "Z"),
            help="specify the logical board coordinate")
        control_args.add_argument(
            "--physical", "-p", nargs=4,
            metavar=("MACHINE", "CABINET", "FRAME", "BOARD"),
            help="specify a board's physical location")
        control_args.add_argument(
            "--chip", "-c", nargs=3, metavar=("MACHINE", "X", "Y"),
            help="specify a board by chip coordinates (as if the whole "
            "machine is being used)")
        control_args.add_argument(
            "--job-chip", "-j", nargs=3, metavar=("JOB_ID", "X", "Y"),
            help="specify the chip coordinates of a chip within a job's "
            "boards")
        self.parser = parser
        return parser

    @overrides(Script.verify_arguments)
    def verify_arguments(self, args: argparse.Namespace) -> None:
        try:
            if args.board:
                machine, x, y, z = args.board
                self.where_is_kwargs = {
                    "machine": machine,
                    "x": int(x),
                    "y": int(y),
                    "z": int(z),
                }
                self.show_board_chip = False
            elif args.physical:
                machine, c, f, b = args.physical
                self.where_is_kwargs = {
                    "machine": machine,
                    "cabinet": int(c),
                    "frame": int(f),
                    "board": int(b),
                }
                self.show_board_chip = False
            elif args.chip:
                machine, x, y = args.chip
                self.where_is_kwargs = {
                    "machine": machine,
                    "chip_x": int(x),
                    "chip_y": int(y),
                }
                self.show_board_chip = True
            elif args.job_chip:
                job_id, x, y = args.job_chip
                self.where_is_kwargs = {
                    "job_id": int(job_id),
                    "chip_x": int(x),
                    "chip_y": int(y),
                }
                self.show_board_chip = True
        except ValueError as e:
            assert self.parser is not None
            self.parser.error(f"Error: {e}")

    @overrides(Script.body)
    def body(self, client: ProtocolClient, args: argparse.Namespace) -> int:
        # Ask the server
        assert self.where_is_kwargs is not None
        location = client.where_is(**self.where_is_kwargs)
        if location is None:
            raise Terminate(4, "No boards at the specified location")

        out: Dict[str, Any] = dict()
        out["Machine"] = cast(str, location["machine"])
        cabinet, frame, board = cast(list, location["physical"])
        out["Physical location"] = (
            f"Cabinet {cabinet}, Frame {frame}, Board {board}")
        out["Board coordinate"] = tuple(cast(list, location["logical"]))
        out["Machine chip coordinates"] = tuple(cast(list, location["chip"]))
        if self.show_board_chip:
            out["Coordinates within board"] = tuple(
                cast(list, location["board_chip"]))
        out["Job using board"] = location["job_id"]
        if location["job_id"]:
            out["Coordinates within job"] = tuple(
                cast(list, location["job_chip"]))
        print(render_definitions(out))
        return 0


main = WhereIsScript()
if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
