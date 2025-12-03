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

""" A command-line utility for creating jobs.

.. note::

    In the examples below, it is assumed that the spalloc server hostname and a
    suitable owner name have been specified in a :py:mod:`~spalloc.config`
    file.


Basic usage
-----------

By default, the ``spalloc`` command allocates a job according to the
command-line specification and then waits for boards to be allocated and
powered on.

.. note::

    By default, allocated machines are powered on but not booted. If Rig_ is
    installed, ``spalloc`` provides a ``--boot`` option which also boots the
    allocated machine once it has been powered on.

.. image:: _static/spalloc.gif
    :alt: Animated GIF showing the typical execution of a spalloc call.

The ``spalloc`` command can be called in one of the following styles though
most users will probably only require the first two.

=====================  ====================================================
Invocation             Allocation
=====================  ====================================================
``spalloc``            A single SpiNN-5 board
``spalloc 5``          A machine with *at least* 5 boards
``spalloc 4 2``        A 4x2 *triad* machine.
``spalloc 3 4 0``      A single SpiNN-5 board at logical position (3, 4, 0)
=====================  ====================================================

.. _Rig: https://github.com/project-rig/rig

A range of additional command-line arguments are available to control various
aspects of Job allocation, run ``spalloc --help`` for a complete listing.


Wrapping other commands
-----------------------

The ``spalloc`` command can alternatively wrap an existing command, calling it
once a SpiNNaker machine is allocated and cleaning up the job when the command
exits::

    $ spalloc 24 -c "rig-boot {} {w} {h} && python my_app.py {}"

The example above attempts to allocate a 24-board machine and, once allocated
and powered on, calls the command above, with the arguments in curly braces
substituted for details of the allocated machine.

The following substitutions are available:

==================  =================================
Token               Substitution
==================  =================================
``{}``              Chip 0, 0's hostname
``{hostname}``      Chip 0, 0's hostname
``{w}``             Width of the system (in chips)
``{width}``         Width of the system (in chips)
``{h}``             Height of the system (in chips)
``{height}``        Height of the system (in chips)
``{ethernet_ips}``  Filename of a CSV of Ethernet IPs
``{id}``            The job ID
==================  =================================

Ethernet-connected chip hostname CSV Format
-------------------------------------------

Hostnames for all Ethernet-connected SpiNNaker chips in a machine are provided
in a CSV file with three columns:: x, y and hostname. The CSV file is newline
(``\\n``) delimited and the first row is a header.

Disconnecting and resuming jobs
-------------------------------

.. warning::

    This functionality is intended for advanced users only.

By default, when the ``spalloc`` command exits, the job will be destroyed and
any allocated boards freed. This behaviour can be disabled with the
``--no-destroy`` argument, leaving the job allocated after the command exits.

Such a job may be 'resumed' by calling ``spalloc`` with the ``--resume
[JOB_ID]`` option.

Note that by default, jobs require a 'keepalive' message to be sent to the
server at a regular interval. While the ``spalloc`` command is running, these
messages are sent automatically but after exiting the commands are no longer
sent. Adding the ``--keepalive -1`` option when creating a job disables this.
"""
import argparse
from collections import OrderedDict
import logging
import os
import subprocess
import sys
import tempfile
from six import iteritems
from six.moves import input, shlex_quote  # @UnresolvedImport
from spalloc import (
    config, Job, JobState, __version__, ProtocolError, ProtocolTimeoutError,
    SpallocServerException)
from spalloc.term import Terminal, render_definitions
# Rig is used to implement the optional '--boot' facility.
try:
    from rig.machine_control import MachineController
except ImportError:  # pragma: no cover
    MachineController = None

arguments = None
t = None


def write_ips_to_csv(connections, ip_file_filename):
    """ Write the supplied IP addresses to a CSV file.

    The produced CSV has three columns: x, y and hostname where x and y give
    chip coordinates of Ethernet-connected chips and hostname gives the IP
    address for that chip.

    Parameters
    ----------
    connections : {(x, y): hostname, ...}
    ip_file_filename : str
    """
    with open(ip_file_filename, "w") as f:
        f.write("x,y,hostname\n")
        f.write("".join("{},{},{}\n".format(x, y, hostname)
                        for (x, y), hostname
                        in sorted(iteritems(connections))))


def print_info(machine_name, connections, width, height, ip_file_filename):
    """ Print the current machine info in a human-readable form and wait for
    the user to press enter.

    Parameters
    ----------
    machine_name : str
        The machine the job is running on.
    connections : {(x, y): hostname, ...}
        The connections to the boards.
    width, height : int
        The width and height of the machine in chips.
    ip_file_filename : str
    """
    t_stdout = Terminal()

    to_print = OrderedDict()

    to_print["Hostname"] = t_stdout.bright(connections[(0, 0)])
    to_print["Width"] = width
    to_print["Height"] = height

    if len(connections) > 1:
        to_print["Num boards"] = len(connections)
        to_print["All hostnames"] = ip_file_filename

    to_print["Running on"] = machine_name

    print(render_definitions(to_print))

    try:
        input(t_stdout.dim("<Press enter when done>"))
    except (KeyboardInterrupt, EOFError):
        print("")


def run_command(command, job_id, machine_name, connections, width, height,
                ip_file_filename):
    """ Run a user-specified command, substituting arguments for values taken
    from the allocated board.

    Arguments may include the following substitution tokens:

    ==================  =================================
    Token               Substitution
    ==================  =================================
    ``{}``              Chip 0, 0's hostname
    ``{hostname}``      Chip 0, 0's hostname
    ``{w}``             Width of the system (in chips)
    ``{width}``         Width of the system (in chips)
    ``{h}``             Height of the system (in chips)
    ``{height}``        Height of the system (in chips)
    ``{ethernet_ips}``  Filename of a CSV of Ethernet IPs
    ``{id}``            The job ID
    ==================  =================================

    Parameters
    ----------
    command : [command, arg, ...]
    machine_name : str
        The machine the job is running on.
    connections : {(x, y): hostname, ...}
        The connections to the boards.
    width, height : int
        The width and height of the machine in chips.
    ip_file_filename : str

    Returns
    -------
    int
        The return code of the supplied command.
    """
    # pylint: disable=too-many-arguments
    root_hostname = connections[(0, 0)]

    # Print essential info in log
    logging.info("Allocated %d x %d chip machine in '%s'",
                 width, height, machine_name)
    logging.info("Chip (0, 0) IP: %s", root_hostname)
    logging.info("All board IPs listed in: %s", ip_file_filename)

    # Make substitutions in command arguments
    command = [arg.format(root_hostname,
                          hostname=root_hostname,
                          w=width,
                          width=width,
                          h=height,
                          height=height,
                          ethernet_ips=ip_file_filename,
                          id=job_id)
               for arg in command]

    # NB: When using shell=True, commands should be given as a string rather
    # than the usual list of arguments.
    command = " ".join(map(shlex_quote, command))
    p = subprocess.Popen(command, shell=True)

    # Pass through keyboard interrupts
    while True:
        try:
            return p.wait()
        except KeyboardInterrupt:
            p.terminate()


def info(msg):
    if not arguments.quiet:
        t.stream.write("{}\n".format(msg))


def update(msg, colour, *args):
    info(t.update(colour(msg.format(*args))))


def wait_for_job_ready(job):
    # Wait for it to become ready, keeping the user informed along the
    # way
    old_state = None
    cur_state = job.state
    try:
        while True:
            # Show debug info on state-change
            if old_state != cur_state:
                if cur_state == JobState.queued:
                    update("Job {}: Waiting in queue...", t.yellow,
                           job.id)
                elif cur_state == JobState.power:
                    update("Job {}: Waiting for power on...", t.yellow,
                           job.id)
                elif cur_state == JobState.ready:
                    # Here we go!
                    return 0, None
                elif cur_state == JobState.destroyed:
                    # Exit with error state
                    reason = None
                    try:
                        reason = job.reason
                    except (IOError, OSError):
                        pass

                    if reason is not None:
                        update("Job {}: Destroyed: {}", t.red,
                               job.id, reason)
                    else:
                        update("Job {}: Destroyed.", t.red,
                               job.id)
                    return 1, reason
                elif cur_state == JobState.unknown:
                    update("Job {}: Job not recognised by server.", t.red,
                           job.id)
                    return 2, None
                else:
                    update("Job {}: Entered an unrecognised state {}.",
                           t.red, job.id, cur_state)
                    return 3, None

            old_state, cur_state = \
                cur_state, job.wait_for_state_change(cur_state)
    except KeyboardInterrupt:
        # Gracefully terminate from keyboard interrupt
        update("Job {}: Keyboard interrupt.", t.red, job.id)
        return 4, "Keyboard interrupt."


def parse_argv(argv):
    cfg = config.read_config()

    parser = argparse.ArgumentParser(
        description="Request (and allocate) a SpiNNaker machine.")
    parser.add_argument("--version", "-V", action="version",
                        version=__version__)
    parser.add_argument("--quiet", "-q", action="store_true",
                        default=False,
                        help="suppress informational messages")
    parser.add_argument("--debug", action="store_true",
                        default=False,
                        help="enable additional diagnostic information")
    parser.add_argument("--no-destroy", "-D", action="store_true",
                        default=False,
                        help="do not destroy the job on exit")
    if MachineController is not None:
        parser.add_argument("--boot", "-B", action="store_true",
                            default=False,
                            help="boot the machine once powered on")

    allocation_args = parser.add_argument_group(
        "allocation requirement arguments")
    allocation_args.add_argument(
        "what", nargs="*", default=[], type=int, metavar="WHAT",
        help="what to allocate: nothing or 1 requests 1 SpiNN-5 board, NUM "
        "requests at least NUM SpiNN-5 boards, WIDTH HEIGHT means "
        "WIDTHxHEIGHT triads of SpiNN-5 boards and X Y Z requests a "
        "board the specified logical board coordinate.")
    allocation_args.add_argument(
        "--resume", "-r", type=int,
        help="if given, resume keeping the specified job alive rather than "
        "creating a new job (all allocation requirements will be ignored)")
    allocation_args.add_argument(
        "--machine", "-m", nargs="?", default=cfg["machine"],
        help="only allocate boards which are part of a specific machine, or "
        "any machine if no machine is given (default: %(default)s)")
    allocation_args.add_argument(
        "--tags", "-t", nargs="*", metavar="TAG",
        default=cfg["tags"] or ["default"],
        help="only allocate boards which have (at least) the specified flags "
        "(default: {})".format(" ".join(cfg["tags"] or [])))
    allocation_args.add_argument(
        "--min-ratio", type=float, metavar="RATIO", default=cfg["min_ratio"],
        help="when allocating by number of boards, require that the "
        "allocation be at least as square as this ratio (default: "
        "%(default)s)")
    allocation_args.add_argument(
        "--max-dead-boards", type=int, metavar="NUM", default=(
            -1 if cfg["max_dead_boards"] is None else cfg["max_dead_boards"]),
        help="boards allowed to be dead in the allocation, or -1 to allow "
        "any number of dead boards (default: %(default)s)")
    allocation_args.add_argument(
        "--max-dead-links", type=int, metavar="NUM", default=(
            -1 if cfg["max_dead_links"] is None else cfg["max_dead_links"]),
        help="inter-board links allowed to be dead in the allocation, or -1 "
        "to allow any number of dead links (default: %(default)s)")
    allocation_args.add_argument(
        "--require-torus", "-w", action="store_true",
        default=cfg["require_torus"],
        help="require that the allocation contain torus (a.k.a. wrap-around) "
        "links {}".format("(default)" if cfg["require_torus"] else ""))
    allocation_args.add_argument(
        "--no-require-torus", "-W", action="store_false", dest="require_torus",
        help="do not require that the allocation contain torus (a.k.a. "
        "wrap-around) links {}".format(
            "" if cfg["require_torus"] else "(default)"))

    command_args = parser.add_argument_group("command wrapping arguments")
    command_args.add_argument(
        "--command", "-c", nargs=argparse.REMAINDER,
        help="execute the specified command once boards have been allocated "
        "and deallocate the boards when the application exits ({} and "
        "{hostname} are substituted for the chip chip at (0, 0)'s hostname, "
        "{w} and {h} give the dimensions of the SpiNNaker machine in chips, "
        "{ethernet_ips} is a temporary file containing a CSV with three "
        "columns: x, y and hostname giving the hostname of each Ethernet "
        "connected SpiNNaker chip)")

    server_args = parser.add_argument_group("spalloc server arguments")
    server_args.add_argument(
        "--owner", default=cfg["owner"],
        help="by convention, the email address of the owner of the job "
        "(default: %(default)s)")
    server_args.add_argument(
        "--hostname", "-H", default=cfg["hostname"],
        help="hostname or IP of the spalloc server (default: %(default)s)")
    server_args.add_argument(
        "--port", "-P", default=cfg["port"], type=int,
        help="port number of the spalloc server (default: %(default)s)")
    server_args.add_argument(
        "--keepalive", type=int, metavar="SECONDS",
        default=(-1 if cfg["keepalive"] is None else cfg["keepalive"]),
        help="the interval at which to require keepalive messages to be "
        "sent to prevent the server cancelling the job, or -1 to not "
        "require keepalive messages (default: %(default)s)")
    server_args.add_argument(
        "--reconnect-delay", default=cfg["reconnect_delay"], type=float,
        metavar="SECONDS",
        help="seconds to wait before reconnecting to the server if the "
        "connection is lost (default: %(default)s)")
    server_args.add_argument(
        "--timeout", default=cfg["timeout"], type=float, metavar="SECONDS",
        help="seconds to wait for a response from the server "
        "(default: %(default)s)")
    return parser, parser.parse_args(argv)


def run_job(job_args, job_kwargs, ip_file_filename):
    # Reason for destroying the job
    reason = None

    # Create the job
    try:
        job = Job(*job_args, **job_kwargs)
    except (OSError, IOError, ProtocolError, ProtocolTimeoutError) as e:
        info(t.red("Could not connect to server: {}".format(e)))
        return 6

    try:
        # Wait for it to become ready, keeping the user informed along the
        # way
        code, reason = wait_for_job_ready(job)
        if code != 0:
            return code

        # Machine is now ready
        write_ips_to_csv(job.connections, ip_file_filename)

        # Boot the machine if required
        if MachineController is not None and arguments.boot:
            update("Job {}: Booting...", t.yellow, job.id)
            mc = MachineController(job.hostname)
            mc.boot(job.width, job.height)

        update("Job {}: Ready!", t.green, job.id)

        # Either run the user's application or just print the details.
        if not arguments.command:
            print_info(job.machine_name, job.connections,
                       job.width, job.height, ip_file_filename)
            return 0
        return run_command(arguments.command, job.id, job.machine_name,
                           job.connections, job.width, job.height,
                           ip_file_filename)
    finally:
        # Destroy job and disconnect client
        if arguments.no_destroy:
            job.close()
        else:
            job.destroy(reason)


def _minzero(value):
    return value if value >= 0.0 else None


def main(argv=None):
    global arguments, t
    parser, arguments = parse_argv(argv)
    t = Terminal(stream=sys.stderr)

    # Fail if no owner is defined (unless resuming)
    if not arguments.owner and arguments.resume is None:
        parser.error(
            "--owner must be specified (typically your email address)")

    # Fail if server not specified
    if arguments.hostname is None:
        parser.error("--hostname of spalloc server must be specified")

    # Set universal job arguments
    job_kwargs = {
        "hostname": arguments.hostname,
        "port": arguments.port,
        "reconnect_delay": _minzero(arguments.reconnect_delay),
        "timeout": _minzero(arguments.timeout),
    }

    if arguments.resume:
        job_args = []
        job_kwargs.update({
            "resume_job_id": arguments.resume,
        })
    else:
        # Make sure 'what' takes the right form
        if len(arguments.what) not in (0, 1, 2, 3):
            parser.error(
                "expected either no arguments, one argument, NUM, two "
                "arguments, WIDTH HEIGHT, or three arguments, X Y Z")

        # Unpack arguments for the job and server
        job_args = arguments.what
        job_kwargs.update({
            "owner": arguments.owner,
            "keepalive": _minzero(arguments.keepalive),
            "machine": arguments.machine,
            "tags": arguments.tags if arguments.machine is None else None,
            "min_ratio": arguments.min_ratio,
            "max_dead_boards": _minzero(arguments.max_dead_boards),
            "max_dead_links": _minzero(arguments.max_dead_links),
            "require_torus": arguments.require_torus,
        })

    # Set debug level
    if arguments.debug:
        logging.basicConfig(level=logging.DEBUG)

    # Create temporary file in which to write CSV of all board IPs
    _, ip_file_filename = tempfile.mkstemp(".csv", "spinnaker_ips_")

    try:
        return run_job(job_args, job_kwargs, ip_file_filename)
    except SpallocServerException as e:  # pragma: no cover
        info(t.red("Error from server: {}".format(e)))
        return 6
    finally:
        # Delete IP address list file
        os.remove(ip_file_filename)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
