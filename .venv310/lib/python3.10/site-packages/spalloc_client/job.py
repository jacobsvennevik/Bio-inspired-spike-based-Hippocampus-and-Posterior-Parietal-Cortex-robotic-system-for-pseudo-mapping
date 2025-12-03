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

# A high-level Python interface for allocating SpiNNaker boards.

from collections import namedtuple
import logging
import subprocess
import time
from types import TracebackType
from typing import (cast, Dict, List, Optional, Tuple, Type, TypeVar, Union)
import sys

from typing_extensions import Literal, Self, TypeAlias

from spinn_utilities.typing.json import JsonArray

from spalloc_client.scripts.support import (
    VERSION_RANGE_START, VERSION_RANGE_STOP)

from .protocol_client import ProtocolClient, ProtocolTimeoutError
from .spalloc_config import SpallocConfig, SEARCH_PATH
from .states import JobState
from ._utils import time_left, time_left_float, timed_out, make_timeout

logger = logging.getLogger(__name__)

# pylint: disable=wrong-spelling-in-docstring
# In Python 2, no default handler exists for software which doesn't configure
# its own logging so we must add one ourselves as per
# https://docs.python.org/3.1/library/logging.html#configuring-logging-for-a-library
logger.addHandler(logging.StreamHandler())

F = TypeVar('F', bound='float')
_Int: TypeAlias = Union[int, None, Literal["USE_CONFIG"]]
_Float: TypeAlias = Union[float, None, Literal["USE_CONFIG"]]
_List: TypeAlias = Union[List[str], None, Literal["USE_CONFIG"]]
_Bool: TypeAlias = Union[bool, None, Literal["USE_CONFIG"]]


def pick_str(param: Optional[str], config: Optional[str]) -> Optional[str]:
    """ Use the param unless it is the default value, otherwise use config

    :returns: param unless it is "USE_CONFIG" when the config is returned
    """
    if param == "USE_CONFIG":
        return config
    return param


def pick_list(param: _List,
              config: Optional[List[str]]) -> Optional[List[str]]:
    """ Use the param unless it is the default value, otherwise use config

    :returns: param unless it is "USE_CONFIG" when the config is returned
    """
    if param == "USE_CONFIG":
        return config
    else:
        return param


def pick_num(param: Union[F, None, Literal["USE_CONFIG"]],
             config: Optional[F]) -> Optional[F]:
    """ Use the param unless it is the default value, otherwise use config

    :returns: param unless it is "USE_CONFIG" when the config is returned
    """
    if param == "USE_CONFIG":
        return config
    return param


def pick_bool(param: _Bool, config: Optional[bool]) -> Optional[bool]:
    """ Use the param if None or a bool, otherwise use config

    :returns: param unless it is "USE_CONFIG" when the config is returned
    """
    if param is None or isinstance(param, bool):
        return param
    else:
        return config


class Job(object):
    """ A high-level interface for requesting and managing allocations of
    SpiNNaker boards.

    Constructing a :py:class:`.Job` object connects to a `spalloc-server
    <https://github.com/SpiNNakerManchester/spalloc_server>`_ and requests a
    number of SpiNNaker boards. See the :py:meth:`constructor <.Job.__init__>`
    for details of the types of requests which may be made. The job object may
    then be used to monitor the state of the request, control the boards
    allocated and determine their IP addresses.

    In its simplest form, a :py:class:`.Job` can be used as a context manager
    like so::

        >>> from spalloc_client import Job
        >>> with Job(6) as j:
        ...     my_boot(j.hostname, j.width, j.height)
        ...     my_application(j.hostname)

    In this example a six-board machine is requested and the ``with`` context
    is entered once the allocation has been made and the allocated boards are
    fully powered on. When control leaves the block, the job is destroyed and
    the boards shut down by the server ready for another job.

    For more fine-grained control, the same functionality is available via
    various methods::

        >>> from spalloc_client import Job
        >>> j = Job(6)
        >>> j.wait_until_ready()
        >>> my_boot(j.hostname, j.width, j.height)
        >>> my_application(j.hostname)
        >>> j.destroy()

    .. note::

        More complex applications may wish to log the following attributes of
        their job to support later debugging efforts:

        * ``job.id`` -- May be used to query the state of the job and find out
          its fate if cancelled or destroyed. The ``spalloc-job`` command can
          be used to discover the state/fate of the job and
          ``spalloc-where-is`` may be used to find out what boards problem
          chips reside on.
        * ``job.machine_name`` and ``job.boards`` together give a complete
          record of the hardware used by the job. The ``spalloc-where-is``
          command may be used to find out the physical locations of the boards
          used.

    :py:class:`.Job` objects have the following attributes which describe the
    job and its allocated machines:

    Attributes
    ----------
    job.id : int or None
        The job ID allocated by the server to the job.
    job.state : :py:class:`.JobState`
        The current state of the job.
    job.power : bool or None
        If boards have been allocated to the job, are they on (True) or off
        (False). None if no boards are allocated to the job.
    job.reason : str or None
        If the job has been destroyed, gives the reason (which may be None), or
        None if the job has not been destroyed.
    job.hostname : str or None
        The hostname of the SpiNNaker chip at (0, 0), or None if no boards have
        been allocated to the job.
    job.connections : {(x, y): hostname, ...} or None
        The hostnames of all Ethernet-connected SpiNNaker chips, or None if no
        boards have been allocated to the job.
    job.width : int or None
        The width of the SpiNNaker network in chips, or None if no boards have
        been allocated to the job.
    job.height : int or None
        The height of the SpiNNaker network in chips, or None if no boards have
        been allocated to the job.
    job.machine_name : str or None
        The name of the machine the boards are allocated in, or None if not yet
        allocated.
    job.boards : [[x, y, z], ...] or None
        The logical coordinates allocated to the job, or None if not yet
        allocated.
    """

    def __init__(self, *args: int, hostname: Optional[str] = "USE_CONFIG",
                 port: _Int = "USE_CONFIG",
                 reconnect_delay: _Float = "USE_CONFIG",
                 timeout: _Float = "USE_CONFIG",
                 config_filenames: _List = "USE_CONFIG",
                 resume_job_id: Optional[int] = None,
                 owner: Optional[str] = "USE_CONFIG",
                 keepalive: _Float = "USE_CONFIG",
                 machine: Optional[str] = "USE_CONFIG",
                 tags: _List = "USE_CONFIG",
                 min_ratio: _Float = "USE_CONFIG",
                 max_dead_boards: _Int = "USE_CONFIG",
                 max_dead_links: _Int = "USE_CONFIG",
                 require_torus: _Bool = "USE_CONFIG"):
        """ Request a SpiNNaker machine.

        A :py:class:`.Job` is constructed in one of the following styles::

            >>> # Any single (SpiNN-5) board
            >>> Job()
            >>> Job(1)

            >>> # Any machine with at least 4 boards
            >>> Job(4)

            >>> # Any 7-or-more board machine with an aspect ratio at least as
            >>> # square as 1:2
            >>> Job(7, min_ratio=0.5)

            >>> # Any 4x5 triad segment of a machine (may or may-not be a
            >>> # torus/full machine)
            >>> Job(4, 5)

            >>> # Any torus-connected (full machine) 4x2 machine
            >>> Job(4, 2, require_torus=True)

            >>> # Board x=3, y=2, z=1 on the machine named "m"
            >>> Job(3, 2, 1, machine="m")

            >>> # Keep using (and keeping-alive) an existing allocation
            >>> Job(resume_job_id=123)

        Once finished with a Job, the :py:meth:`.destroy` (or in unusual
        applications :py:meth:`.Job.close`) method must be called to destroy
        the job, close the connection to the server and terminate the
        background keep-alive thread. Alternatively, a Job may be used as a
        context manager which automatically calls :py:meth:`.destroy` on
        exiting the block::

            >>> with Job() as j:
            ...     # ...for example...
            ...     my_boot(j.hostname, j.width, j.height)
            ...     my_application(j.hostname)

        The following keyword-only parameters below are used both to specify
        the server details as well as the job requirements. Most parameters
        default to the values supplied in the local
        :py:mod:`~spalloc_client.config`
        file allowing usage as in the examples above.

        Parameters
        ----------
        hostname :
            **Required.** The name of the spalloc server to connect to. (Read
            from config file if not specified.)
        port :
            The port number of the spalloc server to connect to. (Read from
            config file if not specified.)
        reconnect_delay :
            Number of seconds between attempts to reconnect to the server.
            (Read from config file if not specified.)
        timeout :
            Timeout for waiting for replies from the server. If None, will keep
            trying forever. (Read from config file if not specified.)
        config_filenames :
            If given must be a list of filenames to read configuration options
            from. If not supplied, the default config file locations are
            searched. Set to an empty list to prevent using values from config
            files.

        Other Parameters
        ----------------
        resume_job_id :
            If supplied, rather than creating a new job, take on an existing
            one, keeping it alive as required by the original job. If this
            argument is used, all other requirements are ignored.
        owner :
            **Required.** The name of the owner of the job. By convention this
            should be your email address. (Read from config file if not
            specified.)
        keepalive :
            The number of seconds after which the server may consider the job
            dead if this client cannot communicate with it. If None, no timeout
            will be used and the job will run until explicitly destroyed. Use
            with extreme caution. (Read from config file if not specified.)
        machine :
            Specify the name of a machine which this job must be executed on.
            If None, the first suitable machine available will be used,
            according to the tags selected below. Must be None when tags are
            given. (Read from config file if not specified.)
        tags :
            The set of tags which any machine running this job must have. If
            None is supplied, only machines with the "default" tag will be
            used. If machine is given, this argument must be None.  (Read from
            config file if not specified.)
        min_ratio :
            The aspect ratio (h/w) which the allocated region must be 'at least
            as square as'. Set to 0.0 for any allowable shape, 1.0 to be
            exactly square etc. Ignored when allocating single boards or
            specific rectangles of triads.
        max_dead_boards :
            The maximum number of broken or unreachable boards to allow in the
            allocated region. If None, any number of dead boards is permitted,
            as long as the board on the bottom-left corner is alive. (Read from
            config file if not specified.)
        max_dead_links :
            The maximum number of broken links allow in the allocated region.
            When require_torus is True this includes wrap-around links,
            otherwise peripheral links are not counted.  If None, any number of
            broken links is allowed. (Read from config file if not specified.).
        require_torus :
            If True, only allocate blocks with torus connectivity. In general
            this will only succeed for requests to allocate an entire machine.
            Must be False when allocating boards. (Read from config file if not
            specified.)
        """
        # Read configuration
        config_filenames = pick_list(config_filenames, SEARCH_PATH)
        config = SpallocConfig(config_filenames)

        # Get protocol client options
        hostname = pick_str(hostname, config.hostname)
        owner = pick_str(owner, config.owner)
        port = pick_num(port, config.port)
        reconnect_delay = pick_num(reconnect_delay, config.reconnect_delay)
        if reconnect_delay is None:
            raise ValueError("A reconnect_delay must be specified.")
        self._reconnect_delay = reconnect_delay
        self._timeout = pick_num(timeout, config.timeout)

        if hostname is None:
            raise ValueError("A hostname must be specified.")
        if port is None:
            raise ValueError("A port must be specified.")

        # Cached responses of _get_state and _get_machine_info
        self._last_machine_info: Optional["_JobMachineInfoTuple"] = None

        # Connection to server (and associated lock)
        self._client = ProtocolClient(hostname, port)

        # Check version compatibility (fail fast if can't communicate with
        # server)
        self._client.connect(timeout=self._timeout)
        self._assert_compatible_version()

        # Resume/create the job
        if resume_job_id:
            self.id = resume_job_id

            # If the job no longer exists, we can't get the keepalive interval
            # (and there's nothing to keepalive) so just bail out.
            job_state = self._get_state()
            if (job_state.state == JobState.unknown or
                    job_state.state == JobState.destroyed):
                if job_state.reason is not None:
                    reason = job_state.reason
                else:
                    reason = ""
                raise JobDestroyedError(
                    f"Job {resume_job_id} does not exist: "
                    f"{job_state.state.name}"
                    f"{': ' if job_state.reason is not None else ''}{reason}")

            # Snag the keepalive interval from the job
            self._keepalive = job_state.keepalive

            logger.info("Spalloc resumed job %d", self.id)
        else:
            # Get job creation arguments
            machine = pick_str(machine, config.machine)
            tags = pick_list(tags, config.tags)

            # Sanity check arguments
            if owner is None:
                raise ValueError("An owner must be specified.")
            if tags is not None and machine is not None:
                raise ValueError(
                    "Only one of tags and machine may be specified.")

            self._keepalive = pick_num(keepalive, config.keepalive)

            # Create the job (failing fast if can't communicate)
            self.id = self._client.create_job(
                self._timeout, *args, owner=owner,
                keepalive=self._keepalive, machine=machine, tags=tags,
                min_ratio=pick_num(min_ratio, config.min_ratio),
                max_dead_boards=pick_num(
                    max_dead_boards, config.max_dead_boards),
                max_dead_links=pick_num(
                    max_dead_links, config.max_dead_links),
                require_torus=pick_bool(
                    require_torus, config.require_torus))

            logger.info("Created spalloc job %d", self.id)

        # Set-up and start background keepalive thread
        self._keepalive_process = subprocess.Popen(
            [sys.executable, "-m", "spalloc_client._keepalive_process",
             str(hostname), str(port), str(self.id), str(self._keepalive),
             str(self._timeout), str(self._reconnect_delay)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        # Wait for it to announce that it is working
        stdout = self._keepalive_process.stdout
        assert stdout is not None
        while not stdout.closed:
            line = stdout.readline().decode("utf-8").strip()
            if line == "KEEPALIVE":
                break
            # Two special cases
            if "pydev debugger" in line or "thread" in line:
                continue
            if line:
                raise ValueError(f"Keepalive process wrote odd line: {line}")

    def __enter__(self) -> Self:
        """ Convenience context manager for common case where a new job is to
        be created and then destroyed once some code has executed.

        Waits for machine to be ready before the context enters and frees the
        allocation when the context exits.

        Example::

            >>> from spalloc_client import Job
            >>> with Job(6) as j:
            ...     my_boot(j.hostname, j.width, j.height)
            ...     my_application(j.hostname)
        """
        logger.info("Waiting for boards to become ready...")
        try:
            self.wait_until_ready()
            return self
        except Exception:
            self.destroy()
            raise

    def __exit__(self, exc_type: Optional[Type],
                 exc_value: Optional[BaseException],
                 exc_tb: Optional[TracebackType]) -> Literal[False]:
        self.destroy()
        return False

    def _assert_compatible_version(self) -> None:
        """ Assert that the server version is compatible.
        """
        v = self._client.version(timeout=self._timeout)
        v_ints = tuple(map(int, v.split(".")[:3]))

        if not (VERSION_RANGE_START <= v_ints < VERSION_RANGE_STOP):
            self._client.close()
            raise ValueError(
                f"Server version {v} is not compatible with this client.")

    def _reconnect(self) -> None:
        """ Reconnect to the server and check version.

        If reconnection fails, the error is reported as a warning but no
        exception is raised.
        """
        try:
            self._client.connect(self._timeout)
            self._assert_compatible_version()
            logger.info("Reconnected to spalloc server successfully.")
        except (IOError, OSError) as e:
            # Connect/version command failed... Leave the socket clearly
            # broken so that we retry again
            logger.warning(
                "Spalloc server is unreachable (%s), will keep trying...", e)
            self._client.close()

    def destroy(self, reason: Optional[str] = None) -> None:
        """ Destroy the job and disconnect from the server.

        Parameters
        ----------
        reason :
            *Optional.* Gives a human-readable explanation for the destruction
            of the job.
        """
        # Attempt to inform the server that the job was destroyed, fail
        # quietly on failure since the server will eventually time-out the job
        # itself.
        try:
            self._client.destroy_job(self.id, reason)
        except (IOError, OSError, ProtocolTimeoutError) as e:
            logger.warning("Could not destroy spalloc job: %s", e)

        self.close()

    def close(self) -> None:
        """ Disconnect from the server and stop keeping the job alive.

        .. warning::

            This method does not free the resources allocated by the job but
            rather simply disconnects from the server and ceases sending
            keep-alive messages. Most applications should use
            :py:meth:`.destroy` instead.
        """
        # Stop background thread
        if self._keepalive_process.poll() is None:
            self._keepalive_process.communicate(input="exit\n".encode("ascii"))
            self._keepalive_process.wait()

        # Disconnect
        self._client.close()

    def _get_state(self) -> "_JobStateTuple":
        """ Get the state of the job.

        Returns
        -------
        :py:class:`._JobStateTuple`
        """
        state = self._client.get_job_state(self.id, timeout=self._timeout)
        return _JobStateTuple(
            state=JobState(cast(int, state["state"])),
            power=state["power"],
            keepalive=state["keepalive"],
            reason=state["reason"])

    def set_power(self, power: bool) -> None:
        """ Turn the boards allocated to the job on or off.

        Does nothing if the job has not yet been allocated any boards.

        The :py:meth:`.wait_until_ready` method may be used to wait for the
        boards to fully turn on or off.

        Parameters
        ----------
        power :
            True to power on the boards, False to power off. If the boards are
            already turned on, setting power to True will reset them.
        """
        if power:
            self._client.power_on_job_boards(self.id, timeout=self._timeout)
        else:
            self._client.power_off_job_boards(self.id, timeout=self._timeout)

    def reset(self) -> None:
        """ Reset (power-cycle) the boards allocated to the job.

        Does nothing if the job has not been allocated.

        The :py:meth:`.wait_until_ready` method may be used to wait for the
        boards to fully turn on or off.
        """
        self.set_power(True)

    def _get_machine_info(self) -> "_JobMachineInfoTuple":
        """ Get information about the boards allocated to the job, e.g. the IPs
        and system dimensions.

        Returns
        -------
        :py:class:`._JobMachineInfoTuple`
        """
        info = self._client.get_job_machine_info(
            self.id, timeout=self._timeout)

        info_connections = cast(list, info["connections"])
        return _JobMachineInfoTuple(
            width=info["width"],
            height=info["height"],
            connections=({(x, y): hostname
                          for (x, y), hostname in info_connections}
                         if info_connections is not None
                         else None),
            machine_name=info["machine_name"],
            boards=info["boards"])

    @property
    def state(self) -> JobState:
        """ The current state of the job.
        """
        return self._get_state().state

    @property
    def power(self) -> bool:
        """ Are the boards powered/powering on or off?
        """
        return self._get_state().power

    @property
    def reason(self) -> str:
        """ For what reason was the job destroyed (if any and if destroyed).
        """
        return self._get_state().reason

    @property
    def connections(self) -> Dict[Tuple[int, int], str]:
        """ The list of Ethernet connected chips and their IPs.

        {(x, y): hostname, ...} or None
        """
        # Note that the connections for a job will never change once defined so
        # only need to get this once.
        if (self._last_machine_info is None or
                self._last_machine_info.connections is None):
            self._last_machine_info = self._get_machine_info()

        return self._last_machine_info.connections

    @property
    def hostname(self) -> Optional[str]:
        """ The hostname of chip 0, 0 (or None if not allocated yet).
        """
        return self.connections[(0, 0)]

    @property
    def width(self) -> int:
        """ The width of the allocated machine in chips (or None).
        """
        # Note that the dimensions of a job will never change once defined so
        # only need to get this once.
        if (self._last_machine_info is None or
                self._last_machine_info.width is None):
            self._last_machine_info = self._get_machine_info()

        return self._last_machine_info.width

    @property
    def height(self) -> int:
        """ The height of the allocated machine in chips (or None).
        """
        # Note that the dimensions of a job will never change once defined so
        # only need to get this once.
        if (self._last_machine_info is None or
                self._last_machine_info.height is None):
            self._last_machine_info = self._get_machine_info()

        return self._last_machine_info.height

    @property
    def machine_name(self) -> str:
        """ The name of the machine the job is allocated on (or None).
        """
        # Note that the machine will never change once defined so only need to
        # get this once.
        if (self._last_machine_info is None or
                self._last_machine_info.machine_name is None):
            self._last_machine_info = self._get_machine_info()

        return self._last_machine_info.machine_name

    @property
    def boards(self) -> Optional[JsonArray]:
        """ The coordinates of the boards allocated for the job (or None).
        """
        # Note that the machine will never change once defined so only need to
        # get this once.
        if (self._last_machine_info is None or
                self._last_machine_info.machine_name is None):
            self._last_machine_info = self._get_machine_info()

        return self._last_machine_info.boards

    def wait_for_state_change(self, old_state: JobState,
                              timeout: Optional[float] = None) -> JobState:
        """ Block until the job's state changes from the supplied state.

        Parameters
        ----------
        old_state :
            The current state.
        timeout :
            The number of seconds to wait for a change before timing out. If
            None, wait forever.

        Returns
        -------
        :py:class:`~spalloc_client.JobState`
            The new state, or old state if timed out.
        """
        finish_time = make_timeout(timeout)

        # We may get disconnected while waiting so keep listening...
        while not timed_out(finish_time):
            try:
                # Watch for changes in this Job's state
                self._client.notify_job(self.id)

                # Wait for job state to change
                while not timed_out(finish_time):
                    # Has the job changed state?
                    new_state = self._get_state().state
                    if new_state != old_state:
                        return new_state

                    # Wait for a state change and keep the job alive
                    if not self._do_wait_for_a_change(finish_time):
                        # The user's timeout expired while waiting for a state
                        # change, return the old state and give up.
                        return old_state
            except (IOError, OSError, ProtocolTimeoutError):
                # Something went wrong while communicating with the server,
                # reconnect after the reconnection delay (or timeout, whichever
                # came first.
                self._do_reconnect(finish_time)

        # If we get here, the timeout expired without a state change, just
        # return the old state
        return old_state

    def _do_wait_for_a_change(self, finish_time: Optional[float]) -> bool:
        """ Wait for a state change and keep the job alive.
        """
        # Since we're about to block holding the client lock, we must be
        # responsible for keeping everything alive.
        while not timed_out(finish_time):
            self._client.job_keepalive(self.id, timeout=self._timeout)

            # Wait for the job to change
            try:
                # Block waiting for the job to change no-longer than the
                # user-specified timeout or half the keepalive interval.
                if finish_time is not None and self._keepalive is not None:
                    wait_timeout = min(self._keepalive / 2.0,
                                       time_left_float(finish_time))
                elif finish_time is None:
                    wait_timeout = None if self._keepalive is None \
                        else self._keepalive / 2.0
                else:
                    wait_timeout = time_left_float(finish_time)
                if wait_timeout is None or wait_timeout >= 0.0:
                    self._client.wait_for_notification(wait_timeout)
                    return True
            except ProtocolTimeoutError:
                # Its been a while, send a keep-alive since we're still
                # holding the lock
                pass
        # The user's timeout expired while waiting for a state change
        return False

    def _do_reconnect(self, finish_time: Optional[float]) -> None:
        """ Reconnect after the reconnection delay (or timeout, whichever
        came first).
        """
        self._client.close()
        if finish_time is not None:
            delay = min(time_left_float(finish_time), self._reconnect_delay)
        else:
            delay = self._reconnect_delay
        time.sleep(max(0.0, delay))
        self._reconnect()

    def wait_until_ready(self, timeout: Optional[float] = None) -> None:
        """ Block until the job is allocated and ready.

        Parameters
        ----------
        timeout :
            The number of seconds to wait before timing out. If None, wait
            forever.

        Raises
        ------
        StateChangeTimeoutError
            If the timeout expired before the ready state was entered.
        JobDestroyedError
            If the job was destroyed before becoming ready.
        """
        cur_state = None
        finish_time = make_timeout(timeout)
        while not timed_out(finish_time):
            if cur_state is None:
                # Get initial state (NB: done here such that the command is
                # never sent if the timeout has already occurred)
                cur_state = self._get_state().state

            # Are we ready yet?
            if cur_state == JobState.ready:
                # Now in the ready state!
                return
            elif cur_state == JobState.queued:
                logger.info("Job has been queued by the spalloc server.")
            elif cur_state == JobState.power:
                logger.info("Waiting for board power commands to complete.")
            elif cur_state == JobState.destroyed:
                # In a state which can never become ready
                raise JobDestroyedError(self._get_state().reason)
            elif cur_state == JobState.unknown:
                # Server has forgotten what this job even was...
                raise JobDestroyedError(
                    "Spalloc server no longer recognises job.")

            # Wait for a state change...
            cur_state = self.wait_for_state_change(
                cur_state, time_left(finish_time))

        # Timed out!
        raise StateChangeTimeoutError()

    def where_is_machine(
            self, chip_x: int, chip_y: int) -> Tuple[int, int, int]:
        """ Locates and returns cabinet, frame, board for a given chip in a\
        machine allocated to this job.

        :param chip_x: chip x location
        :param chip_y: chip y location
        :return: tuple of (cabinet, frame, board)
        """
        result = self._client.where_is(
            job_id=self.id, chip_x=chip_x, chip_y=chip_y)
        if result is None:
            raise ValueError("received None instead of machine location")
        [cabinet, frame, board] = cast(list, result['physical'])
        return (cast(int, cabinet), cast(int, frame), cast(int, board))


class StateChangeTimeoutError(Exception):
    """ Thrown when a state change takes too long to occur.
    """


class JobDestroyedError(Exception):
    """ Thrown when the job was destroyed while waiting for it to become\
        ready.
    """


class _JobStateTuple(namedtuple("_JobStateTuple",
                                "state,power,keepalive,reason")):
    """ Tuple describing the state of a particular job, returned by\
        :py:meth:`.Job._get_state`.

    Parameters
    ----------
    state : :py:class:`.JobState`
        The current state of the queried job.
    power : bool or None
        If job is in the ready or power states, indicates whether the boards
        are power{ed,ing} on (True), or power{ed,ing} off (False). In other
        states, this value is None.
    keepalive : float or None
        The Job's keepalive value: the number of seconds between queries
        about the job before it is automatically destroyed. None if no
        timeout is active (or when the job has been destroyed).
    reason : str or None
        If the job has been destroyed, this may be a string describing the
        reason the job was terminated.
    """

    __slots__ = ()


class _JobMachineInfoTuple(namedtuple("_JobMachineInfoTuple",
                                      "width,height,connections,"
                                      "machine_name,boards")):
    """ Tuple describing the machine allocated to a job, returned by\
        :py:meth:`.Job._get_machine_info`.

    Parameters

    from collections import namedtuple
    ----------
    width, height : int or None
        The dimensions of the machine in *chips* or None if no machine
        allocated.
    connections : {(x, y): hostname, ...} or None
        A dictionary mapping from SpiNNaker Ethernet-connected chip coordinates
        in the machine to hostname or None if no machine allocated.
    machine_name : str or None
        The name of the machine the job is allocated on or None if no machine
        allocated.
    boards : [[x, y, z], ...] or None
        The logical board coordinates of all boards allocated to the job or
        None if none allocated yet.
    """

    __slots__ = ()
