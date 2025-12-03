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

import time
from threading import Thread, Event
import pytest
from mock import Mock
from spalloc import Job, JobState, JobDestroyedError, ProtocolTimeoutError
from spalloc._keepalive_process import keep_job_alive
from spalloc.job import (
    _JobStateTuple, _JobMachineInfoTuple, StateChangeTimeoutError,
    VERSION_RANGE_START, VERSION_RANGE_STOP)

GOOD_VERSION = ".".join(map(str, VERSION_RANGE_START))
BAD_VERSION = ".".join(map(str, VERSION_RANGE_STOP))


@pytest.fixture
def client(monkeypatch):
    # Mock out the client.
    client = Mock()
    client.version.return_value = GOOD_VERSION
    client.create_job.return_value = 123

    import spalloc.job
    monkeypatch.setattr(spalloc.job, "ProtocolClient",
                        Mock(return_value=client))
    import spalloc._keepalive_process
    monkeypatch.setattr(spalloc._keepalive_process, "ProtocolClient",
                        Mock(return_value=client))
    return client


@pytest.yield_fixture
def j(client):
    # Create a job and create/destroy it
    j = Job(hostname="localhost", owner="me")
    yield j
    j.destroy()


class TestConstructor(object):

    def test_no_hostname(self, no_config_files, client):
        with pytest.raises(ValueError):
            Job(owner="foo")

    def test_no_owner(self, no_config_files, client):
        with pytest.raises(ValueError):
            Job(hostname="foo")

    def test_both_tags_and_machine_specified(self, client,
                                             basic_config_file):
        with pytest.raises(ValueError):
            Job(machine="m", tags=["foo"])

    def test_args_from_config(self, basic_config_file, client,
                              basic_job_kwargs):
        # NB: Must override basic config file's tags default
        j = Job(tags=None)
        try:
            basic_job_kwargs.pop("hostname")
            basic_job_kwargs.pop("port")
            basic_job_kwargs.pop("reconnect_delay")
            assert len(client.create_job.mock_calls) == 1
            args = client.create_job.mock_calls[0][1]
            kwargs = client.create_job.mock_calls[0][2]
            assert args == tuple()
            assert kwargs == basic_job_kwargs
        finally:
            j.close()

    def test_override_config_file(self, basic_config_file, client):
        # The basic config file should not be found and thus neither should the
        # hostname.
        with pytest.raises(ValueError):
            Job(config_filenames=[])

    def test_override_config(self, basic_config_file, basic_job_kwargs,
                             client):
        j = Job(hostname="thehost",
                port=123321,
                reconnect_delay=0.1,
                timeout=0.2,
                owner="mossblaser",
                keepalive=0.3,
                machine=None,
                tags=["baz", "quz"],
                min_ratio=0.4,
                max_dead_boards=None,
                max_dead_links=None,
                require_torus=False)
        try:
            assert len(client.create_job.mock_calls) == 1
            args = client.create_job.mock_calls[0][1]
            kwargs = client.create_job.mock_calls[0][2]
            assert args == tuple()
            assert kwargs == dict(timeout=0.2,
                                  owner="mossblaser",
                                  keepalive=0.3,
                                  machine=None,
                                  tags=["baz", "quz"],
                                  min_ratio=0.4,
                                  max_dead_boards=None,
                                  max_dead_links=None,
                                  require_torus=False)
        finally:
            j.close()

    @pytest.mark.parametrize("state", [JobState.queued,
                                       JobState.power,
                                       JobState.ready])
    def test_resume_job_id_ok(self, basic_config_file, client, state):
        client.get_job_state.return_value = {
            "state": int(state),
            "keepalive": 60.0,
            "power": None,
            "reason": None,
        }
        j = Job(resume_job_id=1234)
        try:
            assert len(client.create_job.mock_calls) == 0
            assert len(client.get_job_state.mock_calls) == 1
            assert client.get_job_state.mock_calls[0][1] == (1234, )
        finally:
            j.close()

    @pytest.mark.parametrize("state", [JobState.unknown,
                                       JobState.destroyed])
    @pytest.mark.parametrize("reason", [None, "because"])
    def test_resume_job_id_bad(self, basic_config_file, client, state, reason):
        client.get_job_state.return_value = {
            "state": int(state),
            "keepalive": None,
            "power": None,
            "reason": reason,
        }
        with pytest.raises(JobDestroyedError):
            Job(resume_job_id=1234)
        assert len(client.create_job.mock_calls) == 0
        assert len(client.get_job_state.mock_calls) == 1
        assert client.get_job_state.mock_calls[0][1] == (1234, )


@pytest.mark.parametrize("version,ok",
                         [(GOOD_VERSION, True),
                          (BAD_VERSION, False)])
def test_version_check(client, no_config_files, version, ok):
    client.version.return_value = version
    if ok:
        # If a good version is supplied, should continue to register the job
        Job(hostname="localhost", owner="me").close()
        assert len(client.create_job.mock_calls) == 1
    else:
        # If a bad version number is returned, should just stop
        with pytest.raises(ValueError):
            Job(hostname="localhost", owner="me")
        assert len(client.create_job.mock_calls) == 0


class TestKeepalive(object):

    def test_normal_operation(self, client, no_config_files):
        # Make sure that the keepalive is sent out at the correct interval by
        # the background thread (and make sure this thread is daemonic
        event = Event()
        j = Thread(target=keep_job_alive, args=(
            "localhost", 12345, 1, 0.2, 0.1, 0.1, event))
        j.start()
        time.sleep(0.55)
        event.set()

        assert 4 <= len(client.job_keepalive.mock_calls) <= 6

    def test_reconnect(self, client, no_config_files):
        # Make sure that we can reconnect in the keepalive thread
        client.job_keepalive.side_effect = [
            IOError(), IOError(), None, None, None, None]
        client.connect.side_effect = [
            None, IOError(), None, None, None, None]
        event = Event()
        j = Thread(target=keep_job_alive, args=(
            "localhost", 12345, 1, 0.2, 0.1, 0.2, event))
        j.start()
        time.sleep(0.55)
        event.set()

        # Should have attempted a reconnect after a 0.1 + 0.2 second delay then
        # started sending keepalives as usual every 0.1 sec
        assert 2 <= len(client.job_keepalive.mock_calls) <= 4
        assert len(client.connect.mock_calls) == 3

    def test_stop_while_server_down(self, client, no_config_files):
        client.job_keepalive.side_effect = IOError()

        # Make sure that we can stop the background thread while the server is
        # down.
        j = Job(hostname="localhost", owner="me",
                keepalive=0.1, reconnect_delay=2.0)
        # Make sure we're stuck reconnecting when we close the job
        time.sleep(0.1)
        j.close()


def test_get_state(no_config_files, j, client):
    client.get_job_state.return_value = {
        "state": 3,
        "power": True,
        "keepalive": 60.0,
        "reason": None
    }
    assert j._get_state() == _JobStateTuple(state=JobState.ready, power=True,
                                            keepalive=60.0, reason=None)


@pytest.mark.parametrize("power", [True, False])
def test_set_power(no_config_files, j, power, client):
    j.set_power(power)
    if power:
        assert len(client.power_on_job_boards.mock_calls) == 1
        assert len(client.power_off_job_boards.mock_calls) == 0
    else:
        assert len(client.power_on_job_boards.mock_calls) == 0
        assert len(client.power_off_job_boards.mock_calls) == 1


def test_reset(no_config_files, j, client):
    j.reset()
    assert len(client.power_on_job_boards.mock_calls) == 1
    assert len(client.power_off_job_boards.mock_calls) == 0


@pytest.mark.parametrize("allocated", [True, False])
def test_get_machine_info(no_config_files, j, allocated, client):
    if allocated:
        client.get_job_machine_info.return_value = {
            "width": 8, "height": 8,
            "connections": [((0, 0), "localhost")],
            "machine_name": "m",
            "boards": [[0, 0, 0]],
        }
    else:
        client.get_job_machine_info.return_value = {
            "width": None, "height": None,
            "connections": None, "machine_name": None,
            "boards": None,
        }

    info = j._get_machine_info()

    if allocated:
        assert info == _JobMachineInfoTuple(
            width=8, height=8,
            connections={(0, 0): "localhost"},
            machine_name="m",
            boards=[[0, 0, 0]],
        )
    else:
        assert info == _JobMachineInfoTuple(
            width=None, height=None,
            connections=None, machine_name=None,
            boards=None,
        )


class TestWaitForStateChange(object):

    def test_state_already_changed(self, no_config_files, j, client):
        client.get_job_state.return_value = {
            "state": int(JobState.ready),
            "power": True,
            "keepalive": 60.0,
            "reason": None,
        }
        assert j.wait_for_state_change(JobState.power) == JobState.ready

    def test_change_on_event(self, no_config_files, j, client):
        # If the state change is notified via a notification, this should work.
        # We also send a "false-positive" change notification in this test.
        client.get_job_state.side_effect = [{
            "state": int(JobState.power),
            "power": True,
            "keepalive": 60.0,
            "reason": None,
        }]*2 + [{
            "state": int(JobState.ready),
            "power": True,
            "keepalive": 60.0,
            "reason": None,
        }]

        assert j.wait_for_state_change(JobState.power) == JobState.ready

    @pytest.mark.parametrize("timeout", [None, 5.0])
    def test_keepalive(self, no_config_files, timeout, client):
        # We should check that the timeout given to the wait-for-notification
        # is short enough to ensure keepalives are sent
        j = Job(hostname="localhost", owner="me", keepalive=0.2)

        client.wait_for_notification.side_effect = [
            ProtocolTimeoutError(), None]
        client.get_job_state.side_effect = [
            {
                "state": int(JobState.power),
                "power": True,
                "keepalive": 60.0,
                "reason": None,
            },
            {
                "state": int(JobState.ready),
                "power": True,
                "keepalive": 60.0,
                "reason": None,
            }
        ]

        assert j.wait_for_state_change(JobState.power) == JobState.ready
        assert len(client.job_keepalive.mock_calls) == 2
        assert len(client.wait_for_notification.mock_calls) == 2
        assert 0.05 < client.wait_for_notification.mock_calls[0][1][0] <= 0.1

        j.close()

    def test_impossible_timeout(self, no_config_files, j, client):
        # When an impossible timeout is presented, should terminate immediately
        assert j.wait_for_state_change(2, timeout=0.0) == 2

    @pytest.mark.parametrize("keepalive", [None, 5.0])
    def test_timeout(self, no_config_files, keepalive, client):
        # Make sure that the timeout argument works when presented with a
        # no state-changes. In this instance we verify the timeout given is
        # sensible.
        j = Job(hostname="localhost", owner="me", keepalive=keepalive)

        def wait_for_notification(t):
            time.sleep(t)
            raise ProtocolTimeoutError()
        client.wait_for_notification.side_effect = wait_for_notification
        client.get_job_state.return_value = {
            "state": int(JobState.power),
            "power": True,
            "keepalive": 60.0,
            "reason": None,
        }

        assert j.wait_for_state_change(
            JobState.power, timeout=0.2) == JobState.power
        assert len(client.wait_for_notification.mock_calls) == 1
        assert 0.15 < client.wait_for_notification.mock_calls[0][1][0] <= 0.25

        j.close()

    def test_no_timeout_or_keepalive(self, no_config_files, client):
        # Make sure that the timeout argument works when presented with a
        # no state-changes. In this instance we verify the timeout given is
        # sensible.
        j = Job(hostname="localhost", owner="me", keepalive=None)

        client.get_job_state.side_effect = [
            {
                "state": int(state),
                "power": True,
                "keepalive": 60.0,
                "reason": None,
            }
            for state in [JobState.power, JobState.ready]
        ]

        assert j.wait_for_state_change(
            JobState.power, timeout=None) == JobState.ready
        assert len(client.wait_for_notification.mock_calls) == 1
        assert client.wait_for_notification.mock_calls[0][1][0] is None

        j.close()

    def test_server_timeout(self, no_config_files, client):
        # Make sure that if the server dies, the timeout is still respected
        client.get_job_state.side_effect = IOError()
        j = Job(hostname="localhost", owner="me")

        before = time.time()
        assert j.wait_for_state_change(2, timeout=0.2) == 2
        after = time.time()
        assert 0.2 <= after - before < 0.3

        j.destroy()

    def test_reconnect(self, no_config_files, client):
        # If the server disconnects, the client should reconnect

        j = Job(hostname="localhost", owner="me", keepalive=0.2,
                reconnect_delay=0.1)
        client.get_job_state.side_effect = [
            IOError(),
            {
                "state": int(JobState.ready),
                "power": True,
                "keepalive": 60.0,
                "reason": None,
            }
        ]

        assert j.wait_for_state_change(JobState.power) == JobState.ready
        assert len(client.connect.mock_calls) == 2

        j.close()


class TestWaitUntilReady(object):

    def test_success(self, no_config_files, j, client):
        # Simple mocked implementation where at first the job is in the wrong
        # state then eventually in the correct state.
        client.get_job_state.side_effect = [
            {
                "state": int(state),
                "power": None,
                "keepalive": 60.0,
                "reason": None
            }
            for state in [JobState.queued, JobState.power, JobState.ready]
        ]
        j.wait_until_ready()

    @pytest.mark.parametrize("final_state",
                             [JobState.unknown, JobState.destroyed])
    @pytest.mark.parametrize("reason", ["dead", None])
    def test_bad_state(self, no_config_files, j, client, final_state, reason):
        # Simple mocked implementation where the job enters an unrecoverable
        # state
        client.get_job_state.return_value = {
            "state": int(final_state),
            "power": None,
            "keepalive": None,
            "reason": reason
        }

        with pytest.raises(JobDestroyedError):
            j.wait_until_ready()

    def test_impossible_timeout(self, no_config_files, j):
        with pytest.raises(StateChangeTimeoutError):
            j.wait_until_ready(timeout=0.0)

    def test_timeout(self, no_config_files, j, client):
        # Simple mocked implementation which times out
        client.get_job_state.return_value = {
            "state": int(JobState.power),
            "power": True,
            "keepalive": 60.0,
            "reason": None
        }
        client.wait_for_notification.side_effect = time.sleep

        before = time.time()
        with pytest.raises(StateChangeTimeoutError):
            j.wait_until_ready(timeout=0.3)
        after = time.time()

        assert 0.3 <= after - before < 0.4


def test_context_manager_fail(no_config_files, monkeypatch, client):
    monkeypatch.setattr(Job, "wait_until_ready", Mock(side_effect=IOError()))

    j = Job(hostname="localhost", owner="me")
    with pytest.raises(IOError):
        with j:
            pass  # pragma: no cover

    client.destroy_job.assert_called_once_with(123, None)


def test_context_manager_success(no_config_files, monkeypatch, client):
    monkeypatch.setattr(Job, "wait_until_ready", Mock())

    j = Job(hostname="localhost", owner="me")
    with j as j_:
        assert j is j_

    client.destroy_job.assert_called_once_with(123, None)


def test_destroy_absorbs_error(j, client):
    client.destroy_job.side_effect = IOError()
    j.destroy()


def test_attributes(j, client):
    assert j.id == 123

    client.get_job_state.return_value = {
        "state": JobState.ready,
        "keepalive": 60.0,
        "power": True,
        "reason": None,
    }
    client.get_job_machine_info.return_value = {
        "width": 8, "height": 9,
        "connections": [((0, 0), "localhost")],
        "machine_name": "m",
        "boards": [[0, 0, 0]],
    }

    # Each of these should result in the job-state being requested every time
    assert j.state == JobState.ready
    assert len(client.get_job_state.mock_calls) == 1
    assert j.state == JobState.ready
    assert len(client.get_job_state.mock_calls) == 2

    assert j.power is True
    assert len(client.get_job_state.mock_calls) == 3
    assert j.power is True
    assert len(client.get_job_state.mock_calls) == 4

    assert j.reason is None
    assert len(client.get_job_state.mock_calls) == 5
    assert j.reason is None
    assert len(client.get_job_state.mock_calls) == 6

    # Connection information should only be fetched once
    assert j._last_machine_info is None
    assert j.hostname == "localhost"
    assert len(client.get_job_machine_info.mock_calls) == 1
    assert j.hostname == "localhost"
    assert len(client.get_job_machine_info.mock_calls) == 1

    j._last_machine_info = None
    assert j.connections == {(0, 0): "localhost"}
    assert len(client.get_job_machine_info.mock_calls) == 2
    assert j.connections == {(0, 0): "localhost"}
    assert len(client.get_job_machine_info.mock_calls) == 2

    j._last_machine_info = None
    assert j.width == 8
    assert len(client.get_job_machine_info.mock_calls) == 3
    assert j.width == 8
    assert len(client.get_job_machine_info.mock_calls) == 3

    j._last_machine_info = None
    assert j.height == 9
    assert len(client.get_job_machine_info.mock_calls) == 4
    assert j.height == 9
    assert len(client.get_job_machine_info.mock_calls) == 4

    j._last_machine_info = None
    assert j.machine_name == "m"
    assert len(client.get_job_machine_info.mock_calls) == 5
    assert j.machine_name == "m"
    assert len(client.get_job_machine_info.mock_calls) == 5

    j._last_machine_info = None
    assert j.boards == [[0, 0, 0]]
    assert len(client.get_job_machine_info.mock_calls) == 6
    assert j.boards == [[0, 0, 0]]
    assert len(client.get_job_machine_info.mock_calls) == 6
