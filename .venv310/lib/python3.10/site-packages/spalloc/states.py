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

""" Defines the states a job may be in according to the protocol.
"""

from enum import IntEnum


class JobState(IntEnum):
    """ All the possible states that a job may be in.
    """

    unknown = 0
    """ The job ID requested was not recognised.
    """

    queued = 1
    """ The job is waiting in a queue for a suitable machine.
    """

    power = 2
    """ The boards allocated to the job are currently being powered on or
        powered off.
    """

    ready = 3
    """ The job has been allocated boards and the boards are not currently
        powering on or powering off.
    """

    destroyed = 4
    """ The job has been destroyed.
    """
