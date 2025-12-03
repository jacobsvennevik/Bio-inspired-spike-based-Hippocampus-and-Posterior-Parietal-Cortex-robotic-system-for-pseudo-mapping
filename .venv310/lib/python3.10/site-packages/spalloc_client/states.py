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

""" Defines the states a job may be in according to the protocol.
"""

from enum import IntEnum


class JobState(IntEnum):
    """ All the possible states that a job may be in.
    """

    # pylint: disable=invalid-name
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
