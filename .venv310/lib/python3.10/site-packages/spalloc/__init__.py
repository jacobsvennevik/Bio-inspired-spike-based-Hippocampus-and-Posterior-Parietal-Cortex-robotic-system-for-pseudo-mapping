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

from spalloc._version import __version__  # noqa

# Alias useful objects
from spalloc.protocol_client import ProtocolClient, ProtocolError  # noqa
from spalloc.protocol_client import ProtocolTimeoutError  # noqa
from spalloc.protocol_client import SpallocServerException  # noqa
from spalloc.job import Job, JobDestroyedError, StateChangeTimeoutError  # noqa
from spalloc.states import JobState  # noqa
