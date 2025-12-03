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


def time_left(timestamp):
    """ Convert a timestamp into how long to wait for it.
    """
    if timestamp is None:
        return None
    return max(0.0, timestamp - time.time())


def timed_out(timestamp):
    """ Check if a timestamp has been reached.
    """
    if timestamp is None:
        return False
    return timestamp < time.time()


def make_timeout(delay_seconds):
    """ Convert a delay (in seconds) into a timestamp.
    """
    if delay_seconds is None:
        return None
    return time.time() + delay_seconds
