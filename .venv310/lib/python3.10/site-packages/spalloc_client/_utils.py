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

from datetime import datetime
import time
from typing import Optional


def time_left_float(timestamp: float) -> float:
    """ Convert a not None timestamp into how long to wait for it.

    :return: How far in the future the timestamp is
    """
    return max(0.0, timestamp - time.time())


def time_left(timestamp: Optional[float]) -> Optional[float]:
    """ Convert a timestamp into how long to wait for it.

    :return: How far in the future the timestamp is
        or None if timestamp is None
    """
    if timestamp is None:
        return None
    return max(0.0, timestamp - time.time())


def timed_out(timestamp: Optional[float]) -> bool:
    """ Check if a timestamp has been reached.

    :returns: True if the timeout has been reached or if there is no timeout
    """
    if timestamp is None:
        return False
    return timestamp < time.time()


def make_timeout(delay_seconds: Optional[float]) -> Optional[float]:
    """ Convert a delay (in seconds) into a timestamp.

    :returns: the current time plus the delay or None if delay is None
    """
    if delay_seconds is None:
        return None
    return time.time() + delay_seconds


def render_timestamp(timestamp: float) -> str:
    """ Convert a timestamp (Unix seconds) into a local human-readable\
        timestamp string.

    :returns: timestamp in human readable format
    """
    return datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M:%S")
