# Copyright (c) 2017 The University of Manchester
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

from six import add_metaclass
from spinn_utilities.abstract_base import AbstractBase, abstractmethod


@add_metaclass(AbstractBase)
class AbstractDataWriter(object):
    """ Abstract writer used to write data somewhere.
    """

    __slots__ = []

    @abstractmethod
    def write(self, data):
        """ Write some bytes of data to the underlying storage.\
            Does not return until all the bytes have been written.

        :param data: The data to write
        :type data: bytearray or bytes
        :return: Nothing is returned
        :rtype: None
        :raise IOError: If an error occurs writing to the underlying storage
        """
