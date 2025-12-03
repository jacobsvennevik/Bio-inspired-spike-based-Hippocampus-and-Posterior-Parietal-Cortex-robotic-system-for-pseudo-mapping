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
class AbstractDataReader(object):
    """ Abstract reader used to read data from somewhere.
    """

    __slots__ = []

    @abstractmethod
    def read(self, n_bytes=None):
        """ Read some bytes of data from the underlying storage.  Will block\
            until some bytes are available, but might not return the full\
            `n_bytes`.  The size of the returned array indicates how many\
            bytes were read.

        :param n_bytes: The number of bytes to read; if unspecified, read all\
            remaining bytes
        :type n_bytes: int
        :return: The data that was read
        :rtype: bytearray
        :raise IOError: If an error occurs reading from the underlying storage
        """

    @abstractmethod
    def tell(self):
        """ Returns the position of the file cursor.

        :return: Position of the file cursor
        :rtype: int
        """
