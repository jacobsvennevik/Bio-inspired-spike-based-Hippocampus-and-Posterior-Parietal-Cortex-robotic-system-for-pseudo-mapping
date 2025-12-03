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

import io
from spinn_utilities.overrides import overrides
from spinn_storage_handlers.abstract_classes import (
    AbstractDataReader, AbstractContextManager)
from .exceptions import DataReadException


class FileDataReader(AbstractDataReader, AbstractContextManager):
    """ A reader that can read data from a file.
    """

    __slots__ = [
        # the container for the file
        "_file_container"
    ]

    def __init__(self, filename):
        """
        :param filename: The file to read
        :type filename: str
        :raise DataReadException: \
            If the file cannot found or opened for reading
        """
        try:
            self._file_container = io.open(filename, mode="rb")
        except IOError as e:
            raise DataReadException(str(e))

    @overrides(AbstractDataReader.read)
    def read(self, n_bytes=None):
        return self._file_container.read(n_bytes)

    def readinto(self, data):
        """ Read some bytes of data from the underlying storage into a\
            predefined array.  Will block until some bytes are available,\
            but may not fill the array completely.

        :param data: The place where the data is to be stored
        :type data: bytearray
        :return: The number of bytes stored in data
        :rtype: int
        :raise IOError: If an error occurs reading from the underlying storage
        """
        return self._file_container.readinto(data)

    @overrides(AbstractDataReader.tell)
    def tell(self):
        return self._file_container.tell()

    @overrides(AbstractContextManager.close, extend_doc=False)
    def close(self):
        """ Closes the file.

        :rtype: None
        :raise DataReadException: If the file cannot be closed
        """
        try:
            self._file_container.close()
        except IOError as e:
            raise DataReadException(str(e))
