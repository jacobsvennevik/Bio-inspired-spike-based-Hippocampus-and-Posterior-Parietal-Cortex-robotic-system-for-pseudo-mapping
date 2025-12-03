import os

import xlsxwriter


class SpikeTrace:
    """
    The SpikeTrace class contains useful functions to create traces of the spiking functional blocks.
    """
    def __init__(self, file_name, simtime):
        """
        Constructor of the class.

        :param str file_name: Name of the file which will contain the SpikeTrace object (without extension).
        :param int simtime: Time during which the simulation runs.
        """
        self.excel = xlsxwriter.Workbook(os.getcwd() + "/" + file_name + ".xlsx")
        self.worksheets = []
        self.header_format = self.createFormat("#F4B084")
        self.background_format = self.createFormat("#FCE4D6")
        self.simtime = int(simtime)

        self.writeHeader()

    def createFormat(self, color):
        """

        """
        excel_format = self.excel.add_format()
        excel_format.set_border()
        excel_format.set_bold()
        excel_format.set_align('center')
        excel_format.set_align('vcenter')
        excel_format.set_bg_color(color)

        return excel_format

    def writeHeader(self):
        worksheet_index = -1
        local_column = 0

        for i in range(self.simtime):
            if i % 1023 == 0:
                worksheet_index += 1
                local_column = 0
                self.worksheets.append(self.excel.add_worksheet())
                self.worksheets[worksheet_index].write(0, 0, "Time (ms)", self.header_format)
                self.worksheets[worksheet_index].set_column(0, 0, 15)
                self.worksheets[worksheet_index].set_column(1, self.simtime, 5)

            self.worksheets[worksheet_index].write(0, local_column + 1, i, self.header_format)

            local_column += 1

    def printSpikes(self, index, row_name, spikes, color):
        for worksheet in self.worksheets:
            worksheet.write(index, 0, row_name, self.header_format)

        values_format = self.createFormat(color)
        worksheet_index = -1
        local_column = 0

        for i in range(self.simtime):
            if i % 1023 == 0:
                worksheet_index += 1
                local_column = 0

            if i in spikes:
                self.worksheets[worksheet_index].write(index, local_column + 1, 1, values_format)
            else:
                self.worksheets[worksheet_index].write(index, local_column + 1, "", self.background_format)

            local_column += 1

    def printRow(self, index, row_name, values, color):
        for worksheet in self.worksheets:
            worksheet.write(index, 0, row_name, self.header_format)

        values_format = self.createFormat(color)
        worksheet_index = -1
        local_column = 0

        for i in range(len(values)):
            if i % 1023 == 0:
                worksheet_index += 1
                local_column = 0

            self.worksheets[worksheet_index].write(index, local_column + 1, values[i], values_format)

            local_column += 1

        for i in range(len(values), self.simtime):
            if i % 1023 == 0:
                worksheet_index += 1
                local_column = 0

            self.worksheets[worksheet_index].write(index, local_column + 1, "", self.background_format)

            local_column += 1

    def closeExcel(self):
        self.excel.close()
