import os
from typing import Optional, List, Dict, Any, Union, TextIO

SEP_CHAR = ':'

DataDict = Dict[str, Any]


class DataFile:
    """ A data_bkp system plot-oriented and with direct file I/O human readable system """

    def __init__(self, savefile_name: str, keys: Optional[List[str]] = [], buffer_size: Optional[int] = 1):
        """Creates an new DataFile, linked to a file

        :param savefile_name: the name of the file that will contain the data_bkp and from which data_bkp will be loaded
        :param keys: list of keys that will be used, usefull only if the file does not exist
        """

        self.data = []
        self.keys = keys
        self.savefile_name = savefile_name

        self.buffer_size = buffer_size
        self.write_buffer = []

    ####################################################################################################################
    #       File reading operation                                                                                     #
    ####################################################################################################################

    def load(self) -> None:
        """
        Load all the data_bkp and the keys from the savefile
        """
        self.data.clear()

        with open(self.savefile_name, 'r') as savefile:
            header_line = True

            for line in savefile:

                if len(line) > 0:
                    if header_line:
                        header_line = False
                        self.load_keys(line)
                    else:
                        self.load_line(line)

    def load_line(self, string: str) -> None:
        """Parse a string as a dictionnary, with the keys of the current object, and add it to the data_bkp

        :param string: String that will be parsed
        """
        values = self.read_line(string)
        dictionnary = {self.keys[i]: self.convert_data(values[i]) for i in range(min(len(values), len(self.keys)))}

        self.data.append(dictionnary)

    def read_line(self, string: str) -> List[str]:
        """Extract the values of a single line of text

        :param string: line of text to parse
        """
        values = string.rstrip().split(SEP_CHAR)
        return values

    def load_keys(self, string: str) -> None:
        """Extract the values of a single line of text, and set those as keys

        :param string: line of text to parse
        """
        values = self.read_line(string)
        self.keys = list(filter(None, values))

    ####################################################################################################################
    #       File writing operation                                                                                     #
    ####################################################################################################################

    def clear_file(self) -> None:
        """Remove the content of a file"""
        with open(self.savefile_name, 'w') as savefile:
            savefile.write('')

    def save(self) -> None:
        """Save all the data_bkp in the file, after erasing previous data_bkp"""
        with open(self.savefile_name, 'w') as savefile:
            self.save_header(savefile)
            for d in self.data:
                self.save_data(d, savefile)

    def flush(self) -> None:
        """Write the data_bkp stored in the buffer to the file"""
        with open(self.savefile_name, 'a+') as f:
            # Add header if the file is empty
            if not self.file_exists():
                self.save_header(f)

            # Write data_bkp
            for data_dict in self.write_buffer:
                self.save_data(data_dict, f)

        self.write_buffer.clear()

    def save_data(self, data_dict: DataDict, savefile: TextIO) -> None:
        """Save a single line of data_bkp into the savefile

        :param data_dict: data_bkp to save
        :param savefile: target save file
        """
        values = [str(data_dict[k]) for k in self.keys]
        savefile.write(SEP_CHAR.join(values) + '\n')

    def save_header(self, savefile) -> None:
        """Save the header line containing the keys

        :param savefile: target save file
        """
        savefile.write(SEP_CHAR.join(self.keys) + '\n')

    def file_exists(self) -> bool:
        """Check if file exists and is not empty

        :return : True if file exists and is not empty
        """
        return (os.path.exists(self.savefile_name) and os.path.getsize(self.savefile_name) > 0
                and os.stat(self.savefile_name).st_size > 1)

    ####################################################################################################################
    #       Operations on stored data_bkp                                                                                  #
    ####################################################################################################################

    def add_data(self, data_dict: DataDict) -> None:
        """
        Add a single line of data_bkp to the data_bkp and add it into the savefile
        :param data_dict: data_bkp to add and save
        """
        self.data.append(data_dict)

        self.write_buffer.append(data_dict)
        if len(self.write_buffer) >= self.buffer_size:
            self.flush()

    def get_data(self, key: str) -> List[Any]:
        """Get all the data_bkp for a specific key

        :param key: key of the data_bkp to get
        :return: list of data_bkp
        """

        return [d[key] for d in self.data]

    def convert_data(self, data: Any) -> Union[int, float, Any]:
        """Try to convert the data_bkp into int or float

        :param data:
        :return: transformed data_bkp
        """
        try:
            return int(data)
        except ValueError:
            try:
                return float(data)
            except ValueError:
                return data

    def set_data(self, key: str, values: List[Any]) -> None:
        """Set or replace the values of a specific key"""
        for i in range(min(len(self.data), len(values))):
            self.data[i][key] = values[i]

    def average(self, key, length) -> None:
        """Average the data_bkp of the key over a set length.

        Replace the data_bkp of a specific key by an averaged version of this data_bkp.

        Each value is replaced by a mean of itself and the values preceding it.
        The length defines the number of values to average.

        :param key: Key of the data_bkp to average
        :param length: Length of the average (number of values to average)
        """
        self.set_data(key, average(self.get_data(key), length))


def average(list: List[Union[int, float]], length: int) -> List[Union[int, float]]:
    """Average a list of numbers over a set length.

    Each value is replaced by a mean of itself and the values preceding it.
    The length defines the number of values to average.

    :param list: List of values to average
    :param length: Length of the average (number of values to average)
    """
    list_averaged = []

    if len(list) > length:
        list_part = []
        for i in range(length):
            list_part.append(list[i])
            list_averaged.append(sum(list_part) / len(list_part))

        for i in range(length, len(list)):
            list_part = [list[j] for j in range(i, min(len(list), i + length))]
            list_averaged.append(sum(list_part) / len(list_part))

    else:
        for i in range(len(list)):
            list_part = [list[j] for j in range(i + 1)]
            list_averaged.append(sum(list_part) / len(list_part))

    return list_averaged
