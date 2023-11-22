import datetime
from typing import List, Union, Optional

from matplotlib import pyplot as plt

from .datafile import DataFile


class DataPlotter:
    """ A wrapper for drawing methods over DataFile objects. """

    def __init__(self, savefile_names, datasets_names, direct_loading=True):
        """
        :param savefile_names: list of files to use as data sources
        :param datasets_names: list of names to use for the datasets the different files
        :param direct_loading: if True will load all datasets after initializing them
        """

        self.save_data_files = [DataFile(savefile_name) for savefile_name in savefile_names]
        self.names = datasets_names

        # Loading all data
        if direct_loading:
            self.load()

    def load(self):
        """
        Load all the data files
        """
        for save_data_file in self.save_data_files:
            save_data_file.load()

    def draw(self, keys, function_of_key,
             savefile_name=None, savefile_prefix="",
             keys_labels={}, function_of_label=None,
             legend_string="{}", plotname=None, function_of_lim=None, y_lims=None, split=True,
             baselines: Optional[Union[List[Union[None, float]], None]]=None):
        """
        Draw a plot

        :param keys:
        The list of keys used for Y axis.
        If a single key is provided, a simple plot will be drawn.
        :param function_of_key:
        The key used for X axis.
        :param savefile_name:
        The name of the file where the plot will be saved.
        A file extension will automatically be inserted at the end.

        If a date/time is needed, use savefile_prefix as it will automatically insert a date/time before the file
        extension.
        :param savefile_prefix:
        A prefix for date/time generated filename,
        Date/time information and a file extension will automatically be inserted at the end.

        It will only be used if no savefile_name is provided.
        :param keys_labels:
        A dictionnary of labels to use for each key, used to label Y axis.
        If none is provided, the keys themselves will be used as labels.

        Format of the dictionnary is {key: label, ...}
        :param function_of_label:
        A label for the X axis.
        If none is provided, the function_of_key itfelf will be used as label.
        :param legend_string:
        A text in wich insert the name of the dataset.

        Format of the string is "text {} text", and it must contain a "{}".
        :param plotname:
        A specific name for the plot.
        If none is provided, "{} as a function of {}" like name will be generated.
        :param function_of_lim:
        The limits of X axis.
        :param y_lims:
        A list of limits for Y axis.
        :param split:
        If true, split the graphs into sub-graphs. Otherwise, draw everything on a single graph.
        :param baselines:
        List of values to consider as baselines, they are drawn as horizontal lines on plots. There is one per key.
        """
        # [tmin, tmax, ymin, ymax]

        # Clearing previous plot
        plt.gcf().clear()
        # plt.rcParams["figure.figsize"] = (6, len(keys) * 2)

        # Defining file name
        if savefile_name is None:
            date = datetime.datetime.now()
            savefile_name = "{}{}_{:02d}_{:02d}-{:02d}h{:02d}".format(savefile_prefix,
                                                                      date.year, date.month, date.day,
                                                                      date.hour, date.minute)
        savefile_name += ".png"

        # Fixing label
        function_of_label = function_of_label or function_of_key

        # Defining curve legend
        legends = [legend_string.format(datasets_name) for datasets_name in self.names]
        # plt.legend(legends)

        # Defining plot name
        plotname = plotname or "{} as a function of {}".format(
            ", ".join(keys_labels.values() or keys), function_of_label)
        plt.title(plotname)

        # getting X axis data
        fuction_of_datas = []
        for i in range(len(self.save_data_files)):
            fuction_of_datas.append(self.save_data_files[i].get_data(function_of_key))

        # Case where there are more than one key to consider, so more than one datatype
        if len(keys) > 1:
            # Annotating plot
            # Defining axis labels
            plt.xlabel(function_of_label)

            # Preparing all axis
            plot, host = plt.subplots(len(keys), sharex=True, figsize=(6, 2 * len(keys))) if split else plt.subplots()

            # Defining axis if some are defined
            if function_of_lim:
                plt.xlim(function_of_lim)

            plots = {keys[i]: host[i] for i in range(len(keys))} if split else {keys[0]: host}

            for i in range(len(keys)):
                key = keys[i]

                # Configuration of secondary axes (useless if in split mode)
                if i > 0 and not split:
                    plots[key] = host.twinx()

                    # Moving axis
                    if i > 1:
                        plots[key].spines["right"].set_position(("axes", 1 + (0.2 * i)))

                    # Showing frame
                    plots[key].set_frame_on(True)

                    # Hiding patches
                    plots[key].patch.set_visible(False)

                    # Hiding spines
                    for sp in plots[key].spines.values():
                        sp.set_visible(False)

                    if i > 1:
                        plots[key].spines["right"].set_visible(True)

                # use defined label for axis if it exists
                key_label = keys_labels and keys_labels[key] or str(key)
                plots[key].set_ylabel(key_label)
                if y_lims and y_lims[i]:
                    plt.ylim(y_lims[i])

                lines = []
                # Drawing the baselines
                if baselines is not None and baselines[i] is not None:
                    line = plots[key].axhline(baselines[i], linestyle='dotted', label="Baseline {}".format(round(baselines[i], 3)))
                    lines.append(line)

                # Drawing the plot for each dataset
                for j in range(len(self.save_data_files)):
                    data = self.save_data_files[j].get_data(key)

                    # Drawing plot
                    line, = plots[key].plot(fuction_of_datas[j], data, label=legends[j])
                    lines.append(line)

                plots[key].legend(handles=lines)

        # Case where there is only one key to consider, so a single datatype
        else:
            key = keys[0]

            # use defined label for axis if it exists
            key_label = keys_labels and keys_labels[key] or str(key)

            # Annotating plot
            # Defining axis labels
            plt.xlabel(function_of_label)
            plt.ylabel(key_label)

            # Defining axis if some are defined
            if function_of_lim:
                plt.xlim(function_of_lim)
            if y_lims and y_lims[0]:
                plt.ylim(y_lims[0])

            # Defining plot name
            plotname = plotname or "{} as a function of {}".format(key_label, function_of_label)
            plt.title(plotname)

            # Drawing the baselines
            if baselines is not None and baselines[0] is not None:
                line = plt.axhline(baselines[0], linestyle='dotted',
                                          label="Baseline {}".format(round(baselines[0], 3)))

            # Drawing the plot for each dataset
            for i in range(len(self.save_data_files)):
                save_data_file = self.save_data_files[i]

                fuction_of_data = save_data_file.get_data(function_of_key)
                data = save_data_file.get_data(key)

                # Drawing plot
                plt.plot(fuction_of_data, data, label=legends[i])

            plt.legend()

        # produce PNG
        plt.savefig(savefile_name)
