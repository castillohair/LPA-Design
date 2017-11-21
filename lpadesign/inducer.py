# -*- coding: UTF-8 -*-
"""
Module that contains light inducer classes.

"""

import os
import random

import numpy
import pandas

import platedesign
import platedesign.inducer

class LightInducer(platedesign.inducer.InducerBase):
    """
    Object that represents different intensities of light.

    Attributes
    ----------
    name : str
        Name of the inducer.
    doses_table : DataFrame
        Table with information of each inducer dose.

    Methods
    -------
    set_vol_from_shots
        Set volume to prepare from number of shots and replicates.
    shuffle
        Apply random shuffling to the dose table.
    save_exp_setup_instructions
        Save instructions for the Experiment Setup stage.
    save_exp_setup_files
        Save additional files for the Experiment Setup stage.
    save_rep_setup_instructions
        Save instructions for the Replicate Setup stage.
    save_rep_setup_files
        Save additional files for the Replicate Setup stage.

    """
    def __init__(self,
                 name,
                 led_layout,
                 units=u'µmol/(m^2*s)',
                 id_prefix=None,
                 id_offset=0):
        # Store name, led layout name, and units
        self.name = name
        self.led_layout = led_layout
        self.units = units

        # Store ID modifiers for dose table
        if id_prefix is None:
            id_prefix=name[0]
        self.id_prefix=id_prefix
        self.id_offset=id_offset

        # Initialize an empty dose table
        self._doses_table = pandas.DataFrame()
        # Enable shuffling by default, but start with no shuffling and an
        # empty list of inducers to synchronize shuffling with.
        self.shuffling_enabled = True
        self.shuffled_idx = None
        self.shuffling_sync_list = []

    @property
    def _intensities_header(self):
        """
        Header to be used in the dose table to specify intensities.

        """
        return u"{} Intensity ({})".format(self.name, self.units)

    @property
    def intensities(self):
        """
        Light intensities.

        Reading from this attribute will return the contents of the
        "Intensity" column from the dose table. Writing to this attribute
        will reinitialize the doses table with the specified intensities.
        Any columns that are not the intensities or IDs will be lost.

        """
        return self.doses_table[self._intensities_header].values

    @intensities.setter
    def intensities(self, value):
        # Make sure that value is a float array
        value = numpy.array(value, dtype=numpy.float)
        # Initialize dataframe with doses info
        ids = ['{}{:03d}'.format(self.id_prefix, i)
               for i in range(self.id_offset + 1,
                              len(value) + self.id_offset + 1)]
        self._doses_table = pandas.DataFrame({'ID': ids})
        self._doses_table.set_index('ID', inplace=True)
        self._doses_table[self._intensities_header] = value

    @property
    def doses_table(self):
        """
        Table containing information of all the inducer concentrations.

        """
        if self.shuffled_idx is None:
            return self._doses_table
        else:
            return self._doses_table.iloc[self.shuffled_idx]

    def set_gradient(self,
                     min,
                     max,
                     n,
                     scale='linear',
                     use_zero=False,
                     n_repeat=1):
        """
        Set inducer intensities from a specified gradient.

        Using this function will reset the dose table and populate the
        "Intensity" column with the specified gradient.

        Parameters
        ----------
        min, max : float
            Minimum and maximum values on the gradient.
        n : int
            Number of points to use for the gradient.
        scale : {'linear', 'log'}, optional
            Whether to generate the gradient with linear or logarithmic
            spacing.
        use_zero : bool, optional.
            If ``scale`` is 'log', use zero as well. Ignored if ``scale``
            is 'linear'.
        n_repeat : int, optional
            How many times to repeat each intensity. Default: 1 (no
            repeat). Should be an exact divisor of ``n``.

        """
        # Check that n_repeat is an exact divisor of n
        if n%n_repeat != 0:
            raise ValueError("n should be a multiple of n_repeat")

        # Calculate gradient
        if scale == 'linear':
            self.intensities = numpy.linspace(min, max, n/n_repeat)
        elif scale == 'log':
            if use_zero:
                self.intensities = numpy.logspace(numpy.log10(min),
                                                  numpy.log10(max),
                                                  (n/n_repeat - 1))
                self.intensities = \
                    numpy.concatenate(([0], self.intensities))
            else:
                self.intensities = numpy.logspace(numpy.log10(min),
                                                  numpy.log10(max),
                                                  n/n_repeat)
        else:
            raise ValueError("scale {} not recognized".format(scale))

        # Repeat if necessary
        self.intensities = numpy.repeat(self.intensities, n_repeat)

    def set_vol_from_shots(self,
                           n_shots,
                           n_replicates=1):
        """
        Set volume to prepare from number of shots and replicates.

        This function does not have any function in a light inducer, and it
        is only included due to being needed by the parent class.

        """
        pass

    def sync_shuffling(self, inducer):
        """
        Register an inducer to synchronize shuffling with.

        Inducers whose shuffling is synchronized should have the same
        number of doses (i.e. the length of their doses table should be the
        same). Shuffling synchronization is based on the controlling
        inducer being able to directly modify the shuffling indices of the
        controlled inducers. Therefore, this function sets the flag
        ``shuffling_enabled`` in `inducer` to ``False``.

        Parameters
        ----------
        inducer : Inducer
            Inducer to synchronize shuffling with.

        """
        # Check length of doses table
        if len(self.doses_table) != len(inducer.doses_table):
            raise ValueError("inducers to synchronize should have the same "
                "number of doses")
        # Disable shuffling flag
        inducer.shuffling_enabled = False
        # Add to list of inducers to synchronize with
        self.shuffling_sync_list.append(inducer)

    def shuffle(self):
        """
        Apply random shuffling to the dose table.

        """
        if self.shuffling_enabled:
            # Create list of indices, shuffle, and store.
            shuffled_idx = list(range(len(self.doses_table)))
            random.shuffle(shuffled_idx)
            self.shuffled_idx = shuffled_idx
            # Write shuffled indices on inducers to synchronize with
            for inducer in self.shuffling_sync_list:
                inducer.shuffled_idx = self.shuffled_idx


class LightSignal(platedesign.inducer.InducerBase):
    """
    Object that represents a dynamic light signal.

    Attributes
    ----------
    name : str
        Name of the inducer.
    doses_table : DataFrame
        Table with information of each inducer dose.

    Methods
    -------
    set_vol_from_shots
        Set volume to prepare from number of shots and replicates.
    shuffle
        Apply random shuffling to the dose table.
    save_exp_setup_instructions
        Save instructions for the Experiment Setup stage.
    save_exp_setup_files
        Save additional files for the Experiment Setup stage.
    save_rep_setup_instructions
        Save instructions for the Replicate Setup stage.
    save_rep_setup_files
        Save additional files for the Replicate Setup stage.

    """
    def __init__(self,
                 name,
                 led_layout,
                 units=u'µmol/(m^2*s)',
                 id_prefix=None,
                 id_offset=0):
        # Store name, led layout name, and units
        self.name = name
        self.led_layout = led_layout
        self.units = units

        # Light signal
        # Each value is the LED intensity for a minute.
        self.light_signal = numpy.array([])
        # Light intensity before the signal begins.
        self.light_intensity_init = 0.

        # Store ID modifiers for dose table
        if id_prefix is None:
            id_prefix=name[0]
        self.id_prefix=id_prefix
        self.id_offset=id_offset

        # Initialize an empty dose table
        self._doses_table = pandas.DataFrame()
        # Enable shuffling by default, but start with no shuffling and an
        # empty list of inducers to synchronize shuffling with.
        self.shuffling_enabled = True
        self.shuffled_idx = None
        self.shuffling_sync_list = []

    @property
    def _sampling_times_header(self):
        """
        Header used in the dose table to specify signal sampling times.

        """
        return u"{} Sampling time (min)".format(self.name)

    @property
    def _signal_file_name_header(self):
        """
        Header to be used in the dose table with the signal file name.

        """
        return u"{} Signal File".format(self.name)

    @property
    def _signal_file_name(self):
        """
        The signal file name.

        """
        return u"{} Signal.xlsx".format(self.name)

    @property
    def sampling_times(self):
        """
        Sampling times.

        Reading from this attribute will return the contents of the
        "Sampling times" column from the dose table. Writing to this
        attribute will reinitialize the doses table with the specified time
        shifts. Any columns that are not the sampling times or IDs will be
        lost.

        """
        return self.doses_table[self._sampling_times_header].values

    @sampling_times.setter
    def sampling_times(self, value):
        # Make sure that value is an integer array
        value = numpy.array(value, dtype=numpy.int)
        # Initialize dataframe with doses info
        ids = ['{}{:03d}'.format(self.id_prefix, i)
               for i in range(self.id_offset + 1,
                              len(value) + self.id_offset + 1)]
        self._doses_table = pandas.DataFrame({'ID': ids})
        self._doses_table.set_index('ID', inplace=True)
        # Name of file with signal intensity values
        self._doses_table[self._signal_file_name_header] = \
            self._signal_file_name
        # Sampling times
        self._doses_table[self._sampling_times_header] = value

    @property
    def doses_table(self):
        """
        Table containing information of all the inducer concentrations.

        """
        if self.shuffled_idx is None:
            return self._doses_table
        else:
            return self._doses_table.iloc[self.shuffled_idx]

    def set_signal_step(self, intensity_from, intensity_to):
        """
        Set a step light signal.

        """
        # Initial light intensity
        self.light_intensity_init = intensity_from
        # Signal will be constant, as long as the largest sample time
        self.light_signal = numpy.ones(numpy.max(self.sampling_times)) * \
            intensity_to

    def set_vol_from_shots(self,
                           n_shots,
                           n_replicates=1):
        """
        Set volume to prepare from number of shots and replicates.

        This function does not have any function in a light inducer, and it
        is only included due to being needed by the parent class.

        """
        pass

    def sync_shuffling(self, inducer):
        """
        Register an inducer to synchronize shuffling with.

        Inducers whose shuffling is synchronized should have the same
        number of doses (i.e. the length of their doses table should be the
        same). Shuffling synchronization is based on the controlling
        inducer being able to directly modify the shuffling indices of the
        controlled inducers. Therefore, this function sets the flag
        ``shuffling_enabled`` in `inducer` to ``False``.

        Parameters
        ----------
        inducer : Inducer
            Inducer to synchronize shuffling with.

        """
        # Check length of doses table
        if len(self.doses_table) != len(inducer.doses_table):
            raise ValueError("inducers to synchronize should have the same "
                "number of doses")
        # Disable shuffling flag
        inducer.shuffling_enabled = False
        # Add to list of inducers to synchronize with
        self.shuffling_sync_list.append(inducer)

    def shuffle(self):
        """
        Apply random shuffling to the dose table.

        """
        if self.shuffling_enabled:
            # Create list of indices, shuffle, and store.
            shuffled_idx = list(range(len(self.doses_table)))
            random.shuffle(shuffled_idx)
            self.shuffled_idx = shuffled_idx
            # Write shuffled indices on inducers to synchronize with
            for inducer in self.shuffling_sync_list:
                inducer.shuffled_idx = self.shuffled_idx

    def save_exp_setup_files(self, path='.'):
        """
        Save accessory files during the experiment setup stage.

        This function saves an Excel file containing the signal intensity
        values over time.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Create DataFrame
        intensity_header = u"{} Intensity ({})".format(self.name, self.units)
        signal_table = pandas.DataFrame(columns=['Time (min)', intensity_header])

        # Time values for the signal go from zero to the length of the signal
        # in minutes.
        # The initial intensity is considered to be set at time = -inf
        signal_table['Time (min)'] = numpy.append([-numpy.inf],
                                                range(len(self.light_signal)))

        
        signal_table[intensity_header] = numpy.append([self.light_intensity_init],
                                                    self.light_signal)

        # Time should be the index
        signal_table.set_index('Time (min)', inplace=True)

        # Save DataFrame
        writer = pandas.ExcelWriter(os.path.join(path, self._signal_file_name),
                                    engine='openpyxl')
        signal_table.to_excel(writer)
        writer.save()
