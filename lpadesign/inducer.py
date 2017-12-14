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

class LPAInducerBase(platedesign.inducer.InducerBase):
    """
    Generic class that represents one or more doses of an LPA inducer.

    This class is meant to be inherited by a class representing a concrete
    LPA light inducer type.

    Parameters
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str, optional
        Units in which light intensity is expressed.
    led_layout : str, optional
        Name of the LED layout associated with this inducer. A layout
        describes a mapping from LED types to each well of an arbitrary
        LPA, without reference to a specific LPA or LEDs. An Excel file
        mapping LED layouts to calibrated LED sets must be specified to the
        ``LPA-Program`` module. For more information, refer to
        ``LPA-Program``'s documentation. The LED layout name can be
        specified during the object's creation, or sometime before
        generating the experiment files.
    led_channel : int, optional
        The LED channel used by the inducer in an LPA. This can be
        specified during the object's creation, or sometime before
        generating the experiment files.

    Attributes
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str
        Units in which light intensity is expressed.
    led_layout : str
        Name of the LED layout associated with this inducer.
    led_channel : int
        The LED channel used by the inducer in an LPA.
    doses_table : DataFrame
        Table containing information of all the inducer intensities.

    Other Attributes
    ----------------
    shuffling_enabled : bool
        Whether shuffling of the doses table is enabled. If False, the
        `shuffle` function will not change the order of the rows in the
        doses table.
    shuffled_idx : list
        Randomized indices that result in the current shuffling of
        doses.
    shuffling_sync_list : list
        List of inducers with which shuffling should be synchronized.
    time_step_size : int
        Number of milliseconds in each time step. Default: None (not
        specified).
    time_step_units : str
        Specific name of each time step (e.g. minute), to be used in
        generated files. Default: None (not specified).
    n_time_steps : int
        Number of time steps in the LPA program. Default: None (not
        specified).

    """
    def __init__(self,
                 name,
                 units=u'µmol/(m^2*s)',
                 led_layout=None,
                 led_channel=None,):
        # Parent's __init__ stores name, units, initializes doses table, and
        # sets shuffling parameters.
        super(LPAInducerBase, self).__init__(name=name, units=units)

        # Store led layout and led channel
        self.led_layout = led_layout
        self.led_channel = led_channel

        # Initialize time step attributes
        self.time_step_size = None
        self.time_step_units = None
        self.n_time_steps = None

    def get_lpa_intensity(self, dose_idx):
        """
        Get the LPA intensity sequence for a specified dose.

        An LPA light inducer can have time-varying intensities, and/or only
        partial information exposed via ``doses_table``. This function
        returns the fully resolved sequence of light intensities such that
        it can be directly copied into
        ``lpaprogram.LPA.intensity[:,row,col,channel]``.

        Parameters
        ----------
        dose_idx : int
            Dose for which to generate the intensity sequence.

        Returns
        -------
        array
            Array with `n_time_steps` intensities to be recorded in a
            LPA program for dose `dose_idx`.

        """
        raise NotImplementedError


class LightInducer(LPAInducerBase):
    """
    Object that represents fixed light intensities from LPA LEDs.

    Parameters
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str, optional
        Units in which light intensity is expressed.
    led_layout : str, optional
        Name of the LED layout associated with this inducer. A layout
        describes a mapping from LED types to each well of an arbitrary
        LPA, without reference to a specific LPA or LEDs. An Excel file
        mapping LED layouts to calibrated LED sets must be specified to the
        ``LPA-Program`` module. For more information, refer to
        ``LPA-Program``'s documentation. The LED layout name can be
        specified during the object's creation, or sometime before
        generating the experiment files.
    led_channel : int, optional
        The LED channel used by the inducer in an LPA. This can be
        specified during the object's creation, or sometime before
        generating the experiment files.
    id_prefix : str, optional
        Prefix to be used for the ID that identifies each inducer dose.
        If None, use the first letter of the inducer's name.
    id_offset : int, optional
        Offset from which to generate the ID that identifies each inducer
        dose. Default: 0 (no offset).

    Attributes
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str
        Units in which light intensity is expressed.
    led_layout : str
        Name of the LED layout associated with this inducer.
    led_channel : int, optional
        The LED channel used by the inducer in an LPA.
    id_prefix : str
        Prefix to be used for the ID that identifies each inducer dose.
    id_offset : int
        Offset from which to generate the ID that identifies each inducer
        dose.
    intensities : array
        Inducer light intensities.
    doses_table : DataFrame
        Table containing information of all the inducer intensities.

    Other Attributes
    ----------------
    shuffling_enabled : bool
        Whether shuffling of the doses table is enabled. If False, the
        `shuffle` function will not change the order of the rows in the
        doses table.
    shuffled_idx : list
        Randomized indices that result in the current shuffling of
        doses.
    shuffling_sync_list : list
        List of inducers with which shuffling should be synchronized.
    time_step_size : int
        Number of milliseconds in each time step.
    time_step_units : str
        Specific name of each time step (e.g. minute), to be used in
        generated files.
    n_time_steps : int
        Number of time steps in the LPA program.

    """
    def __init__(self,
                 name,
                 units=u'µmol/(m^2*s)',
                 led_layout=None,
                 led_channel=None,
                 id_prefix=None,
                 id_offset=0):
        # Parent's __init__ stores name, units, led_layout, and led_channel,
        # initializes doses table, and sets shuffling and time step parameters.
        super(LightInducer, self).__init__(name=name,
                                           units=units,
                                           led_layout=led_layout,
                                           led_channel=led_channel)

        # Renitialize time step attributes
        # Default: one minute step
        self.time_step_size = 1000*60
        self.time_step_units = 'min'

        # Store ID modifiers for dose table
        if id_prefix is None:
            id_prefix=name[0]
        self.id_prefix=id_prefix
        self.id_offset=id_offset

        # Initialize empty list of doses
        self.intensities = []

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
            self.intensities = numpy.linspace(min, max, n//n_repeat)
        elif scale == 'log':
            if use_zero:
                self.intensities = numpy.logspace(numpy.log10(min),
                                                  numpy.log10(max),
                                                  (n//n_repeat - 1))
                self.intensities = \
                    numpy.concatenate(([0], self.intensities))
            else:
                self.intensities = numpy.logspace(numpy.log10(min),
                                                  numpy.log10(max),
                                                  n//n_repeat)
        else:
            raise ValueError("scale {} not recognized".format(scale))

        # Repeat if necessary
        self.intensities = numpy.repeat(self.intensities, n_repeat)

    def get_lpa_intensity(self, dose_idx):
        """
        Get the LPA intensity sequence for a specified dose.

        Returns a `n_time_steps`-long sequence with all elements set to the
        intensity of dose `dose_idx`.

        Parameters
        ----------
        dose_idx : int
            Dose for which to generate the intensity sequence.

        Returns
        -------
        array
            Array with `n_time_steps` intensities to be recorded in a
            LPA program for dose `dose_idx`.

        """
        if self.n_time_steps is None:
            raise ValueError('number of time steps should be indicated')
        return numpy.ones(self.n_time_steps)*self.intensities[dose_idx]

class StaggeredLightSignal(LPAInducerBase):
    """
    Object that represents a time-varying staggered light signal.

    This inducer applies a time-varying light signal to many samples using
    the "staggered sampling method" (see Notes for details).
    Correspondingly, each dose of this inducer is associated with a
    sampling time. A separate file, created during the Experiment Setup
    phase, contains the signal values over time. The dose table contains
    only the dose's sampling time and the name of the file with the signal
    values.

    Parameters
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str, optional
        Units in which light intensity is expressed.
    led_layout : str, optional
        Name of the LED layout associated with this inducer. A layout
        describes a mapping from LED types to each well of an arbitrary
        LPA, without reference to a specific LPA or LEDs. An Excel file
        mapping LED layouts to calibrated LED sets must be specified to the
        ``LPA-Program`` module. For more information, refer to
        ``LPA-Program``'s documentation. The LED layout name can be
        specified during the object's creation, or sometime before
        generating the experiment files.
    led_channel : int, optional
        The LED channel used by the inducer in an LPA. This can be
        specified during the object's creation, or sometime before
        generating the experiment files.
    id_prefix : str, optional
        Prefix to be used for the ID that identifies each inducer dose.
        If None, use the first letter of the inducer's name.
    id_offset : int, optional
        Offset from which to generate the ID that identifies each inducer
        dose. Default: 0 (no offset).

    Attributes
    ----------
    name : str
        Name of the inducer, to be used in generated files.
    units : str
        Units in which light intensity is expressed.
    led_layout : str
        Name of the LED layout associated with this inducer. 
    led_channel : int, optional
        The LED channel used by the inducer in an LPA.
    id_prefix : str
        Prefix to be used for the ID that identifies each inducer dose.
    id_offset : int
        Offset from which to generate the ID that identifies each inducer
        dose.
    signal : array
        The light signal is specified as an array of light intensities to
        apply on every time step (see attributes `time_step_size` and
        `time_step_units`).
    signal_init
        Constant light intensity to hold before the beginning of the light
        signal.
    sampling_time_steps
        Sampling times, in time steps (see attributes `time_step_size` and
        `time_step_units`).
    doses_table : DataFrame
        Table containing sampling time information.
    n_time_steps : int
        Number of time steps in the LPA program.

    Other Attributes
    ----------------
    shuffling_enabled : bool
        Whether shuffling of the doses table is enabled. If False, the
        `shuffle` function will not change the order of the rows in the
        doses table.
    shuffled_idx : list
        Randomized indices that result in the current shuffling of
        doses.
    shuffling_sync_list : list
        List of inducers with which shuffling should be synchronized.
    time_step_size : int
        Number of milliseconds in each time step. Default: 60000.
    time_step_units : str
        Specific name of each time step (e.g. minute), to be used in
        generated files. Default: 'min'.

    Notes
    -----
    In order to measure a genetic system's time response to a time-varying
    inducer signal, samples have to be taken out of the cell culture at
    different times. If the culture is a bacterial culture in exponential
    phase, this is equivalent to applying the inducer signal to multiple
    samples in a fixed-time experiment, with a different time shift each.
    This time shift is such that at the end of the experiment each sample
    has been exposed to the signal only up to the sampling time. More
    formally, the response of the cell culture to a signal of duration
    ``t_signal`` at time ``ts_i < t_signal`` in an experiment of duration
    ``t_exp`` is obtained by growing sample ``i`` under some initial fixed
    signal value until ``t_exp - ts_i``, and then exposing the sample to
    the signal from the beginning until time ``ts_i``. This is referred to
    as the "staggered sampling method".

    """
    def __init__(self,
                 name,
                 units=u'µmol/(m^2*s)',
                 led_layout=None,
                 led_channel=None,
                 id_prefix=None,
                 id_offset=0):
        # Parent's __init__ stores name, units, led_layout, and led_channel,
        # initializes doses table, and sets shuffling parameters.
        super(StaggeredLightSignal, self).__init__(name=name,
                                                   units=units,
                                                   led_layout=led_layout,
                                                   led_channel=led_channel)

        # Renitialize time step attributes
        # Default: one minute step
        self.time_step_size = 1000*60
        self.time_step_units = 'min'

        # Light signal
        # Each value is the LED intensity for a time step.
        self.signal = numpy.array([])
        # Light intensity before the beginning of the signal.
        self.signal_init = 0.

        # Store ID modifiers for dose table
        if id_prefix is None:
            id_prefix=name[0]
        self.id_prefix=id_prefix
        self.id_offset=id_offset

        # Initialize empty list of doses
        self.sampling_time_steps = []

    @property
    def _sampling_time_steps_header(self):
        """
        Header used in the dose table to specify signal sampling times.

        """
        return u"{} Sampling Time ({})".format(self.name, self.time_step_units)

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
    def sampling_time_steps(self):
        """
        Sampling times.

        Reading from this attribute will return the contents of the
        "Sampling Times" column from the dose table. Writing to this
        attribute will reinitialize the doses table with the specified
        sampling times. Any columns that are not the sampling times or IDs
        will be lost.

        """
        return self.doses_table[self._sampling_time_steps_header].values

    @sampling_time_steps.setter
    def sampling_time_steps(self, value):
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
        self._doses_table[self._sampling_time_steps_header] = value

    def set_step(self, initial, final, n_time_steps=None):
        """
        Set a step light signal.

        Parameters
        ----------
        initial : float
            Intensity value before the step signal.
        final : float
            Intensity value after the step signal.
        n_time_steps : int, optional
            Signal length, in time steps. If not specified, use the largest
            sampling time step.

        """
        if (n_time_steps is None) and (not self.sampling_time_steps.size):
            raise ValueError('n_time_steps or sampling time steps should be '
                'specified')
        # Calculate number of steps if necessary
        if n_time_steps is None:
            n_time_steps = numpy.max(self.sampling_time_steps)
        # Initial light intensity
        self.signal_init = initial
        # Signal will be constant, up to the largest sampling time
        self.signal = numpy.ones(n_time_steps, dtype=float) * final

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
        time_header = u'Time ({})'.format(self.time_step_units)
        signal_table = pandas.DataFrame(columns=[time_header, intensity_header])

        # Time values for the signal go from zero to the length of the signal
        # in time steps.
        # The initial intensity is considered to be set at time = -inf
        signal_table[time_header] = \
            numpy.append([-numpy.inf], range(len(self.signal)))

        signal_table[intensity_header] = numpy.append(
            [self.signal_init], self.signal)

        # Time should be the index
        signal_table.set_index(time_header, inplace=True)

        # Save DataFrame
        writer = pandas.ExcelWriter(os.path.join(path, self._signal_file_name),
                                    engine='openpyxl')
        signal_table.to_excel(writer)
        writer.save()

    def get_lpa_intensity(self, dose_idx):
        """
        Get the LPA intensity sequence for a specified dose.

        This function returns the fully resolved sequence of light
        intensities such that it can be directly copied into
        ``lpaprogram.LPA.intensity[:,row,col,channel]``.

        Parameters
        ----------
        dose_idx : int
            Dose for which to generate the intensity sequence.
        n_steps : int
            Number of steps in the sequence.

        Returns
        -------
        array
            `n_time_steps`-element array, with intensities to be recorded
            in a LPA program for dose `dose_idx` over time.

        """
        if self.n_time_steps is None:
            raise ValueError('number of time steps should be indicated')
        # Get sampling time
        ts = self.sampling_time_steps[dose_idx]
        # Assemble intensity sequence
        intensity = numpy.ones(self.n_time_steps)*self.signal_init
        if (ts > 0) and (ts <= self.n_time_steps):
            # Case 1: sampling time less or equal to total time
            # Copy light signal up to ts to the end of intensity array
            intensity[-ts:] = self.signal[0: ts]
        elif (ts > 0) and (ts > self.n_time_steps):
            # Case 2: sampling time greater than total time
            # Copy a self.n_time_steps-long fragment of signal up to ts.
            intensity = self.signal[ts - self.n_time_steps:ts]

        return intensity
