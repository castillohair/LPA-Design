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

class LightSignal(LPAInducerBase):
    """
    Object that represents time-varying light intensities from LPA LEDs.

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
        Light intensities, as a 2D ``n_time_steps * n_doses`` array.
    signal_labels: array
        Labels assigned to each signal (dose).
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
        Number of time steps currently in `intensity`. Read-only.

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
        super(LightSignal, self).__init__(name=name,
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

        # Initialize empty array of intensities
        self.intensities = numpy.empty((0,)*2)

    @property
    def n_time_steps(self):
        """
        Number of time steps currently in `intensity`.

        Read-only attribute. Writing to this attribute has no effect.

        """
        return self._n_time_steps

    @n_time_steps.setter
    def n_time_steps(self, value):
        return

    @property
    def _signal_labels_header(self):
        """
        Header to be used in the dose table to specify intensities.

        """
        return u"{} Signal Label".format(self.name)

    @property
    def signal_labels(self):
        """
        Labels assigned to each signal (dose).

        Reading from this attribute will return the contents of the
        Signal Label columns from the dose table. Writing to this attribute
        will directly write into the Signal Label column. Note that writing
        to `intensities` will delete all signal labels.

        """
        return self.doses_table[self._signal_labels_header].values

    @signal_labels.setter
    def signal_labels(self, value):
        # Make sure that value is a list with the appropriate length
        if len(value) != len(self._doses_table):
            raise ValueError("signal_labels should have a length of {}".format(
                len(self._doses_table)))
        # Assign values
        self._doses_table[self._signal_labels_header] = value

    @property
    def _intensities_headers(self):
        """
        Headers to be used in the dose table to specify intensities.

        """
        return [u"{} Intensity ({}) at t = {} {}".format(self.name,
                                                         self.units,
                                                         i,
                                                         self.time_step_units)
                for i in range(self.n_time_steps)]

    @property
    def intensities(self):
        """
        Light intensities, as a 2D ``n_time_steps * n_doses`` array.

        Reading from this attribute will return the contents of the
        intensities columns from the dose table. Writing to this attribute
        will reinitialize the doses table with the specified intensities.
        Any columns that are not the intensities or IDs will be lost.

        """
        return self.doses_table[self._intensities_headers].values.T

    @intensities.setter
    def intensities(self, value):
        # Make sure that value is a float 2D array
        value = numpy.array(value, dtype=numpy.float)
        if value.ndim != 2:
            raise ValueError("intensities should be a 2D array")
        # Initialize number of time steps
        self._n_time_steps = value.shape[0]
        # Initialize doses table
        self._doses_table = pandas.DataFrame(data=value.T)
        self._doses_table.columns = self._intensities_headers
        # Row IDs
        ids = ['{}{:03d}'.format(self.id_prefix, i)
               for i in range(self.id_offset + 1,
                              value.shape[1] + self.id_offset + 1)]
        self._doses_table['ID'] = ids
        self._doses_table.set_index('ID', inplace=True)
        # Empty signal labels
        self._doses_table[self._signal_labels_header] = [""]*value.shape[1]

    def set_staggered_signal(self,
                             signal,
                             sampling_time_steps,
                             n_time_steps,
                             signal_init=0,
                             set_signal_labels=True):
        """
        Set a time-varying staggered light signal.

        For each time ``ts`` specified in `sampling_time_steps`, this
        function constructs a light signal that contains the first ``ts``
        values of `signal`, preceeded by as many copies of `signal_init`
        as necessary such that the total light signal length is
        `n_time_steps`. These constructed signals are stored in the
        `intensities` attribute.

        Parameters
        ----------
        signal : array
            Light signal to stagger, as a list of intensity values applied
            at each time step.
        sampling_time_steps : array of int
            Sampling time steps.
        n_time_steps : int
            Total number of time steps in the signal.
        signal_init : float, optional
            Initial signal value.
        set_signal_labels : bool, optional
            If True, `.signal_labels` will be filled with strings of the
            form "Sampling time: 3 min".

        """
        # Check that signal has as many elements as the largest sampling time
        # step.
        if len(signal) < numpy.max(sampling_time_steps):
            raise ValueError("signal should have at least {} elements".format(
                numpy.max(sampling_time_steps)))
        # Initialize intensity array
        intensities = numpy.zeros((n_time_steps,
                                   len(sampling_time_steps)))
        # Iterate over every sampling time
        for ts_idx, ts in enumerate(sampling_time_steps):
            # Assemble intensity sequence
            intensity = numpy.ones(n_time_steps)*signal_init
            if (ts > 0) and (ts <= n_time_steps):
                # Case 1: sampling time less or equal to total time
                # Copy light signal up to ts to the end of intensity array
                intensity[-ts:] = signal[0: ts]
            elif (ts > 0) and (ts > n_time_steps):
                # Case 2: sampling time greater than total time
                # Copy a n_time_steps-long fragment of signal up to ts.
                intensity = signal[ts - n_time_steps:ts]
            # Store intensities
            intensities[:, ts_idx] = intensity
        # Assign to intensities array (dose table)
        self.intensities = intensities
        # Set signal labels if requested
        if set_signal_labels:
            self.signal_labels = ["Sampling time: {} {}".\
                                      format(ts, self.time_step_units)
                                  for ts in sampling_time_steps]

    def get_lpa_intensity(self, dose_idx):
        """
        Get the LPA intensity sequence for a specified dose.

        This function returns the fully resolved sequence of light
        intensities such that it can be directly copied into
        ``lpaprogram.LPA.intensity[:, row, col, channel]``.

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
        return self.intensities[:, dose_idx]
