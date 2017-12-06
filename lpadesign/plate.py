# -*- coding: UTF-8 -*-
"""
Module that contains the LPAPlate and LPAPlateArray classes.

"""

import os

import numpy

import platedesign
import platedesign.plate

import lpaprogram

import lpadesign
import lpadesign.inducer

class LPAPlate(platedesign.plate.Plate):
    """
    Object that represents a plate in an LPA.

    This class can manage all the chemical inducers in ``platedesign``, and
    all LPA inducers in the ``lpadesign.inducer`` module. Method
    ``save_rep_setup_files()`` saves a set of LPA files according to the
    specified LPA inducers, using the ``lpaprogram.LPA`` object in
    ``LPAPlate.lpa``.

    Parameters
    ----------
    name : str
        Name of the plate, to be used in generated files.
    n_rows, n_cols : int, optional
        Number of rows and columns in the plate. Defaults: 4 and 6.
    n_led_channels : int, optional
        Number of LEDs per well. Default: 2.

    Attributes
    ----------
    name : str
        Name of the plate, to be used in generated files.
    n_rows, n_cols : int
        Number of rows and columns in the plate.
    n_led_channels : int
        Number of LEDs per well.
    n_plates : int
        Number of physical plates handled by this object. Fixed to 1.
    samples_to_measure : int
        Number of samples to be measured.
    sample_media_vol : float
        Volume of media per sample (well).
    total_media_vol : float
        Starting total volume of media, to be distributed into wells.
    cell_strain_name : str
        Name of the cell strain to be inoculated in this plate.
    cell_setup_method : str or None
        Method used to determine how much volume of cells to inoculate. Can
        be one of the following: "fixed_od600", "fixed_volume", or
        "fixed_dilution".
    cell_predilution : float
        Dilution factor for the cell preculture/aliquot before inoculating.
    cell_predilution_vol : float
        Volume of diluted preculture/aliquot to make in µL.
    cell_od600_measure_from_dilution : bool
        If True, the OD600 of the diluted preculture/aliquot is measured
        and used to calculate volumes. If False, the OD600 of the undiluted
        preculture/aliquot is used instead. Only used if cell_setup_method`
        is "fixed_od600" and `cell_predilution` is greater than one.
        Default: True.
    cell_initial_od600 : float
        Target initial OD600 for inoculating cells. Only used if
        `cell_setup_method` is "fixed_od600".
    cell_shot_vol : float
        Volume of diluted preculture/aliquot to inoculate in media. Only
        used if `cell_setup_method` is "fixed_volume".
    cell_total_dilution : float
        Total dilution from preculture/aliquot to be inoculated in the
        media. Only used if `cell_setup_method` is "fixed_dilution".
    resources : OrderedDict
        Names of per-plate resources, in a ``key: value`` format, where
        ``value`` is a list of length ``n_plates``. The ClosedPlate
        instance returned by ``close_plates()`` will include this
        information in its ``samples_table`` attribute. In it, a column
        with name ``key`` will be created, and all rows will be set to
        ``value[0]``.
    metadata : OrderedDict
        Additional information about the plate, in a ``key: value`` format.
        The ClosedPlate instance returned by ``close_plates()`` will
        include this information in its ``samples_table`` attribute. In it,
        a column with name ``key`` will be created, and all rows will be
        set to ``value``.
    inducers : OrderedDict
        Keys in this dictionary represent how each inducer is applied
        ("rows", "cols", "wells", "media"), and the values are lists of
        inducers to be applied as specified by the key.
    lpa : lpaprogram.LPA
        LPA object, used to generate LPA files.
    lpa_optimize_dc : list of bool
        Each element indicates whether dot correction should be optimized
        on each LED channel when running ``save_rep_setup_files()``.
        Default: all True.
    lpa_optimize_dc_uniform : list of bool
        Each element indicates whether dot correction should be optimized
        uniformly on each LED channel when running
        ``save_rep_setup_files()``. Default: all True.
    lpa_end_with_leds_off : bool
        Whether to add an additional dark frame to the LPA program, or
        maintain the last frame on forever. Default: True.
    lpa_files_path : str
        Subfolder where to store LPA files when running
        ``save_rep_setup_files()``. Default: 'LPA Files'.

    """
    def __init__(self,
                 name,
                 n_rows=4,
                 n_cols=6,
                 n_led_channels=2):
        # Parent's __init__ stores name, dimensions, initializes samples to
        # measure, sample and total media volume, cell setup parameters,
        # resources, metadata, and the inducers dictionary.
        super(LPAPlate, self).__init__(name=name, n_rows=n_rows, n_cols=n_cols)

        # Store number of LED channels
        self.n_led_channels = n_led_channels

        # Initialize LPA object
        self.lpa = lpaprogram.LPA(n_rows=n_rows,
                                  n_cols=n_cols,
                                  n_channels=n_led_channels)

        # Default gcal and dc values
        self.lpa.set_all_gcal(255)
        self.lpa.set_all_dc(8)
        
        # Initialize LPA options
        self.lpa_optimize_dc = [True]*n_led_channels
        self.lpa_optimize_dc_uniform = [True]*n_led_channels
        self.lpa_end_with_leds_off = True
        self.lpa_files_path = 'LPA Files'

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        This function saves LPA files inside subfolder `lpa_files_path`,
        which is in turn created inside `path`.

        The name of the LPA, used to load calibration info, should be
        specified as a plate resource.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        Notes
        -----
        This function uses the LPA object in the ``lpa`` attribute to save
        the LPA files. To do so, the following is performed on ``lpa``:

        1. LPA name is obtained from the ``resources`` attribute, and LED
          layout names from LPA inducers. Calibration info is loaded.
        2. Time step and number of steps are obtained from the LPA
          inducers.
        3. The ``lpa.intensity`` array is filled from the inducers. If all
          intensities at all times are identical, the array is condensed
          into a single-frame program.
        4. An additional dark frame is added if ``lpa_end_with_leds_off``
          is True.
        5. Dot correction values for channel ``i`` are optimized if
          ``lpa_optimize_dc[i]`` is True, with the ``uniform`` option
          as given by ``lpa_optimize_dc_uniform[i]``.
        6. Intensities are discretized with ``lpa.discretize_intensity()``,
          and the LPA files are saved.

        Any modifications to ``lpa`` that interfere with this process are
        discouraged.

        """
        # Recover LPA inducers
        lpa_inducers = [None]*self.n_led_channels
        lpa_inducers_apply_to = [None]*self.n_led_channels
        for apply_to, inducers in self.inducers.iteritems():
            for inducer in inducers:
                if isinstance(inducer, lpadesign.inducer.LPAInducerBase):
                    # Check that channel is within the allowed number of
                    # channels.
                    if inducer.led_channel >= self.n_led_channels:
                        raise ValueError("inducer {} ".format(inducer.name) +\
                            "assigned to LED channel {} (zero-based), ".format(
                                inducer.led_channel) +\
                            "device only has {} channels".format(
                                self.n_led_channels))
                    # Check that no other inducer exists in channel
                    if lpa_inducers[inducer.led_channel] is not None:
                        raise ValueError("more than one LPA inducers assigned "
                            "to plate {}, LED channel {}".format(
                                self.name, inducer.led_channel))
                    # Store inducer
                    lpa_inducers[inducer.led_channel] = inducer
                    lpa_inducers_apply_to[inducer.led_channel] = apply_to

        # Save nothing if no LPA inducers have been found.
        if all(inducer is None for inducer in lpa_inducers):
            return

        # Create folder for LPA files if necessary
        if not os.path.exists(os.path.join(path, self.lpa_files_path)):
            os.makedirs(os.path.join(path, self.lpa_files_path))

        # Get LPA name from LPA resource
        self.lpa.name = self.resources['LPA'][0]

        # Load LED layout information
        led_layouts = [inducer.led_layout if inducer is not None
                       else None
                       for inducer in lpa_inducers]
        self.lpa.load_led_sets(layout_names=led_layouts)

        # Get time step attributes
        time_step_size_all = [inducer.time_step_size
                              for inducer in lpa_inducers
                              if inducer.time_step_size is not None]
        time_step_units_all = [inducer.time_step_units
                              for inducer in lpa_inducers
                              if inducer.time_step_units is not None]
        n_time_steps_all = [inducer.n_time_steps
                            for inducer in lpa_inducers
                            if inducer.n_time_steps is not None]
        # There should be at least one element in each list
        if not time_step_size_all:
            raise ValueError('time step size should be specified')
        if not time_step_units_all:
            raise ValueError('time step units should be specified')
        if not n_time_steps_all:
            raise ValueError('number of time steps should be specified')
        # All time step attributes should be identical
        if not all([t==time_step_size_all[0] for t in time_step_size_all]):
            raise ValueError('all time step sizes should be the same')
        if not all([t==time_step_units_all[0] for t in time_step_units_all]):
            raise ValueError('all time step units should be the same')
        if not all([t==n_time_steps_all[0] for t in n_time_steps_all]):
            raise ValueError('all number of time steps should be the same')
        # Set attributes in all inducers
        time_step_size = time_step_size_all[0]
        time_step_units = time_step_units_all[0]
        n_time_steps = n_time_steps_all[0]
        for inducer in inducers:
            inducer.time_step_size = time_step_size
            inducer.time_step_units = time_step_units
            inducer.n_time_steps = n_time_steps
        # Load time step attributes into LPA object
        self.lpa.step_size = time_step_size
        self.lpa.set_n_steps(n_time_steps)

        # Reset all intensities to zero
        self.lpa.intensity.fill(0.)

        # Fill light intensity array in LPA object
        for channel, (inducer, apply_to) in \
                enumerate(zip(lpa_inducers, lpa_inducers_apply_to)):
            # Do nothing if there is no inducer
            if inducer is None:
                continue
            # Fill intensity array
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    # Decide what to do based on how to apply the inducer
                    if apply_to == 'rows':
                        self.lpa.intensity[:,i,j,channel] = \
                            inducer.get_lpa_intensity(j)
                    elif apply_to == 'cols':
                        self.lpa.intensity[:,i,j,channel] = \
                            inducer.get_lpa_intensity(i)
                    elif apply_to == 'wells':
                        if (i*self.n_cols + j) < self.samples_to_measure:
                            self.lpa.intensity[:,i,j,channel] = \
                                inducer.get_lpa_intensity(i*self.n_cols + j)
                    elif apply_to == 'media':
                        if (i*self.n_cols + j) < self.samples_to_measure:
                            self.lpa.intensity[:,i,j,channel] = \
                                inducer.get_lpa_intensity(0)

        # Condense intensity array if all intensities over time are the same
        # This comparison will be performed by comparing all frames to the first
        # frame, and testing for all True.
        if numpy.all((self.lpa.intensity[0] == self.lpa.intensity)):
            self.lpa.set_n_steps(1)
            self.lpa.step_size = time_step_size*n_time_steps

        # Add additional frame at the end.
        # If `lpa_end_with_leds_off` is True, the last frame will have all the
        # lights off. Otherwise, the last frame will be copied and maintained.
        self.lpa.set_n_steps(self.lpa.intensity.shape[0] + 1)
        if self.lpa_end_with_leds_off:
            self.lpa.intensity[-1,:,:,:] = 0.

        # Optimize dc values if requested
        for channel, (optimize_dc, uniform) in \
                enumerate(zip(self.lpa_optimize_dc,
                              self.lpa_optimize_dc_uniform)):
            if optimize_dc:
                self.lpa.optimize_dc(channel=channel, uniform=uniform)

        # Discretize intensities
        self.lpa.discretize_intensity()

        # Save files
        self.lpa.save_files(path=os.path.join(path, self.lpa_files_path))

class LPAPlateArray(LPAPlate, platedesign.plate.PlateArray):
    """
    Object that represents a plate array in a set of LPAs.

    This class can manage all the chemical inducers in ``platedesign``, and
    all LPA inducers in the ``lpadesign.inducer`` module. Method
    ``save_rep_setup_files()`` saves a set of LPA files according to the
    specified LPA inducers, using the ``lpaprogram.LPA`` objects in
    ``LPAPlate.lpas``.

    Parameters
    ----------
    name : str
        Name of the plate array, to be used in generated files.
    array_n_rows, array_n_cols : int
        Number of rows and columns in the plate array.
    plate_names : list
        Names of the plates, to be used in generated files.
    plate_n_rows, plate_n_cols : int, optional
        Number of rows and columns in each plate. Defaults: 4 and 6.
    n_led_channels : int, optional
        Number of LEDs per well. Default: 2.

    Attributes
    ----------
    name : str
        Name of the plate, to be used in generated files.
    array_n_rows, array_n_cols : int
        Number of rows and columns in the plate array.
    plate_names : list
        Names of the plates, to be used in generated files.
    plate_n_rows, plate_n_cols : int
        Number of rows and columns in each plate.        
    n_rows, n_cols : int
        Total number of rows and columns in the plate array.
    n_led_channels : int
        Number of LEDs per well.
    n_plates : int
        Number of physical plates handled by this object. Returns
        ``array_n_rows * array_n_cols``.
    samples_to_measure : int
        Number of samples to be measured.
    sample_media_vol : float
        Volume of media per sample (well).
    total_media_vol : float
        Starting total volume of media, to be distributed into wells.
    cell_strain_name : str
        Name of the cell strain to be inoculated in this plate.
    cell_setup_method : str or None
        Method used to determine how much volume of cells to inoculate. Can
        be one of the following: "fixed_od600", "fixed_volume", or
        "fixed_dilution".
    cell_predilution : float
        Dilution factor for the cell preculture/aliquot before inoculating.
    cell_predilution_vol : float
        Volume of diluted preculture/aliquot to make in µL.
    cell_od600_measure_from_dilution : bool
        If True, the OD600 of the diluted preculture/aliquot is measured
        and used to calculate volumes. If False, the OD600 of the undiluted
        preculture/aliquot is used instead. Only used if cell_setup_method`
        is "fixed_od600" and `cell_predilution` is greater than one.
        Default: True.
    cell_initial_od600 : float
        Target initial OD600 for inoculating cells. Only used if
        `cell_setup_method` is "fixed_od600".
    cell_shot_vol : float
        Volume of diluted preculture/aliquot to inoculate in media. Only
        used if `cell_setup_method` is "fixed_volume".
    cell_total_dilution : float
        Total dilution from preculture/aliquot to be inoculated in the
        media. Only used if `cell_setup_method` is "fixed_dilution".
    resources : OrderedDict
        Names of per-plate resources, in a ``key: value`` format, where
        ``value`` is a list of length ``n_plates``. The ClosedPlate
        instance returned by ``close_plates()`` will include this
        information in its ``samples_table`` attribute. In it, a column
        with name ``key`` will be created, and all rows will be set to
        the element of ``value`` corresponding to the specific plate.
    metadata : OrderedDict
        Additional information about the plate, in a ``key: value`` format.
        The ClosedPlate instance returned by ``close_plates()`` will
        include this information in its ``samples_table`` attribute. In it,
        a column with name ``key`` will be created, and all rows will be
        set to ``value``.
    inducers : OrderedDict
        Keys in this dictionary represent how each inducer is applied
        ("rows", "cols", "wells", "media"), and the values are lists of
        inducers to be applied as specified by the key.
    lpas : list of lpaprogram.LPA
        LPA objects, used to generate LPA files.
    lpa_optimize_dc : list of bool
        Each element indicates whether dot correction should be optimized
        on each LED channel when running ``save_rep_setup_files()``.
        Default: all True.
    lpa_optimize_dc_uniform : list of bool
        Each element indicates whether dot correction should be optimized
        uniformly on each LED channel when running
        ``save_rep_setup_files()``. Default: all True.
    lpa_end_with_leds_off : bool
        Whether to add an additional dark frame to the LPA program, or
        maintain the last frame on forever. Default: True.
    lpa_files_path : str
        Subfolder where to store LPA files when running
        ``save_rep_setup_files()``. Default: 'LPA Files'.

    """
    def __init__(self,
                 name,
                 array_n_rows,
                 array_n_cols,
                 plate_names,
                 plate_n_rows=4,
                 plate_n_cols=6,
                 n_led_channels=2):
        # Parent's __init__ stores name, dimensions, plate names, initializes
        # samples to measure, sample and total media volume, cell setup
        # parameters, resources, metadata, and the inducers dictionary.
        platedesign.plate.PlateArray.__init__(self,
                                              name=name,
                                              array_n_rows=array_n_rows,
                                              array_n_cols=array_n_cols,
                                              plate_names=plate_names,
                                              plate_n_rows=plate_n_rows,
                                              plate_n_cols=plate_n_cols,)

        # Store number of LED channels
        self.n_led_channels = n_led_channels

        # Initialize LPA objects
        self.lpas = []
        for i in range(array_n_rows*array_n_cols):
            lpa = lpaprogram.LPA(n_rows=plate_n_rows,
                                 n_cols=plate_n_cols,
                                 n_channels=n_led_channels)
            # Default gcal and dc values
            lpa.set_all_gcal(255)
            lpa.set_all_dc(8)
            # Store
            self.lpas.append(lpa)
        
        # Initialize LPA options
        self.lpa_optimize_dc = [True]*n_led_channels
        self.lpa_optimize_dc_uniform = [True]*n_led_channels
        self.lpa_end_with_leds_off = True
        self.lpa_files_path = 'LPA Files'

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        This function saves LPA files inside subfolder `lpa_files_path`,
        which is in turn created inside `path`.

        The name of the LPAs, used to load calibration info, should be
        specified as a plate resource.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        Notes
        -----
        This function uses the LPA objects in the ``lpas`` attribute to
        save the LPA files. To do so, the following is performed:

        1. LPA name is obtained from the ``resources`` attribute, and LED
          layout names from LPA inducers. Calibration info is loaded.
        2. Time step and number of steps are obtained from the LPA
          inducers.
        3. The ``lpa.intensity`` array is filled from the inducers. If all
          intensities at all times are identical, the array is condensed
          into a single-frame program.
        4. An additional dark frame is added if ``lpa_end_with_leds_off``
          is True.
        5. Dot correction values for channel ``i`` are optimized if
          ``lpa_optimize_dc[i]`` is True, with the ``uniform`` option
          as given by ``lpa_optimize_dc_uniform[i]``.
        6. Intensities are discretized with ``lpa.discretize_intensity()``,
          and the LPA files are saved.

        Any modifications to ``lpas`` that interfere with this process are
        discouraged.

        """
        # Recover LPA inducers
        lpa_inducers = [None]*self.n_led_channels
        lpa_inducers_apply_to = [None]*self.n_led_channels
        for apply_to, inducers in self.inducers.iteritems():
            for inducer in inducers:
                if isinstance(inducer, lpadesign.inducer.LPAInducerBase):
                    # Check that channel is within the allowed number of
                    # channels.
                    if inducer.led_channel >= self.n_led_channels:
                        raise ValueError("inducer {} ".format(inducer.name) +\
                            "assigned to LED channel {} (zero-based), ".format(
                                inducer.led_channel) +\
                            "device only has {} channels".format(
                                self.n_led_channels))
                    # Check that no other inducer exists in channel
                    if lpa_inducers[inducer.led_channel] is not None:
                        raise ValueError("more than one LPA inducers assigned "
                            "to plate {}, LED channel {}".format(
                                self.name, inducer.led_channel))
                    # Store inducer
                    lpa_inducers[inducer.led_channel] = inducer
                    lpa_inducers_apply_to[inducer.led_channel] = apply_to

        # Save nothing if no LPA inducers have been found.
        if all(inducer is None for inducer in lpa_inducers):
            return

        # Create folder for LPA files if necessary
        if not os.path.exists(os.path.join(path, self.lpa_files_path)):
            os.makedirs(os.path.join(path, self.lpa_files_path))

        # Get LED layout information
        led_layouts = [inducer.led_layout if inducer is not None
                       else None
                       for inducer in lpa_inducers]

        # Get time step attributes
        time_step_size_all = [inducer.time_step_size
                              for inducer in lpa_inducers
                              if inducer.time_step_size is not None]
        time_step_units_all = [inducer.time_step_units
                              for inducer in lpa_inducers
                              if inducer.time_step_units is not None]
        n_time_steps_all = [inducer.n_time_steps
                            for inducer in lpa_inducers
                            if inducer.n_time_steps is not None]
        # There should be at least one element in each list
        if not time_step_size_all:
            raise ValueError('time step size should be specified')
        if not time_step_units_all:
            raise ValueError('time step units should be specified')
        if not n_time_steps_all:
            raise ValueError('number of time steps should be specified')
        # All time step attributes should be identical
        if not all([t==time_step_size_all[0] for t in time_step_size_all]):
            raise ValueError('all time step sizes should be the same')
        if not all([t==time_step_units_all[0] for t in time_step_units_all]):
            raise ValueError('all time step units should be the same')
        if not all([t==n_time_steps_all[0] for t in n_time_steps_all]):
            raise ValueError('all number of time steps should be the same')
        # Set attributes in all inducers
        time_step_size = time_step_size_all[0]
        time_step_units = time_step_units_all[0]
        n_time_steps = n_time_steps_all[0]
        for inducer in inducers:
            inducer.time_step_size = time_step_size
            inducer.time_step_units = time_step_units
            inducer.n_time_steps = n_time_steps

        # Set LPA name, load LED layouts, set step size, and empty intensity array
        for lpa_idx, lpa in enumerate(self.lpas):
            # Set name from plate resources
            lpa.name = self.resources['LPA'][lpa_idx]

            # Load LED layout
            lpa.load_led_sets(layout_names=led_layouts)

            # Load time step attributes into LPA object
            lpa.step_size = time_step_size
            lpa.set_n_steps(n_time_steps)

            # Reset all intensities to zero
            lpa.intensity.fill(0.)

        # Fill light intensity array in LPA object
        for channel, (inducer, apply_to) in \
                enumerate(zip(lpa_inducers, lpa_inducers_apply_to)):
            # Do nothing if there is no inducer
            if inducer is None:
                continue
            # Fill intensity array
            for array_i in range(self.array_n_rows):
                for array_j in range(self.array_n_cols):
                    array_k = array_i*self.array_n_cols + array_j
                    for plate_i in range(self.plate_n_rows):
                        for plate_j in range(self.plate_n_cols):
                            i = array_i*self.plate_n_rows + plate_i
                            j = array_j*self.plate_n_cols + plate_j
                            k = i*self.n_cols + j
                            if apply_to == 'rows':
                                self.lpas[array_k].\
                                    intensity[:, plate_i, plate_j, channel]\
                                    = inducer.get_lpa_intensity(j)
                            elif apply_to == 'cols':
                                self.lpas[array_k].\
                                    intensity[:, plate_i, plate_j, channel]\
                                    = inducer.get_lpa_intensity(i)
                            elif apply_to == 'wells':
                                if k < self.samples_to_measure:
                                    self.lpas[array_k].\
                                        intensity[:, plate_i, plate_j, channel]\
                                        = inducer.get_lpa_intensity(k)
                            elif apply_to == 'media':
                                if k < self.samples_to_measure:
                                    self.lpas[array_k].\
                                        intensity[:, plate_i, plate_j, channel]\
                                        = inducer.get_lpa_intensity(0)

        # Do post-processing and save LPA files
        for lpa in self.lpas:
            # Condense intensity array if all intensities over time are the same
            # This comparison will be performed by comparing all frames to the
            # first frame, and testing for all True.
            if numpy.all((lpa.intensity[0] == lpa.intensity)):
                lpa.set_n_steps(1)
                lpa.step_size = time_step_size*n_time_steps

            # Add additional frame at the end.
            # If `lpa_end_with_leds_off` is True, the last frame will have all
            # the lights off. Otherwise, the last frame will be copied and
            # maintained.
            lpa.set_n_steps(lpa.intensity.shape[0] + 1)
            if self.lpa_end_with_leds_off:
                lpa.intensity[-1,:,:,:] = 0.

            # Optimize dc values if requested
            for channel, (optimize_dc, uniform) in \
                    enumerate(zip(self.lpa_optimize_dc,
                                  self.lpa_optimize_dc_uniform)):
                if optimize_dc:
                    lpa.optimize_dc(channel=channel, uniform=uniform)

            # Discretize intensities
            lpa.discretize_intensity()

            # Save files
            lpa.save_files(path=os.path.join(path, self.lpa_files_path))
