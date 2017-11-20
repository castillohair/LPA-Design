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

    This class manages light inducers defined in ``lpadesign.inducers``.
    In addition, this class inherits from ``platedesign.Plate``. As such,
    it has the ability to manage all the chemical inducers in
    ``platedesign``.

    """
    def __init__(self,
                 name,
                 n_rows=4,
                 n_cols=6,
                 n_led_channels=2):

        # Store number of LED channels
        self.n_led_channels = n_led_channels
        # Initialize list of light inducers
        self.light_inducers = [None]*n_led_channels
        # Initialize LPA object
        self.lpa = lpaprogram.LPA(n_rows=n_rows,
                                  n_cols=n_cols,
                                  n_channels=n_led_channels)

        # Initialize LPA options
        # Default gcal and dc values
        self.lpa.set_all_gcal(255)
        self.lpa.set_all_dc(8)
        
        # Options to optimize the dot correction values
        self.lpa_optimize_dc = [True]*self.lpa.n_channels
        self.lpa_optimize_dc_uniform = [True]*self.lpa.n_channels

        # Default program duration is eight hours
        self.lpa_program_duration = 8*60
        # Add additional dark frame at the end
        self.lpa_end_with_leds_off = True
        # Folder for LPA files during replicate setup
        self.lpa_files_folder = 'LPA Files'

        # Initialize container for closed plates
        self.closed_plates = [None]

        # Call Plate's constructor
        super(LPAPlate, self).__init__(name=name, n_rows=n_rows, n_cols=n_cols)

    def apply_inducer(self, inducer, apply_to='wells', led_channel=None):
        """
        Apply an inducer to the plate.

        This function stores the specified inducer in the `inducers`
        attribute, after verifying consistency.

        Parameters
        ----------
        inducer : Inducer object
            The inducer to apply to the plate.
        apply_to : {'rows', 'cols', 'wells', 'media'}
            "rows" applies the specified inducer to all rows equally.
            "cols" applies to all columns equally. "wells" applies to each
            well individually. 'media' applies inducer to the media at the
            beginning of the replicate setup stage. 'media' is not allowed
            if ``inducer`` is a LightInducer.
        led_channel : int
            The LED channel used by a light inducer. Only necessary if
            `inducer` is a light inducer (i.e. object from a class in
            ``lpadesign.inducer``). Should be lower than attribute
            ``n_led_channels``.

        """
        # Check "apply_to" input
        if apply_to not in ['rows', 'cols', 'wells', 'media']:
            raise ValueError('"{}"" not recognized'.format(apply_to))
        # Only "wells" or "media" supported if not measuring full plate
        if (self.samples_to_measure != self.n_cols*self.n_rows) and\
                (apply_to not in ['wells', 'media']):
            raise ValueError('"{}"" not possible if not measuring all wells'.\
                format(apply_to))

        # Check that the inducer has the appropriate number of samples
        n_wells = self.samples_to_measure
        if (apply_to=='rows' and (len(inducer.doses_table)!=self.n_cols)) or \
           (apply_to=='cols' and (len(inducer.doses_table)!=self.n_rows)) or \
           (apply_to=='wells' and (len(inducer.doses_table)!=n_wells)) or \
           (apply_to=='media' and (len(inducer.doses_table)!=1)):
                raise ValueError('inducer does not have the appropriate' + \
                    ' number of doses')

        # Check that the inducer is not repeated
        if inducer in self.inducers[apply_to]:
            raise ValueError("inducer already in plate's inducer list")

        # Check that, if light inducer, led_channel is specified and valid, and
        # that apply_to is not 'media'
        if isinstance(inducer, lpadesign.inducer.LightInducer):
            if led_channel is None:
                raise ValueError("led_channel should be specified")
            if led_channel < 0 or led_channel >= self.n_led_channels:
                raise ValueError("led_channel should be between 0 and {}".\
                    format(self.n_led_channels - 1))
            if apply_to == 'media':
                raise ValueError('apply_to="media" not possible for inducer of '
                    'type {}'.format(type(inducer)))

        # Store inducer
        self.inducers[apply_to].append(inducer)
        if isinstance(inducer, lpadesign.inducer.LightInducer):
            self.light_inducers[led_channel] = inducer

    def close_plates(self):
        """
        Generate ``ClosedPlate`` objects using this plate's information.

        The individual ``ClosedPlate`` instances contain general plate
        information such as plate name, dimensions, cell inoculation
        conditions, and metadata, as well as well-specific information such
        as inducer concentrations. All this info is generated when calling
        `close_plates()`, and will remain fixed even after modifying
        inducers or other information in the ``PlateArray`` object.

        Within an experiment workflow, this function is meant to be called
        at the end of the Replicate Setup stage.

        Return
        ------
        list
            ``ClosedPlate`` instances with information about each sample.
            This list will only contain one ``ClosedPlate`` instance, as
            ``Plate`` represents a single plate.

        """
        # Call parent method, save list of closed plates, and return list.
        self.closed_plates = super(LPAPlate, self).close_plates()
        return self.closed_plates

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Create folder for LPA files if necessary
        if not os.path.exists(os.path.join(path, self.lpa_files_folder)):
            os.makedirs(os.path.join(path, self.lpa_files_folder))

        # Get LPA name from closed plate location
        self.lpa.name = self.closed_plates[0].plate_info['Location']
        # Get the led layout names from light inducers
        led_layouts = []
        for light_inducer in self.light_inducers:
            if light_inducer is not None:
                led_layouts.append(light_inducer.led_layout)
            else:
                led_layouts.append(None)
        # Load LED set information
        self.lpa.load_led_sets(layout_names=led_layouts)

        # Decide number of frames based on type of light inducers.
        # LightInducer are steady state inducers, and only need one frame.
        # For anything else, we will generate one frame per minute.
        steady_state = numpy.all([isinstance(l, lpadesign.inducer.LightInducer)
                                  for l in self.light_inducers])

        if steady_state:
            # Step size is the duration of the whole experiment
            self.lpa.step_size = 1000*60*self.lpa_program_duration
            self.lpa.set_n_steps(1)
        else:
            self.lpa.step_size = 1000*60
            self.lpa.set_n_steps(self.lpa_program_duration)

        # Fill light intensity array in LPA object
        for channel, inducer in enumerate(self.light_inducers):
            if isinstance(inducer, lpadesign.inducer.LightInducer):
                # Light inducer
                # Steady state inducer
                # Intensities per well should be resolved in closed plate
                intensities = self.closed_plates[0].well_info[
                    inducer._intensities_header].values
                # 'NaN' values are registered for wells that are not to be
                # measured. Switch these to zero.
                intensities[numpy.isnan(intensities)] = 0
                # Resize and add to all frames of intensity array in LPA object.
                intensities.resize(1, self.n_rows, self.n_cols)
                self.lpa.intensity[:,:,:,channel] = intensities.repeat(
                    self.lpa.intensity.shape[0], axis=0)
            else:
                raise NotImplementedError

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
        
        # Feed back to closed plates
        # Only necessary for steady state light inducers
        for channel, inducer in enumerate(self.light_inducers):
            if isinstance(inducer, lpadesign.inducer.LightInducer):
                intensities = self.lpa.intensity[0,:,:,channel].flatten()
                # Restore nan values on wells that will not be measured.
                intensities[self.closed_plates[0].well_info['Measure']==False] \
                    = numpy.nan
                # Update closed plate
                self.closed_plates[0].well_info[inducer._intensities_header] = \
                    intensities

        # Save files
        self.lpa.save_files(path=os.path.join(path, self.lpa_files_folder))

class LPAPlateArray(LPAPlate, platedesign.plate.PlateArray):
    """
    Object that represents a plate array in a set of LPAs.

    This class manages light inducers defined in ``lpadesign.inducers``.
    In addition, this class inherits from ``platedesign.PlateArray``. As
    such, it has the ability to manage all the chemical inducers in
    ``platedesign``.

    """
    def __init__(self,
                 name,
                 array_n_rows,
                 array_n_cols,
                 plate_names,
                 plate_n_rows=4,
                 plate_n_cols=6,
                 n_led_channels=2):

        # Store number of LED channels
        self.n_led_channels = n_led_channels
        # Initialize list of light inducers
        self.light_inducers = [None]*n_led_channels
        # Initialize LPA objects
        self.lpas = []
        for i in range(array_n_rows*array_n_cols):
            lpa = lpaprogram.LPA(n_rows=plate_n_rows,
                                 n_cols=plate_n_cols,
                                 n_channels=n_led_channels)
            # Initialize LPA options
            # Default gcal and dc values
            lpa.set_all_gcal(255)
            lpa.set_all_dc(8)
            # Store
            self.lpas.append(lpa)
        
        # Options to optimize the dot correction values
        self.lpa_optimize_dc = [True]*n_led_channels
        self.lpa_optimize_dc_uniform = [True]*n_led_channels

        # Default program duration is eight hours
        self.lpa_program_duration = 8*60
        # Add additional dark frame at the end
        self.lpa_end_with_leds_off = True
        # Folder for LPA files during replicate setup
        self.lpa_files_folder = 'LPA Files'

        # Initialize container for closed plates
        self.closed_plates = [None]*(array_n_rows*array_n_cols)

        # Call Plate's constructor
        platedesign.plate.PlateArray.__init__(self,
                                              name=name,
                                              array_n_rows=array_n_rows,
                                              array_n_cols=array_n_cols,
                                              plate_names=plate_names,
                                              plate_n_rows=plate_n_rows,
                                              plate_n_cols=plate_n_cols,)

    def close_plates(self):
        """
        Generate ``ClosedPlate`` objects for each plate in the array.

        The individual ``ClosedPlate`` instances contain general plate
        information such as plate name, dimensions, cell inoculation
        conditions, and metadata, as well as well-specific information such
        as inducer concentrations. All this info is generated when calling
        `close_plates()`, and will remain fixed even after modifying
        inducers or other information in the ``PlateArray`` object.

        Within an experiment workflow, this function is meant to be called
        at the end of the Replicate Setup stage.

        Return
        ------
        list
            ``ClosedPlate`` instances with information about each sample.
            The number of closed plates in this list is the number of
            plates in the array, i.e., ``array_n_rows * array_n_cols``.

        """
        # Call parent method, save list of closed plates, and return list.
        self.closed_plates = super(LPAPlateArray, self).close_plates()
        return self.closed_plates

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Create folder for LPA files if necessary
        if not os.path.exists(os.path.join(path, self.lpa_files_folder)):
            os.makedirs(os.path.join(path, self.lpa_files_folder))

        # Get the led layout names from light inducers
        led_layouts = []
        for light_inducer in self.light_inducers:
            if light_inducer is not None:
                led_layouts.append(light_inducer.led_layout)
            else:
                led_layouts.append(None)

        # Decide number of frames based on type of light inducers.
        # LightInducer are steady state inducers, and only need one frame.
        # For anything else, we will generate one frame per minute.
        steady_state = numpy.all([isinstance(l, lpadesign.inducer.LightInducer)
                                  for l in self.light_inducers])
        if steady_state:
            # Step size is the duration of the whole experiment
            lpa_step_size = 1000*60*self.lpa_program_duration
            lpa_n_steps = 1
        else:
            lpa_step_size = 1000*60
            lpa_n_steps = self.lpa_program_duration

        for lpa, closed_plate in zip(self.lpas, self.closed_plates):
            # Get LPA name from closed plate location
            lpa.name = closed_plate.plate_info['Location']
            # Load LED set information
            lpa.load_led_sets(layout_names=led_layouts)
            # Set step size and number of steps
            lpa.step_size = lpa_step_size
            lpa.set_n_steps(lpa_n_steps)

            # Fill light intensity array in LPA object
            for channel, inducer in enumerate(self.light_inducers):
                if isinstance(inducer, lpadesign.inducer.LightInducer):
                    # Light inducer
                    # Steady state inducer
                    # Intensities per well should be resolved in closed plate
                    intensities = closed_plate.well_info[
                        inducer._intensities_header].values
                    # 'NaN' values are registered for wells that are not to be
                    # measured. Switch these to zero.
                    intensities[numpy.isnan(intensities)] = 0
                    # Resize and add to all frames of intensity array in LPA.
                    intensities.resize(1, self.plate_n_rows, self.plate_n_cols)
                    lpa.intensity[:,:,:,channel] = intensities.repeat(
                        lpa.intensity.shape[0], axis=0)
                else:
                    raise NotImplementedError

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
            
            # Feed back to closed plates
            # Only necessary for steady state light inducers
            for channel, inducer in enumerate(self.light_inducers):
                if isinstance(inducer, lpadesign.inducer.LightInducer):
                    intensities = lpa.intensity[0,:,:,channel].flatten()
                    # Restore nan values on wells that will not be measured.
                    intensities[closed_plate.well_info['Measure']==False] \
                        = numpy.nan
                    # Update closed plate
                    closed_plate.well_info[inducer._intensities_header] = \
                        intensities

            # Save files
            lpa.save_files(path=os.path.join(path, self.lpa_files_folder))
