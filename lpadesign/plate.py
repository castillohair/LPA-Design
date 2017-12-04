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

        # Call Plate's constructor
        super(LPAPlate, self).__init__(name=name, n_rows=n_rows, n_cols=n_cols)

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Recover LPA inducers
        # LPA inducers are recognized from having the attributes "led_layout"
        # and "led_channel"
        lpa_inducers = [None]*self.n_led_channels
        lpa_inducers_apply_to = [None]*self.n_led_channels
        for apply_to, inducers in self.inducers.iteritems():
            for inducer in inducers:
                if hasattr(inducer, "led_layout") and \
                        hasattr(inducer, "led_channel"):
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
        if not os.path.exists(os.path.join(path, self.lpa_files_folder)):
            os.makedirs(os.path.join(path, self.lpa_files_folder))

        # Get LPA name from LPA resource
        self.lpa.name = self.resources['LPA'][0]

        # Load LED layout information
        led_layouts = [inducer.led_layout if inducer is not None
                       else None
                       for inducer in lpa_inducers]
        self.lpa.load_led_sets(layout_names=led_layouts)

        # Step size is one minute
        # Number of steps is the duration of the experiment
        self.lpa.step_size = 1000*60
        self.lpa.set_n_steps(self.lpa_program_duration)

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
                            inducer.intensities[j]
                    elif apply_to == 'cols':
                        self.lpa.intensity[:,i,j,channel] = \
                            inducer.intensities[i]
                    elif apply_to == 'wells':
                        if (i*self.n_cols + j) < self.samples_to_measure:
                            self.lpa.intensity[:,i,j,channel] = \
                                inducer.intensities[i*self.n_cols + j]
                    elif apply_to == 'media':
                        if (i*self.n_cols + j) < self.samples_to_measure:
                            self.lpa.intensity[:,i,j,channel] = \
                                inducer.intensities[0]

        # Condense intensity array if all intensities over time are the same
        # This comparison will be performed by comparing all frames to the first
        # frame, and testing for all True.
        if numpy.all((self.lpa.intensity[0] == self.lpa.intensity)):
            self.lpa.set_n_steps(1)

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

        # Call Plate's constructor
        platedesign.plate.PlateArray.__init__(self,
                                              name=name,
                                              array_n_rows=array_n_rows,
                                              array_n_cols=array_n_cols,
                                              plate_names=plate_names,
                                              plate_n_rows=plate_n_rows,
                                              plate_n_cols=plate_n_cols,)

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Recover LPA inducers
        # LPA inducers are recognized from having the attributes "led_layout"
        # and "led_channel"
        lpa_inducers = [None]*self.n_led_channels
        lpa_inducers_apply_to = [None]*self.n_led_channels
        for apply_to, inducers in self.inducers.iteritems():
            for inducer in inducers:
                if hasattr(inducer, "led_layout") and \
                        hasattr(inducer, "led_channel"):
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
        if not os.path.exists(os.path.join(path, self.lpa_files_folder)):
            os.makedirs(os.path.join(path, self.lpa_files_folder))

        # Get LED layout information
        led_layouts = [inducer.led_layout if inducer is not None
                       else None
                       for inducer in lpa_inducers]

        # Set LPA name, load LED layouts, set step size, and empty intensity array
        for lpa_idx, lpa in enumerate(self.lpas):
            # Set name from plate resources
            lpa.name = self.resources['LPA'][lpa_idx]

            # Load LED layout
            lpa.load_led_sets(layout_names=led_layouts)

            # Step size is one minute
            # Number of steps is the duration of the experiment
            lpa.step_size = 1000*60
            lpa.set_n_steps(self.lpa_program_duration)

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
                                    = inducer.intensities[j]
                            elif apply_to == 'cols':
                                self.lpas[array_k].\
                                    intensity[:, plate_i, plate_j, channel]\
                                    = inducer.intensities[i]
                            elif apply_to == 'wells':
                                if k < self.samples_to_measure:
                                    self.lpas[array_k].\
                                        intensity[:, plate_i, plate_j, channel]\
                                        = inducer.intensities[k]
                            elif apply_to == 'media':
                                if k < self.samples_to_measure:
                                    self.lpas[array_k].\
                                        intensity[:, plate_i, plate_j, channel]\
                                        = inducer.intensities[0]

        # Do post-processing and save LPA files
        for lpa in self.lpas:
            # Condense intensity array if all intensities over time are the same
            # This comparison will be performed by comparing all frames to the
            # first frame, and testing for all True.
            if numpy.all((lpa.intensity[0] == lpa.intensity)):
                lpa.set_n_steps(1)

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
            lpa.save_files(path=os.path.join(path, self.lpa_files_folder))
