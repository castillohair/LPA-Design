"""
Module that contains the LPA and LPAArray classes.

"""

import numpy

import platedesign
import platedesign.plate

import lpaprogram

import lpadesign
import lpadesign.inducer

class LightPlate(platedesign.plate.Plate):
    """
    Object that represents a plate in an LPA.

    This class manages light inducers defined in ``lpadesign.inducers``.
    In addition, this class inherits from ``Plate`` in ``platedesign``. As
    such, it has the ability to manage all the chemical inducers in
    ``platedesign``.

    """
    def __init__(self,
                 name,
                 lpa_name,
                 lpa_n_channels=2,
                 n_rows=4,
                 n_cols=6,
                 id_prefix='S',
                 id_offset=0):

        # Store number of LED channels
        self.lpa_n_channels = lpa_n_channels
        # Initialize list of light inducers
        self.light_inducers = [None]*lpa_n_channels
        # Initialize LPA object
        self.lpa = lpaprogram.LPA(name=lpa_name,
                                  n_rows=n_rows,
                                  n_cols=n_cols,
                                  n_channels=lpa_n_channels)

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

        # Call Plate's constructor
        super().__init__(name, n_rows, n_cols, id_prefix, id_offset)

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
            ``lpa_n_channels``.

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
            if led_channel < 0 or led_channel >= self.lpa_n_channels:
                raise ValueError("led_channel should be between 0 and {}".\
                    format(self.lpa_n_channels - 1))
            if apply_to == 'media':
                raise ValueError('apply_to="media" not possible for inducer of '
                    'type {}'.format(type(inducer)))

        # Store inducer
        self.inducers[apply_to].append(inducer)
        if isinstance(inducer, lpadesign.inducer.LightInducer):
            self.light_inducers[led_channel] = inducer

    def save_rep_setup_files(self, path='.'):
        """
        Save additional files required for the Replicate Setup stage.

        Parameters
        ----------
        path : str
            Folder in which to save files.

        """
        # Get the led layout names from inducers
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

        # Fill light intensity array
        for channel, inducer in enumerate(self.light_inducers):
            if isinstance(inducer, lpadesign.inducer.LightInducer):
                if inducer in self.inducers['rows']:
                    # Inducer should be applied to rows
                    for j in range(self.n_cols):
                        self.lpa.intensity[:,:,j,channel] = \
                            inducer.intensities[j]
                elif inducer in self.inducers['cols']:
                    # Inducer should be applied to columns
                    for i in range(self.n_rows):
                        self.lpa.intensity[:,i,:,channel] = \
                            inducer.intensities[i]
                elif inducer in self.inducers['wells']:
                    # Inducer should be applied to all wells
                    for i in range(self.n_rows):
                        for j in range(self.n_cols):
                            self.lpa.intensity[:,i,j,channel] = \
                                inducer.intensities[i*self.n_cols + j]

                else:
                    raise ValueError("LightInducer not found in list of "
                        "inducers")
            else:
                raise NotImplementedError

        # Add additional frame at the end
        self.lpa.set_n_steps(self.lpa.intensity.shape[0] + 1)
        if self.lpa_end_with_leds_off:
            self.lpa.intensity[-1,:,:,:] = 0.

        # Optimize dc values if requested
        for channel, (optimize_dc, uniform) in \
                enumerate(zip(self.lpa_optimize_dc,
                              self.lpa_optimize_dc_uniform)):
            if optimize_dc:
                self.lpa.optimize_dc(channel=channel, uniform=uniform)

        # Discretize intensities and feed back to inducers
        self.lpa.discretize_intensity()
        # TODO: feed back to inducers

        # Save files
        self.lpa.save_files(path=path)

class LPAArray(platedesign.plate.PlateArray):
    pass