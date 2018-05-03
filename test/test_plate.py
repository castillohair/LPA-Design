# -*- coding: UTF-8 -*-
"""
Unit tests for plate classes

"""

import collections
import itertools
import os
import random
import six
import shutil
import unittest

import numpy
import pandas

import lpaprogram
import lpadesign

class TestLPAPlate(unittest.TestCase):
    """
    Tests for the LPAPlate class.

    """
    def setUp(self):
        lpaprogram.LED_CALIBRATION_PATH = "test/test_plate_files/led-calibration"
        # Directory where to save temporary files
        self.temp_dir = "test/temp_lpa_plate"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        # Signal
        self.signal = 10.*numpy.sin(2*numpy.pi*numpy.arange(72)/72.) + 12.
        self.signal_init = 5.
        self.n_time_steps = 100

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        p = lpadesign.plate.LPAPlate(name='P1')

    def test_default_attributes(self):
        p = lpadesign.plate.LPAPlate(name='P1')
        # Check all attributes
        self.assertEqual(p.name, 'P1')
        self.assertEqual(p.n_rows, 4)
        self.assertEqual(p.n_cols, 6)
        self.assertEqual(p.n_led_channels, 2)
        self.assertEqual(p.n_plates, 1)
        self.assertEqual(p.samples_to_measure, 24)
        self.assertIsNone(p.sample_media_vol)
        self.assertIsNone(p.total_media_vol)
        self.assertIsNone(p.cell_strain_name)
        self.assertIsNone(p.cell_setup_method)
        self.assertEqual(p.cell_predilution, 1)
        self.assertIsNone(p.cell_predilution_vol)
        self.assertIsNone(p.cell_initial_od600)
        self.assertIsNone(p.cell_shot_vol)
        self.assertEqual(p.resources, collections.OrderedDict())
        self.assertEqual(p.metadata, collections.OrderedDict())
        self.assertEqual(p.inducers, {'rows': [],
                                      'cols': [],
                                      'wells': [],
                                      'media': []})
        self.assertIsInstance(p.lpa, lpaprogram.LPA)
        self.assertEqual(p.lpa.n_rows, 4)
        self.assertEqual(p.lpa.n_cols, 6)
        self.assertEqual(p.lpa.n_channels, 2)
        numpy.testing.assert_almost_equal(p.lpa.dc,
                                          numpy.ones((4, 6, 2))*8)
        numpy.testing.assert_almost_equal(p.lpa.gcal,
                                          numpy.ones((4, 6, 2))*255)
        numpy.testing.assert_almost_equal(p.lpa.intensity,
                                          numpy.zeros((1, 4, 6, 2)))
        self.assertEqual(p.lpa_optimize_dc, [True, True])
        self.assertEqual(p.lpa_optimize_dc_uniform, [True, True])
        self.assertEqual(p.lpa_end_with_leds_off, True)
        self.assertEqual(p.lpa_files_path, 'LPA Files')

    def test_non_default_attributes(self):
        p = lpadesign.plate.LPAPlate(name='P1',
                                     n_rows=8,
                                     n_cols=12,
                                     n_led_channels=4)
        # Check all attributes
        self.assertEqual(p.name, 'P1')
        self.assertEqual(p.n_rows, 8)
        self.assertEqual(p.n_cols, 12)
        self.assertEqual(p.n_led_channels, 4)
        self.assertEqual(p.n_plates, 1)
        self.assertEqual(p.samples_to_measure, 96)
        self.assertIsNone(p.sample_media_vol)
        self.assertIsNone(p.total_media_vol)
        self.assertIsNone(p.cell_strain_name)
        self.assertIsNone(p.cell_setup_method)
        self.assertEqual(p.cell_predilution, 1)
        self.assertIsNone(p.cell_predilution_vol)
        self.assertIsNone(p.cell_initial_od600)
        self.assertIsNone(p.cell_shot_vol)
        self.assertEqual(p.resources, collections.OrderedDict())
        self.assertEqual(p.metadata, collections.OrderedDict())
        self.assertEqual(p.inducers, {'rows': [],
                                      'cols': [],
                                      'wells': [],
                                      'media': []})
        self.assertIsInstance(p.lpa, lpaprogram.LPA)
        self.assertEqual(p.lpa.n_rows, 8)
        self.assertEqual(p.lpa.n_cols, 12)
        self.assertEqual(p.lpa.n_channels, 4)
        numpy.testing.assert_almost_equal(p.lpa.dc,
                                          numpy.ones((8, 12, 4))*8)
        numpy.testing.assert_almost_equal(p.lpa.gcal,
                                          numpy.ones((8, 12, 4))*255)
        numpy.testing.assert_almost_equal(p.lpa.intensity,
                                          numpy.zeros((1, 8, 12, 4)))
        self.assertEqual(p.lpa_optimize_dc, [True, True, True, True])
        self.assertEqual(p.lpa_optimize_dc_uniform, [True, True, True, True])
        self.assertEqual(p.lpa_end_with_leds_off, True)
        self.assertEqual(p.lpa_files_path, 'LPA Files')

    def test_save_rep_setup_files(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_channel_negative(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=-1,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'LED channel must be non-negative'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_channel_out_of_range(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=3,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = r'inducer 520nm Light assigned to LED channel 3 ' +\
            r'\(zero-based\), device only has 2 channels'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_too_many_inducers(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        light_750 = lpadesign.inducer.LightInducer(
            name='750nm Light',
            led_layout='750-FL',
            led_channel=1,
            id_prefix='R')
        light_750.intensities = range(24)
        p.apply_inducer(light_750, 'wells')

        # Attempt to generate rep setup files
        errmsg = 'more than one LPA inducer assigned to plate P1, LED channel 1'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_no_lpa(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'LPA name should be specified as a plate resource'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_no_time_step_info(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.time_step_size = None
        light_520.time_step_units = None
        light_520.n_time_steps = None
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        light_660.time_step_size = None
        light_660.time_step_units = None
        light_660.n_time_steps = None
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'time step size should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'time step units should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'number of time steps should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_520.n_time_steps = 2*60

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_conflicting_time_step_info_1(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.time_step_size = 60*1000
        light_520.time_step_units = 'min'
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        light_660.time_step_size = 60*60*1000
        light_660.time_step_units = 'hour'
        light_660.n_time_steps = 8
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'all time step sizes should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_660.n_time_steps = 8*60

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_conflicting_time_step_info_2(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.time_step_size = 60*1000
        light_520.time_step_units = 'min'
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        light_660.time_step_size = 60*60*1000
        light_660.time_step_units = 'hour'
        light_660.n_time_steps = 8
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'all time step sizes should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = None

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = None

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_660.n_time_steps = None

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_rows_and_cols(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 1
        dc_exp[:,:,1] = 13
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        , 10.0067361 ],
                                       [  1.00072838,  9.9997325 ],
                                       [  2.00071772,  9.99870525],
                                       [  2.99970161, 10.00640307],
                                       [  3.9997761 ,  9.99812794],
                                       [  5.00066595, 10.00112147]],

                                      [[  0.        , 20.00606138],
                                       [  1.00034242, 20.00573403],
                                       [  2.00026252, 19.99949528],
                                       [  2.99921931, 20.00540636],
                                       [  3.99957631, 20.00431076],
                                       [  5.00079841, 19.99753874]],

                                      [[  0.        , 29.99734398],
                                       [  1.00040402, 29.99570267],
                                       [  2.00079316, 29.99976748],
                                       [  2.9997588 , 29.99587517],
                                       [  3.99950927, 30.00428066],
                                       [  4.99980386, 29.99647402]],

                                      [[  0.        , 39.99495576],
                                       [  0.99922729, 39.99641917],
                                       [  2.00036745, 39.99971443],
                                       [  2.99934376, 39.99877411],
                                       [  3.99979618, 39.99520967],
                                       [  4.99957863, 39.99862333]]],


                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_rows_and_cols_end_with_leds_on(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.lpa_end_with_leds_off = False
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 1
        dc_exp[:,:,1] = 13
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        , 10.0067361 ],
                                       [  1.00072838,  9.9997325 ],
                                       [  2.00071772,  9.99870525],
                                       [  2.99970161, 10.00640307],
                                       [  3.9997761 ,  9.99812794],
                                       [  5.00066595, 10.00112147]],

                                      [[  0.        , 20.00606138],
                                       [  1.00034242, 20.00573403],
                                       [  2.00026252, 19.99949528],
                                       [  2.99921931, 20.00540636],
                                       [  3.99957631, 20.00431076],
                                       [  5.00079841, 19.99753874]],

                                      [[  0.        , 29.99734398],
                                       [  1.00040402, 29.99570267],
                                       [  2.00079316, 29.99976748],
                                       [  2.9997588 , 29.99587517],
                                       [  3.99950927, 30.00428066],
                                       [  4.99980386, 29.99647402]],

                                      [[  0.        , 39.99495576],
                                       [  0.99922729, 39.99641917],
                                       [  2.00036745, 39.99971443],
                                       [  2.99934376, 39.99877411],
                                       [  3.99979618, 39.99520967],
                                       [  4.99957863, 39.99862333]]],


                                     [[[  0.        , 10.0067361 ],
                                       [  1.00072838,  9.9997325 ],
                                       [  2.00071772,  9.99870525],
                                       [  2.99970161, 10.00640307],
                                       [  3.9997761 ,  9.99812794],
                                       [  5.00066595, 10.00112147]],

                                      [[  0.        , 20.00606138],
                                       [  1.00034242, 20.00573403],
                                       [  2.00026252, 19.99949528],
                                       [  2.99921931, 20.00540636],
                                       [  3.99957631, 20.00431076],
                                       [  5.00079841, 19.99753874]],

                                      [[  0.        , 29.99734398],
                                       [  1.00040402, 29.99570267],
                                       [  2.00079316, 29.99976748],
                                       [  2.9997588 , 29.99587517],
                                       [  3.99950927, 30.00428066],
                                       [  4.99980386, 29.99647402]],

                                      [[  0.        , 39.99495576],
                                       [  0.99922729, 39.99641917],
                                       [  2.00036745, 39.99971443],
                                       [  2.99934376, 39.99877411],
                                       [  3.99979618, 39.99520967],
                                       [  4.99957863, 39.99862333]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_rows_and_cols_full_dc_optimization(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.lpa_optimize_dc_uniform = [False, False]
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.array([[[ 1,  3],
                               [ 1,  3],
                               [ 1,  3],
                               [ 1,  2],
                               [ 1,  3],
                               [ 1,  2]],

                              [[ 1,  5],
                               [ 1,  5],
                               [ 1,  5],
                               [ 1,  5],
                               [ 1,  5],
                               [ 1,  5]],

                              [[ 1,  7],
                               [ 1,  7],
                               [ 1,  7],
                               [ 1,  7],
                               [ 1,  7],
                               [ 1,  9]],

                              [[ 1,  9],
                               [ 1, 10],
                               [ 1, 11],
                               [ 1, 11],
                               [ 1,  9],
                               [ 1, 13]]])
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   9.99887005],
                                       [  1.00072838,   9.9997325 ],
                                       [  2.00071772,  10.00083876],
                                       [  2.99970161,   9.99975606],
                                       [  3.9997761 ,   9.99928446],
                                       [  5.00066595,  10.00112147]],

                                      [[  0.        ,  20.00189649],
                                       [  1.00034242,  20.00214406],
                                       [  2.00026252,  19.99840187],
                                       [  2.99921931,  20.00191156],
                                       [  3.99957631,  19.99816379],
                                       [  5.00079841,  20.00200722]],

                                      [[  0.        ,  29.99851649],
                                       [  1.00040402,  30.00269468],
                                       [  2.00079316,  29.99864562],
                                       [  2.9997588 ,  29.99818601],
                                       [  3.99950927,  29.99983147],
                                       [  4.99980386,  29.99912928]],

                                      [[  0.        ,  40.00066998],
                                       [  0.99922729,  40.00361962],
                                       [  2.00036745,  39.9949529 ],
                                       [  2.99934376,  40.00412357],
                                       [  3.99979618,  39.99863186],
                                       [  4.99957863,  39.99862333]]],

                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_rows_and_cols_no_dc_optimization(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.lpa_optimize_dc = [False, False]
        p.lpa.set_all_dc(8, channel=0)
        p.lpa.set_all_dc(16, channel=1)
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [10, 20, 30, 40,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 8
        dc_exp[:,:,1] = 16
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   9.99662261],
                                       [  0.99775886,  10.00423081],
                                       [  2.00071772,  10.00190552],
                                       [  2.99970161,   9.99709726],
                                       [  3.99813213,   9.99234533],
                                       [  5.00380905,   9.99622136]],

                                      [[  0.        ,  20.00814383],
                                       [  0.99889055,  20.00812735],
                                       [  2.00026252,  19.99621506],
                                       [  3.00089391,  19.99958169],
                                       [  3.99957631,  19.99816379],
                                       [  4.99745787,  20.0008901 ]],

                                      [[  0.        ,  29.99734398],
                                       [  1.00040402,  30.00036401],
                                       [  2.00584142,  29.99415816],
                                       [  2.99672568,  30.00396312],
                                       [  4.00120469,  30.00539296],
                                       [  4.99673461,  29.99381875]],

                                      [[  0.        ,  40.00866988],
                                       [  1.00575819,  39.99333327],
                                       [  2.00206989,  39.99685751],
                                       [  2.99437522,  39.99966569],
                                       [  4.00451849,  40.00775772],
                                       [  5.00453035,  39.99704467]]],

                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_wells_media(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(24)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [20]
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  20.0033587 ],
                                       [  0.99775886,  20.00283873],
                                       [  2.00071772,  19.9974105 ],
                                       [  2.99970161,  19.99818273],
                                       [  3.99813213,  19.99625588],
                                       [  4.99752286,  19.99734283]],

                                      [[  6.00179289,  19.99981404],
                                       [  6.99804133,  20.00334072],
                                       [  8.00105007,  19.99949528],
                                       [  9.00268173,  20.00307649],
                                       [  9.99894077,  20.00123727],
                                       [ 10.99707975,  20.00312434]],

                                      [[ 11.99983339,  20.00174683],
                                       [ 12.99849282,  20.00179645],
                                       [ 14.00050384,  20.00171476],
                                       [ 15.00182715,  20.00148666],
                                       [ 15.99803708,  20.00248301],
                                       [ 16.99749156,  19.99941952]],

                                      [[ 17.9996932 ,  19.99976357],
                                       [ 18.99838029,  20.00283844],
                                       [ 20.00026957,  19.99842876],
                                       [ 21.00037487,  20.00250757],
                                       [ 21.99966602,  20.00273813],
                                       [ 23.0023532 ,  20.00167966]]],

                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_wells_media_samples_to_measure(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.samples_to_measure = 19
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(19)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [20]
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 6
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  19.9977401 ],
                                       [  0.99775886,  19.99946501],
                                       [  2.00071772,  20.00167752],
                                       [  2.99970161,  19.99685333],
                                       [  3.99813213,  19.99856893],
                                       [  4.99752286,  19.99979288]],

                                      [[  6.00179289,  19.9977316 ],
                                       [  6.99804133,  20.00334072],
                                       [  8.00105007,  20.0027755 ],
                                       [  9.00268173,  19.99725183],
                                       [  9.99894077,  20.00226177],
                                       [ 10.99707975,  20.0008901 ]],

                                      [[ 11.99983339,  20.00057433],
                                       [ 12.99849282,  19.99713511],
                                       [ 14.00050384,  19.99834917],
                                       [ 15.00182715,  20.00033124],
                                       [ 15.99803708,  20.00137071],
                                       [ 16.99749156,  19.99941952]],

                                      [[ 17.9996932 ,  20.00204925],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]],

                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_one_channel_zero(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [0, 0, 0, 0,]
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 1
        dc_exp[:,:,1] = 1
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  0.        ],
                                       [  1.00072838,  0.        ],
                                       [  2.00071772,  0.        ],
                                       [  2.99970161,  0.        ],
                                       [  3.9997761 ,  0.        ],
                                       [  5.00066595,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00034242,  0.        ],
                                       [  2.00026252,  0.        ],
                                       [  2.99921931,  0.        ],
                                       [  3.99957631,  0.        ],
                                       [  5.00079841,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00040402,  0.        ],
                                       [  2.00079316,  0.        ],
                                       [  2.9997588 ,  0.        ],
                                       [  3.99950927,  0.        ],
                                       [  4.99980386,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99922729,  0.        ],
                                       [  2.00036745,  0.        ],
                                       [  2.99934376,  0.        ],
                                       [  3.99979618,  0.        ],
                                       [  4.99957863,  0.        ]]],


                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)


    def test_save_rep_setup_files_one_channel_only(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000*60*8)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 1
        dc_exp[:,:,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  0.        ],
                                       [  1.00072838,  0.        ],
                                       [  2.00071772,  0.        ],
                                       [  2.99970161,  0.        ],
                                       [  3.9997761 ,  0.        ],
                                       [  5.00066595,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00034242,  0.        ],
                                       [  2.00026252,  0.        ],
                                       [  2.99921931,  0.        ],
                                       [  3.99957631,  0.        ],
                                       [  5.00079841,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00040402,  0.        ],
                                       [  2.00079316,  0.        ],
                                       [  2.9997588 ,  0.        ],
                                       [  3.99950927,  0.        ],
                                       [  4.99980386,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99922729,  0.        ],
                                       [  2.00036745,  0.        ],
                                       [  2.99934376,  0.        ],
                                       [  3.99979618,  0.        ],
                                       [  4.99957863,  0.        ]]],


                                     [[[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ],
                                       [  0.        ,  0.        ]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_light_signal_wells_media(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        # Write sampling times, signal, etc
        light_520.signal = self.signal
        light_520.signal_init = self.signal_init
        light_520.n_time_steps = self.n_time_steps
        light_520.sampling_time_steps = numpy.array([69,  6, 21, 63, 30, 36,
                                                     54, 45, 18, 12, 42, 66,
                                                     60, 51,  0,  3, 39, 33,
                                                     24, 27, 15, 48, 57,  9])
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = [20]
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Check red light intensity
        intensity_red_exp = numpy.array(
            [[ 20.0033587 ,  20.00283873,  19.9974105 ,
               19.99818273,  19.99625588,  19.99734283],
             [ 19.99981404,  20.00334072,  19.99949528,
               20.00307649,  20.00123727,  20.00312434],
             [ 20.00174683,  20.00179645,  20.00171476,
               20.00148666,  20.00248301,  19.99941952],
             [ 19.99976357,  20.00283844,  19.99842876,
               20.00250757,  20.00273813,  20.00167966]])
        intensity_red_exp.shape=(1,4,6)
        intensity_red_exp = numpy.repeat(intensity_red_exp, 101, 0)
        intensity_red_exp[-1,:,:] = 0.
        numpy.testing.assert_almost_equal(lpa.intensity[:,:,:,1], intensity_red_exp)
        # Check green light intensity
        intensity_init_green_exp = numpy.array(
            [[ 5.00242541,  5.00067236,  5.00179431,
               4.99723018,  4.99766516,  4.99752286],
             [ 5.00258254,  5.00026023,  5.00065629,
               4.99702424,  5.00288299,  4.99745787],
             [ 5.00265534,  5.00202012,  5.00114152,
               4.99858697,  4.99811502,  5.0028731 ],
             [ 5.00274759,  5.00266734,  4.99836495,
               5.00166657,  4.99935169,  4.99792806]])
        numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,0],
                                          intensity_init_green_exp)
        # Check specific wells of green light over time
        numpy.testing.assert_almost_equal(lpa.intensity[:,0,0,0], numpy.array([
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,   5.00242541,
             5.00242541,   5.00242541,   5.00242541,  12.00232685,
            12.8700351 ,  13.73774335,  14.58798096,  15.42074794,
            16.22439719,  16.99892871,  17.73851896,  18.42569731,
            19.07211084,  19.66028891,  20.19023153,  20.6619387 ,
            21.06376333,  21.39570541,  21.65776495,  21.84994194,
            21.9605893 ,  22.00135412,  21.9605893 ,  21.84994194,
            21.65776495,  21.39570541,  21.06376333,  20.6619387 ,
            20.19023153,  19.66028891,  19.07211084,  18.42569731,
            17.73851896,  16.99892871,  16.22439719,  15.42074794,
            14.58798096,  13.73774335,  12.8700351 ,  12.00232685,
            11.12879506,  10.26108681,   9.41084919,   8.57808222,
             7.77443297,   6.99990144,   6.26613474,   5.57313285,
             4.92671932,   4.33854124,   3.80859862,   3.33689145,
             2.93506683,   2.60312475,   2.34106521,   2.15471176,
             2.03824085,   1.99747604,   2.03824085,   2.15471176,
             2.34106521,   2.60312475,   2.93506683,   3.33689145,
             3.80859862,   4.33854124,   4.92671932,   5.57313285,
             6.26613474,   6.99990144,   7.77443297,   8.57808222,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,1,1,0], numpy.array([
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,   5.00026023,
             5.00026023,   5.00026023,   5.00026023,  11.99830156,
            12.86942704,  13.73474501,  14.58844798,  15.41892093,
            16.22616387,  16.99856179,  17.7361147 ,  18.42720757,
            19.07184043,  19.65839825,  20.19268854,  20.6630963 ,
            21.06381402,  21.3948417 ,  21.66198684,  21.84782694,
            21.96397701,  21.99882203,  21.96397701,  21.84782694,
            21.66198684,  21.3948417 ,  21.06381402,  20.6630963 ,
            20.19268854,  19.65839825,  19.07184043,  18.42720757,
            17.7361147 ,  16.99856179,  16.22616387,  15.41892093,
            14.58844798,  13.73474501,  12.86942704,  11.99830156,
            11.12717608,  10.26185811,   9.41396265,   8.57768219,
             7.77624675,   6.99804133,   6.26629593,   5.56939554,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,2,3,0], numpy.array([
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,   4.99858697,   4.99858697,   4.99858697,
             4.99858697,  11.99903522,  12.87257469,  13.73398167,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,3,5,0], numpy.array([
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,   4.99792806,
             4.99792806,   4.99792806,   4.99792806,  12.00295009,
            12.87445142,  13.73935046,  14.59104494,  15.42293257,
            16.22841107,  17.00087816,  17.73373154,  18.42697124,
            0.        ]))

    def test_save_rep_setup_files_light_signal_rows_cols(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = [0, 1, 2, 3, 4, 5]
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.StaggeredLightSignal(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        # Write sampling times, signal, etc
        light_660.signal = self.signal
        light_660.signal_init = self.signal_init
        light_660.n_time_steps = self.n_time_steps
        light_660.sampling_time_steps = numpy.array([ 72.,  20.,  30.,  10.])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Load LPA file and compare
        lpa = lpaprogram.LPA(name='Jennie', layout_names=['520-2-KB', '660-LS'])
        lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/Jennie'))
        self.assertEqual(lpa.n_rows, 4)
        self.assertEqual(lpa.n_cols, 6)
        self.assertEqual(lpa.n_channels, 2)
        self.assertEqual(lpa.step_size, 60000)
        # Dot correction
        dc_exp = numpy.ones((4, 6, 2))
        dc_exp[:,:,0] = 1
        dc_exp[:,:,1] = 7
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Check green light intensity
        intensity_green_exp = numpy.array(
            [[ 0.        ,  1.00072838,  2.00071772,
               2.99970161,  3.9997761 ,  5.00066595],
             [ 0.        ,  1.00034242,  2.00026252,
               2.99921931,  3.99957631,  5.00079841],
             [ 0.        ,  1.00040402,  2.00079316,
               2.9997588 ,  3.99950927,  4.99980386],
             [ 0.        ,  0.99922729,  2.00036745,
               2.99934376,  3.99979618,  4.99957863]])
        intensity_green_exp.shape=(1,4,6)
        intensity_green_exp = numpy.repeat(intensity_green_exp, 101, 0)
        intensity_green_exp[-1,:,:] = 0.
        numpy.testing.assert_almost_equal(lpa.intensity[:,:,:,0], intensity_green_exp)
        # # Check red light intensity
        intensity_init_red_exp = numpy.array(
            [[ 5.00280619,  4.99874168,  5.00308627,
               4.99721923,  5.0031118 ,  4.99933571],
             [ 4.99995351,  5.00083518,  4.99796036,
               4.99873049,  4.99851645,  4.99687117],
             [ 4.99838483,  5.00044911,  5.00239195,
               4.99834968,  4.99867423,  4.99985488],
             [ 4.99994089,  4.99710939,  4.99960719,
               4.99906663,  4.99868825,  5.00041992]])
        numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,1],
                                          intensity_init_red_exp)
        # Check specific wells of green light over time
        numpy.testing.assert_almost_equal(lpa.intensity[:,0,0,1], numpy.array([
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
             5.00280619,   5.00280619,   5.00280619,   5.00280619,
            12.00358843,  12.86885365,  13.73411887,  14.59151805,
            15.41745303,  16.22765592,  16.99852857,  17.73793703,
            18.43014921,  19.06729905,  19.65725261,  20.19214384,
            20.66410669,  21.06527511,  21.3956491 ,  21.66309472,
            21.85187986,  21.96200452,  22.00133476,  21.96200452,
            21.85187986,  21.66309472,  21.3956491 ,  21.06527511,
            20.66410669,  20.19214384,  19.65725261,  19.06729905,
            18.43014921,  17.73793703,  16.99852857,  16.22765592,
            15.41745303,  14.59151805,  13.73411887,  12.86885365,
            12.00358843,  11.13045716,  10.26519194,   9.41565881,
             8.58185778,   7.77165489,   7.00078224,   6.26137378,
             5.5691616 ,   4.93201176,   4.3420582 ,   3.80716697,
             3.34307017,   2.9340357 ,   2.60366171,   2.34408214,
             2.15529701,   2.03730629,   1.99797606,   2.03730629,
             2.15529701,   2.34408214,   2.60366171,   2.9340357 ,
             3.34307017,   3.80716697,   4.3420582 ,   4.93201176,
             5.5691616 ,   6.26137378,   7.00078224,   7.77165489,
             8.58185778,   9.41565881,  10.26519194,  11.13045716,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,1,1,1], numpy.array([
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
             5.00083518,   5.00083518,   5.00083518,   5.00083518,
            12.00367975,  12.87484702,  13.73763768,  14.59205173,
            15.42133595,  16.22549035,  16.99613832,  17.73327986,
            18.42853835,  19.07353719,  19.65989977,  20.1876261 ,
            20.65671617,  21.06716998,  21.3938577 ,  21.66190917,
            21.84619455,  21.96346707,  21.9969735 ,  21.96346707,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,2,3,1], numpy.array([
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,   4.99834968,   4.99834968,
             4.99834968,   4.99834968,  12.00250958,  12.86791964,
            13.7333297 ,  14.59065181,  15.42371009,  16.22441659,
            17.00085926,  17.73686221,  18.42433749,  19.07137304,
            19.66179299,  20.18750938,  20.65661016,  21.06100738,
            21.40070105,  21.65951527,  21.84553799,  21.95876922,
            21.99920894,  21.95876922,  21.84553799,  21.65951527,
            21.40070105,  21.06100738,  20.65661016,  20.18750938,
            19.66179299,  19.07137304,  18.42433749,  17.73686221,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,3,5,1], numpy.array([
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,   5.00041992,   5.00041992,
             5.00041992,   5.00041992,  12.0010078 ,  12.87400929,
            13.73596012,  14.58686031,  15.42118451,  16.22788209,
            17.00142771,  17.73629605,  18.42696179,  19.07342492,
             0.        ]))

class TestLPAPlateArray(unittest.TestCase):
    """
    Tests for the LPAPlate class.

    """
    def setUp(self):
        lpaprogram.LED_CALIBRATION_PATH = "test/test_plate_files/led-calibration"
        # Directory where to save temporary files
        self.temp_dir = "test/temp_lpa_plate_array"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        # Signal
        self.signal = 10.*numpy.sin(2*numpy.pi*numpy.arange(200)/200.) + 12.
        self.signal_init = 5.
        self.n_time_steps = 250

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])

    def test_create_dim_mismatch_error(self):
        self.assertRaises(ValueError,
                          lpadesign.plate.LPAPlateArray,
                          name='PA1',
                          array_n_rows=2,
                          array_n_cols=3,
                          plate_names=['P{}'.format(i+1)
                                       for i in range(4)])

    def test_default_attributes(self):
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        # Check all attributes
        self.assertEqual(p.name, 'PA1')
        self.assertEqual(p.plate_names, ['P{}'.format(i+1)
                                         for i in range(6)])
        self.assertEqual(p.array_n_rows, 2)
        self.assertEqual(p.array_n_cols, 3)
        self.assertEqual(p.plate_n_rows, 4)
        self.assertEqual(p.plate_n_cols, 6)
        self.assertEqual(p.n_rows, 8)
        self.assertEqual(p.n_cols, 18)
        self.assertEqual(p.n_led_channels, 2)
        self.assertEqual(p.n_plates, 6)
        self.assertEqual(p.samples_to_measure, 144)
        self.assertIsNone(p.sample_media_vol)
        self.assertIsNone(p.total_media_vol)
        self.assertIsNone(p.cell_strain_name)
        self.assertIsNone(p.cell_setup_method)
        self.assertEqual(p.cell_predilution, 1)
        self.assertIsNone(p.cell_predilution_vol)
        self.assertIsNone(p.cell_initial_od600)
        self.assertIsNone(p.cell_shot_vol)
        self.assertEqual(p.resources, collections.OrderedDict())
        self.assertEqual(p.metadata, collections.OrderedDict())
        self.assertEqual(p.inducers, {'rows': [],
                                      'cols': [],
                                      'wells': [],
                                      'media': []})
        for lpa in p.lpas:
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            numpy.testing.assert_almost_equal(lpa.dc,
                                              numpy.ones((4, 6, 2))*8)
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              numpy.ones((4, 6, 2))*255)
            numpy.testing.assert_almost_equal(lpa.intensity,
                                              numpy.zeros((1, 4, 6, 2)))
        self.assertEqual(p.lpa_optimize_dc, [True, True])
        self.assertEqual(p.lpa_optimize_dc_uniform, [True, True])
        self.assertEqual(p.lpa_end_with_leds_off, True)
        self.assertEqual(p.lpa_files_path, 'LPA Files')

    def test_non_default_attributes(self):
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)],
                                          plate_n_rows=8,
                                          plate_n_cols=12,
                                          n_led_channels=4)
        # Check all attributes
        self.assertEqual(p.name, 'PA1')
        self.assertEqual(p.plate_names, ['P{}'.format(i+1)
                                         for i in range(6)])
        self.assertEqual(p.array_n_rows, 2)
        self.assertEqual(p.array_n_cols, 3)
        self.assertEqual(p.plate_n_rows, 8)
        self.assertEqual(p.plate_n_cols, 12)
        self.assertEqual(p.n_rows, 16)
        self.assertEqual(p.n_cols, 36)
        self.assertEqual(p.n_led_channels, 4)
        self.assertEqual(p.n_plates, 6)
        self.assertEqual(p.samples_to_measure, 576)
        self.assertIsNone(p.sample_media_vol)
        self.assertIsNone(p.total_media_vol)
        self.assertIsNone(p.cell_strain_name)
        self.assertIsNone(p.cell_setup_method)
        self.assertEqual(p.cell_predilution, 1)
        self.assertIsNone(p.cell_predilution_vol)
        self.assertIsNone(p.cell_initial_od600)
        self.assertIsNone(p.cell_shot_vol)
        self.assertEqual(p.resources, collections.OrderedDict())
        self.assertEqual(p.metadata, collections.OrderedDict())
        self.assertEqual(p.inducers, {'rows': [],
                                      'cols': [],
                                      'wells': [],
                                      'media': []})
        for lpa in p.lpas:
            self.assertEqual(lpa.n_rows, 8)
            self.assertEqual(lpa.n_cols, 12)
            self.assertEqual(lpa.n_channels, 4)
            numpy.testing.assert_almost_equal(lpa.dc,
                                              numpy.ones((8, 12, 4))*8)
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              numpy.ones((8, 12, 4))*255)
            numpy.testing.assert_almost_equal(lpa.intensity,
                                              numpy.zeros((1, 8, 12, 4)))
        self.assertEqual(p.lpa_optimize_dc, [True, True, True, True])
        self.assertEqual(p.lpa_optimize_dc_uniform, [True, True, True, True])
        self.assertEqual(p.lpa_end_with_leds_off, True)
        self.assertEqual(p.lpa_files_path, 'LPA Files')

    def test_save_rep_setup_files(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_channel_negative(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=-1,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'LED channel must be non-negative'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_channel_out_of_range(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=3,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = r'inducer 520nm Light assigned to LED channel 3 ' +\
            r'\(zero-based\), device only has 2 channels'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_too_many_inducers(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        p.apply_inducer(light_660, 'cols')

        light_750 = lpadesign.inducer.LightInducer(
            name='750nm Light',
            led_layout='750-FL',
            led_channel=1,
            id_prefix='R')
        light_750.intensities = range(2*3*4*6)
        p.apply_inducer(light_750, 'wells')

        # Attempt to generate rep setup files
        errmsg = 'more than one LPA inducer assigned to plate PA1, LED channel 1'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_no_lpa(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.n_time_steps = 8*60
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'LPA names should be specified as plate resources'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_no_time_step_info(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.time_step_size = None
        light_520.time_step_units = None
        light_520.n_time_steps = None
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        light_660.time_step_size = None
        light_660.time_step_units = None
        light_660.n_time_steps = None
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'time step size should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'time step units should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'number of time steps should be specified'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_520.n_time_steps = 2*60

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_conflicting_time_step_info_1(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.time_step_size = 60*1000
        light_520.time_step_units = 'min'
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        light_660.time_step_size = 60*60*1000
        light_660.time_step_units = 'hour'
        light_660.n_time_steps = 8
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'all time step sizes should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_660.n_time_steps = 8*60

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_error_conflicting_time_step_info_2(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.arange(18)
        light_520.time_step_size = 60*1000
        light_520.time_step_units = 'min'
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = (numpy.arange(8) + 1)*10
        light_660.time_step_size = 60*60*1000
        light_660.time_step_units = 'hour'
        light_660.n_time_steps = 8
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        errmsg = 'all time step sizes should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = None

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = None

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify number of time steps
        light_660.n_time_steps = None

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

    def test_save_rep_setup_files_rows_and_cols(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([30, 60, 70, 40,
                                             20, 10, 80, 50])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 3*numpy.ones((4,6)),
                                      22*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         18*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                         18*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 3*numpy.ones((4,6)),
                                        23*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 3*numpy.ones((4,6)),
                                      25*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 3*numpy.ones((4,6)),
                                        21*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
            [[[  3.00122676,  29.99309214],
              [  8.99840934,  30.0017219 ],
              [ 12.00207294,  29.99495378],
              [ 17.99944501,  29.9958323 ],
              [ 15.99761037,  30.00349145],
              [  7.00035189,  29.99157411]],

             [[  3.00154443,  59.98883945],
              [  9.00201225,  59.99686604],
              [ 11.99902147,  59.99101413],
              [ 18.00024996,  60.00004788],
              [ 16.0007696 ,  59.99835672],
              [  7.00105535,  60.00882681]],

             [[  3.00022384,  69.9879331 ],
              [  8.99975124,  69.99338272],
              [ 12.00117286,  70.00471918],
              [ 17.99910341,  69.98881915],
              [ 15.99995828,  70.00092911],
              [  7.00106816,  69.9954012 ]],

             [[  3.00156701,  40.00994994],
              [  8.99793022,  40.01038974],
              [ 12.0009139 ,  40.00285966],
              [ 18.00039669,  40.00939077],
              [ 15.9993679 ,  39.99054221],
              [  7.0018766 ,  39.99614629]]])
        intensity['Tiffani'] = numpy.array(
            [[[  7.99895766,  29.99901343],
              [ 14.0024914 ,  30.00782224],
              [ 10.99900043,  30.0026154 ],
              [ 18.99962248,  30.00324108],
              [  9.99951414,  29.99027986],
              [  6.00115403,  29.99213568]],

             [[  8.0010562 ,  59.9940102 ],
              [ 13.99935256,  60.00416481],
              [ 10.99870858,  59.99309583],
              [ 19.00124298,  59.9944177 ],
              [ 10.00189558,  59.9948695 ],
              [  5.99862513,  59.99959731]],

             [[  7.99777081,  69.99723009],
              [ 13.99815478,  70.00197669],
              [ 11.00174834,  69.99146133],
              [ 18.99771176,  69.99518762],
              [  9.99846556,  70.00041462],
              [  6.00159872,  69.99436246]],

             [[  8.00233506,  39.99564361],
              [ 14.00234413,  39.99974521],
              [ 10.99906616,  39.99445508],
              [ 18.99904434,  40.00221765],
              [ 10.00191126,  40.00818938],
              [  6.0006131 ,  39.99296203]]])
        intensity['Shannen'] = numpy.array(
            [[[  3.99959473,  29.99053711],
              [ 14.9995695 ,  30.00156808],
              [ 16.99977485,  29.99239598],
              [  4.99731087,  30.00629783],
              [ 20.00119698,  29.99848159],
              [ 13.00222986,  30.00625054]],

             [[  3.99877874,  59.99166428],
              [ 15.00172159,  60.00634492],
              [ 17.00156084,  59.99493787],
              [  5.00193377,  60.00757215],
              [ 19.99879302,  60.00353054],
              [ 12.99915308,  59.99490441]],

             [[  3.99744805,  69.996955  ],
              [ 15.00158322,  69.99954309],
              [ 17.00193243,  70.00037511],
              [  4.99717009,  70.00274202],
              [ 19.99848449,  70.00757907],
              [ 12.99902022,  69.99285235]],

             [[  4.00183995,  39.99612374],
              [ 14.9991082 ,  39.99401122],
              [ 16.99891201,  39.99361561],
              [  4.99909126,  39.99250343],
              [ 19.99837409,  39.99181591],
              [ 13.00028124,  39.99184876]]])
        intensity['Jennie'] = numpy.array(
            [[[  3.00058171,  20.00448242],
              [  9.0021011 ,  19.99384212],
              [ 11.99940261,  19.99634374],
              [ 17.99820965,  19.99685333],
              [ 15.99910438,  20.00319501],
              [  7.00124665,  20.00469299]],

             [[  2.99763105,  10.01031925],
              [  8.99872617,   9.99090043],
              [ 12.0015751 ,  10.00904159],
              [ 18.00033965,   9.99396618],
              [ 16.00171785,   9.99088592],
              [  7.00011562,   9.99485946]],

             [[  3.0015932 ,  80.01284986],
              [  8.9985666 ,  80.00602047],
              [ 11.99971068,  79.98890922],
              [ 17.99855283,  79.98977074],
              [ 16.00142793,  79.99769674],
              [  6.99788385,  80.00298859]],

             [[  2.99994887,  49.99483754],
              [  8.99794377,  49.99063795],
              [ 12.00220467,  50.00464265],
              [ 18.00103111,  49.99423265],
              [ 15.99918471,  50.00741569],
              [  7.00172089,  49.9978845 ]]])
        intensity['Kirk'] = numpy.array(
            [[[  8.00048728,  19.9924968 ],
              [ 14.00167924,  19.9999909 ],
              [ 11.00192195,  20.00892807],
              [ 19.00231816,  20.00953621],
              [  9.99770114,  20.01188991],
              [  5.99900056,  20.00745135]],

             [[  7.99990768,   9.99431994],
              [ 14.00049248,   9.99994969],
              [ 10.99941874,   9.99857882],
              [ 18.99755728,  10.01129987],
              [  9.99776876,   9.99029581],
              [  6.00094267,  10.01064293]],

             [[  7.99961532,  79.9981038 ],
              [ 14.00194254,  79.99047153],
              [ 10.99842258,  80.01423394],
              [ 18.99741554,  80.01094343],
              [ 10.00077766,  79.98714707],
              [  6.00227918,  80.01213324]],

             [[  7.99841476,  49.99317272],
              [ 14.00053015,  50.00275268],
              [ 11.00067501,  49.98674246],
              [ 19.00108849,  49.99883948],
              [  9.99853032,  50.00430064],
              [  6.00152107,  49.99647708]]])
        intensity['Picard'] = numpy.array(
            [[[  3.99903334,  20.00197521],
              [ 14.99798174,  20.0088434 ],
              [ 17.00225243,  20.00165375],
              [  4.99915397,  20.01053966],
              [ 19.99952816,  20.00736148],
              [ 12.99899432,  19.99001261]],

             [[  4.0018762 ,  10.01043073],
              [ 14.99889641,  10.01109208],
              [ 17.00139244,  10.00110511],
              [  5.00220569,   9.99998564],
              [ 20.00196701,  10.00041424],
              [ 13.00160209,  10.00818915]],

             [[  4.00249787,  80.00760701],
              [ 15.00224886,  79.99289893],
              [ 16.99932637,  80.00019142],
              [  4.99864414,  80.0096808 ],
              [ 20.00237459,  80.00703171],
              [ 12.99922181,  80.01020877]],

             [[  4.00036777,  50.01072369],
              [ 14.99928653,  50.00849318],
              [ 16.99741191,  49.99430132],
              [  5.00226785,  50.00195886],
              [ 19.99866726,  50.00701966],
              [ 12.99831301,  50.0023608 ]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_rows_and_cols_full_dc_optimization(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        p.lpa_optimize_dc_uniform = [False, False]
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([30, 60, 70, 40,
                                             20, 10, 80, 50])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.array(
          [[[ 1,  8],
            [ 2,  8],
            [ 2,  7],
            [ 3,  7],
            [ 3,  7],
            [ 1,  7]],

           [[ 1, 12],
            [ 2, 13],
            [ 2, 15],
            [ 3, 15],
            [ 3, 13],
            [ 2, 15]],

           [[ 1, 15],
            [ 2, 15],
            [ 2, 22],
            [ 3, 17],
            [ 3, 14],
            [ 2, 16]],

           [[ 1,  9],
            [ 2,  8],
            [ 2,  9],
            [ 3,  9],
            [ 3,  9],
            [ 1,  9]]])
        dc_exp['Tiffani'] = numpy.array(
          [[[ 2,  7],
            [ 3,  6],
            [ 2,  7],
            [ 3,  8],
            [ 2,  7],
            [ 1,  7]],

           [[ 2, 15],
            [ 2, 17],
            [ 2, 16],
            [ 3, 16],
            [ 2, 13],
            [ 1, 18]],

           [[ 2, 15],
            [ 3, 15],
            [ 2, 17],
            [ 3, 16],
            [ 2, 18],
            [ 1, 17]],

           [[ 2,  9],
            [ 3,  8],
            [ 2, 10],
            [ 3, 10],
            [ 2,  9],
            [ 1,  9]]])
        dc_exp['Shannen'] = numpy.array(
          [[[ 1,  7],
            [ 3,  7],
            [ 3,  8],
            [ 1,  9],
            [ 4,  8],
            [ 2,  7]],

           [[ 1, 14],
            [ 3, 13],
            [ 3, 13],
            [ 1, 13],
            [ 4, 13],
            [ 2, 15]],

           [[ 1, 18],
            [ 3, 16],
            [ 3, 16],
            [ 1, 17],
            [ 3, 17],
            [ 2, 16]],

           [[ 1, 11],
            [ 3, 12],
            [ 3, 11],
            [ 1,  9],
            [ 4,  9],
            [ 2,  9]]])
        dc_exp['Jennie'] = numpy.array(
          [[[ 1,  5],
            [ 2,  5],
            [ 2,  5],
            [ 3,  4],
            [ 3,  5],
            [ 2,  4]],

           [[ 1,  3],
            [ 2,  3],
            [ 2,  3],
            [ 3,  3],
            [ 3,  3],
            [ 2,  3]],

           [[ 1, 17],
            [ 2, 17],
            [ 2, 18],
            [ 3, 17],
            [ 3, 18],
            [ 2, 23]],

           [[ 1, 11],
            [ 2, 12],
            [ 2, 13],
            [ 3, 14],
            [ 3, 11],
            [ 2, 16]]])
        dc_exp['Kirk'] = numpy.array(
          [[[ 2,  5],
            [ 3,  5],
            [ 2,  5],
            [ 3,  4],
            [ 2,  5],
            [ 2,  6]],

           [[ 2,  3],
            [ 3,  2],
            [ 2,  3],
            [ 3,  3],
            [ 2,  3],
            [ 1,  3]],

           [[ 2, 25],
            [ 2, 19],
            [ 2, 17],
            [ 3, 18],
            [ 2, 18],
            [ 1, 17]],

           [[ 2, 12],
            [ 3, 11],
            [ 2, 11],
            [ 3, 11],
            [ 2, 12],
            [ 1, 11]]])
        dc_exp['Picard'] = numpy.array(
          [[[ 1,  5],
            [ 3,  5],
            [ 3,  4],
            [ 1,  5],
            [ 3,  6],
            [ 2,  5]],

           [[ 1,  3],
            [ 3,  3],
            [ 3,  2],
            [ 1,  3],
            [ 3,  3],
            [ 2,  3]],

           [[ 1, 17],
            [ 3, 18],
            [ 3, 18],
            [ 1, 18],
            [ 3, 21],
            [ 2, 16]],

           [[ 1, 10],
            [ 3, 12],
            [ 3, 11],
            [ 1, 13],
            [ 3, 11],
            [ 2, 12]]])
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  2.99975557,  30.00247495],
            [  8.99991737,  30.00374973],
            [ 12.00031902,  30.00336987],
            [ 17.99944501,  29.99908635],
            [ 15.99761037,  29.99757734],
            [  7.00035189,  29.99694547]],

           [[  2.99982533,  59.99872799],
            [  9.00047265,  59.99805278],
            [ 11.99902147,  60.00734266],
            [ 18.00024996,  60.00522297],
            [ 16.0007696 ,  60.00306894],
            [  7.00105535,  59.99951735]],

           [[  3.00022384,  70.00815807],
            [  8.99975124,  69.99915679],
            [ 12.00117286,  70.00471918],
            [ 17.99910341,  69.99407751],
            [ 15.99995828,  69.99599599],
            [  6.99953988,  69.99995612]],

           [[  2.9998479 ,  40.00087944],
            [  8.99960675,  39.99525531],
            [ 11.99916628,  40.00400614],
            [ 18.00039669,  39.99815091],
            [ 15.9993679 ,  39.99662165],
            [  7.00016674,  39.99724878]]])
        intensity['Tiffani'] = numpy.array(
          [[[  7.99895766,  30.00342537],
            [ 14.0024914 ,  30.00046738],
            [ 11.00066165,  30.00141452],
            [ 18.99962248,  29.99738222],
            [  9.99951414,  30.00119097],
            [  5.99951839,  30.00208002]],

           [[  7.99935854,  60.00618184],
            [ 14.00106377,  59.99378526],
            [ 11.00035043,  60.00052716],
            [ 19.00124298,  59.996312  ],
            [  9.9985177 ,  59.99839529],
            [  6.0002722 ,  59.99959731]],

           [[  8.00111996,  69.99379481],
            [ 13.99815478,  70.00540917],
            [ 10.99871755,  70.00183874],
            [ 18.99771176,  70.00627579],
            [ 10.00015649,  70.00041462],
            [  5.99989903,  70.00687934]],

           [[  7.99900906,  39.99564361],
            [ 14.00234413,  40.00473333],
            [ 11.00071396,  40.00067603],
            [ 18.99904434,  39.99597803],
            [ 10.00015931,  39.99813707],
            [  6.0006131 ,  40.00378264]]])
        intensity['Shannen'] = numpy.array(
          [[[  3.99959473,  29.99610949],
            [ 14.99796355,  29.99915773],
            [ 16.99977485,  30.00016402],
            [  5.00061597,  29.99835966],
            [ 20.00119698,  29.99848159],
            [ 12.99882969,  29.9997175 ]],

           [[  4.00048471,  60.00061159],
            [ 14.9984303 ,  60.00634492],
            [ 17.00156084,  60.00195236],
            [  5.00025075,  59.99608041],
            [ 19.99879302,  60.00702481],
            [ 12.99915308,  60.00663018]],

           [[  4.00028311,  69.996955  ],
            [ 14.99988774,  70.00175834],
            [ 17.00025239,  69.99582401],
            [  5.00046637,  70.00172101],
            [ 20.00015159,  69.99300847],
            [ 12.99902022,  70.00625324]],

           [[  4.00016132,  39.99807031],
            [ 15.00221618,  39.99900237],
            [ 17.00218607,  39.99641041],
            [  5.0007444 ,  40.0024816 ],
            [ 19.99837409,  40.00261867],
            [ 13.00028124,  40.00222009]]])
        intensity['Jennie'] = numpy.array(
          [[[  3.00058171,  20.00223498],
            [  9.00061634,  20.00058958],
            [ 12.00103719,  20.00167752],
            [ 17.99820965,  19.99951213],
            [ 15.99910438,  20.00203849],
            [  6.9996751 ,  20.00224294]],

           [[  2.99926375,   9.9988658 ],
            [  8.99872617,  10.00167036],
            [ 12.0015751 ,  10.00138775],
            [ 18.00033965,   9.99862591],
            [ 16.00171785,  10.00113088],
            [  6.99844534,  10.00044505]],

           [[  2.99995835,  80.00933235],
            [  9.00025647,  79.99553246],
            [ 12.00139344,  80.00685905],
            [ 17.99855283,  80.00248036],
            [ 16.00142793,  80.00548284],
            [  7.0009531 ,  80.00298859]],

           [[  2.99994887,  49.99598039],
            [  8.9995765 ,  50.0040102 ],
            [ 11.99879979,  50.00273803],
            [ 18.00103111,  50.00314841],
            [ 15.99918471,  50.00399349],
            [  6.99841975,  49.99946316]]])
        intensity['Kirk'] = numpy.array(
          [[[  7.99874766,  19.99835111],
            [ 14.00167924,  19.9999909 ],
            [ 11.00023583,  19.99869332],
            [ 19.00231816,  19.99829841],
            [ 10.00115816,  20.00050335],
            [  6.00045698,  20.00044656]],

           [[  7.99990768,  10.00103881],
            [ 14.00049248,  10.00118807],
            [ 10.99941874,  10.00052029],
            [ 18.99755728,   9.99931029],
            [ 10.0012366 ,   9.99926668],
            [  5.99919414,  10.00110898]],

           [[  7.99961532,  79.9981038 ],
            [ 14.00021454,  79.99150367],
            [ 10.99987913,  79.99671897],
            [ 18.99741554,  80.00979426],
            [ 10.00077766,  80.00783105],
            [  6.00059077,  80.00861751]],

           [[  8.00011836,  50.00499271],
            [ 14.00053015,  49.99553641],
            [ 11.00067501,  49.99718052],
            [ 19.00108849,  49.99549507],
            [ 10.00020092,  49.99550867],
            [  5.99988265,  50.0032601 ]]])
        intensity['Picard'] = numpy.array(
          [[[  4.00069545,  19.99998467],
            [ 14.99798174,  20.0017063 ],
            [ 17.00225243,  20.00165375],
            [  4.99915397,  19.99750136],
            [ 19.99952816,  20.00229376],
            [ 13.00066794,  19.9986959 ]],

           [[  4.0001616 ,  10.00085442],
            [ 14.99889641,  10.00060194],
            [ 17.00139244,  10.00110511],
            [  5.00051803,   9.99998564],
            [ 20.00196701,  10.00041424],
            [ 13.00000817,  10.0011287 ]],

           [[  4.00081756,  80.00170387],
            [ 15.00224886,  80.00314786],
            [ 16.99932637,  80.00019142],
            [  5.00021604,  79.99937426],
            [ 20.00237459,  80.00703171],
            [ 12.99922181,  79.99670246]],

           [[  4.00036777,  50.00453807],
            [ 14.99928653,  50.00536393],
            [ 16.99741191,  50.00337922],
            [  5.00061366,  50.00578537],
            [ 19.99866726,  49.99897477],
            [ 12.99994126,  50.0054924 ]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_rows_and_cols_no_dc_optimization(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        p.lpa_optimize_dc = [False, False]
        for lpa in p.lpas:
            lpa.set_all_dc(5, channel=0)
            lpa.set_all_dc(25, channel=1)
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([30, 60, 70, 40,
                                             20, 10, 80, 50])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 5*numpy.ones((4,6)),
                                      25*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         25*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         25*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 5*numpy.ones((4,6)),
                                        25*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 5*numpy.ones((4,6)),
                                      25*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 5*numpy.ones((4,6)),
                                        25*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  3.00122676,  30.00153667],
            [  9.00293343,  30.01186104],
            [ 11.99681118,  29.99735838],
            [ 18.00289219,  29.99149356],
            [ 16.00254182,  30.01413686],
            [  6.99685346,  29.99909402]],

           [[  2.99982533,  60.01108867],
            [  8.99893306,  59.98974559],
            [ 12.00416243,  60.00734266],
            [ 18.00024996,  60.00522297],
            [ 16.0007696 ,  59.99246644],
            [  6.99936672,  59.99434542]],

           [[  3.00022384,  70.01410659],
            [  9.00141049,  70.01070495],
            [ 12.00292895,  69.99908869],
            [ 18.00421099,  69.98881915],
            [ 15.9982108 ,  69.98859632],
            [  6.99953988,  70.0033723 ]],

           [[  2.9998479 ,  39.99521038],
            [  9.00295981,  40.01165094],
            [ 11.99741866,  40.01203148],
            [ 18.00039669,  40.01388671],
            [ 15.9993679 ,  40.00270109],
            [  7.0018766 ,  39.99283882]]])
        intensity['Tiffani'] = numpy.array(
          [[[  7.99895766,  30.0012194 ],
            [ 13.99736417,  30.00169319],
            [ 10.9973392 ,  29.99180754],
            [ 19.00303569,  30.0022646 ],
            [  9.99784087,  30.00555541],
            [  6.00278966,  29.99876524]],

           [[  7.99596323,  59.99603881],
            [ 13.99764136,  60.0067597 ],
            [ 11.00035043,  60.00795848],
            [ 19.00291946,  60.00199491],
            [  9.9985177 ,  59.99722003],
            [  6.00356634,  59.99098411]],

           [[  7.99609623,  69.99379481],
            [ 13.99985585,  69.99396757],
            [ 11.00174834,  69.99561229],
            [ 19.00262199,  69.9940788 ],
            [ 10.00184742,  70.00332839],
            [  5.99989903,  69.99019017]],

           [[  7.99900906,  40.01370849],
            [ 13.99729095,  39.99849818],
            [ 10.99906616,  39.99549191],
            [ 19.0007118 ,  40.01157709],
            [ 10.00366322,  40.013774  ],
            [  6.0006131 ,  40.00618722]]])
        intensity['Shannen'] = numpy.array(
          [[[  4.00098156,  30.00725424],
            [ 14.9995695 ,  30.00879911],
            [ 16.99977485,  30.00404803],
            [  4.99896342,  30.01070793],
            [ 19.99963633,  30.01085721],
            [ 12.99712961,  29.99753982]],

           [[  4.00048471,  60.00284841],
            [ 15.00007594,  59.9922291 ],
            [ 17.00156084,  60.00312144],
            [  4.99856773,  59.98688701],
            [ 20.00333407,  60.01401334],
            [ 13.00074144,  59.99685871]],

           [[  3.99744805,  70.00499577],
            [ 14.99649679,  70.00175834],
            [ 17.00193243,  70.00151288],
            [  5.00211452,  69.99048986],
            [ 19.9968174 ,  69.99092696],
            [ 12.99902022,  69.99173561]],

           [[  4.00351857,  40.00196345],
            [ 15.00377017,  39.99151565],
            [ 17.00054904,  39.9889576 ],
            [  5.0007444 ,  39.99582949],
            [ 20.00299194,  40.00021806],
            [ 13.00028124,  39.98723928]]])
        intensity['Jennie'] = numpy.array(
          [[[  2.99912583,  20.00223498],
            [  8.99764682,  19.98934382],
            [ 11.99776804,  20.00167752],
            [ 17.99820965,  20.00748854],
            [ 16.00403629,  20.0078211 ],
            [  7.00124665,  19.99856786]],

           [[  2.99599835,   9.99574213],
            [  9.00162992,   9.99209709],
            [ 11.99993015,  10.00466797],
            [ 18.00201426,   9.98930644],
            [ 15.99659893,   9.98883693],
            [  6.99844534,   9.99821081]],

           [[  2.99995835,  79.99408979],
            [  8.9985666 ,  80.0001938 ],
            [ 11.99802793,  79.98890922],
            [ 18.00158595,  80.01287915],
            [ 15.99634165,  80.00214594],
            [  6.99788385,  79.98971229]],

           [[  3.00278168,  49.99940892],
            [  8.99631105,  49.99166658],
            [ 12.00220467,  49.99607189],
            [ 18.00268729,  49.99512423],
            [ 16.00075881,  49.99258617],
            [  6.99841975,  50.00419916]]])
        intensity['Kirk'] = numpy.array(
          [[[  8.00222689,  19.9924968 ],
            [ 14.00167924,  19.9999909 ],
            [ 11.00192195,  20.00892807],
            [ 19.0040984 ,  20.00953621],
            [  9.99942965,  20.01188991],
            [  6.00045698,  20.00745135]],

           [[  8.00150638,   9.99431994],
            [ 13.99885901,   9.99994969],
            [ 10.99788594,   9.99857882],
            [ 18.99755728,  10.01129987],
            [  9.99603485,   9.99029581],
            [  5.99744562,  10.01064293]],

           [[  8.00110667,  79.9981038 ],
            [ 13.99675855,  79.99047153],
            [ 10.99696602,  80.01423394],
            [ 18.99565506,  80.01094343],
            [  9.99727599,  79.98714707],
            [  6.00227918,  80.01213324]],

           [[  7.99841476,  49.99317272],
            [ 14.00391683,  50.00275268],
            [ 11.00067501,  49.98674246],
            [ 18.99757854,  49.99883948],
            [  9.99853032,  50.00430064],
            [  5.99660582,  49.99647708]]])
        intensity['Picard'] = numpy.array(
          [[[  3.99737123,  20.00496102],
            [ 15.00071287,  20.01360146],
            [ 17.00393765,  19.98684868],
            [  5.00264013,  19.99206873],
            [ 20.00285227,  19.99638142],
            [ 13.00401517,  19.9986959 ]],

           [[  4.0035908 ,  10.00191846],
            [ 14.99889641,   9.99477408],
            [ 16.99664477,   9.99847394],
            [  5.00389334,   9.98837125],
            [ 20.00369743,   9.99327619],
            [ 12.99841425,  10.00230544]],

           [[  3.99913725,  79.98753634],
            [ 15.00388684,  79.99859278],
            [ 16.99932637,  79.98933805],
            [  4.99864414,  79.99021289],
            [ 20.00237459,  80.00606252],
            [ 12.99576089,  79.99424676]],

           [[  4.00209281,  50.01072369],
            [ 15.00272752,  49.9897177 ],
            [ 16.99913771,  50.01359187],
            [  5.00392204,  50.00769862],
            [ 20.00037086,  49.99322842],
            [ 13.00156951,  50.00131693]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_wells_media(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array(
            [  1.8,  10. ,   2. ,  15.4,   4.8,  20. ,  10.6,   3.4,  20.8,
               3.8,   7.8,  26.8,   9. ,   6.6,  11.4,   5.6,  18.6,  13.8,
              14.6,   0.8,  24.8,   9.6,  25. ,   0.2,  12. ,  13.4,   1.2,
              17.4,  19.2,  15.2,  23. ,   3.6,   1. ,   1.4,  12.8,   3.2,
               4.4,   4.6,  20.2,  27.6,  28.2,  17. ,  17.8,   1.6,   9.8,
               8. ,  15. ,  21.4,  21.8,  27.2,  15.6,   7.6,  27.8,   8.8,
               8.6,  26.6,  21. ,  19.6,  17.2,  17.6,  19.4,  16.8,  22.4,
               2.8,  11. ,  16.2,  24.2,  13.2,  20.6,   2.2,  25.6,  19.8,
              14.2,  22.6,  18. ,  22.2,   7.4,   6. ,  16. ,  12.4,   8.2,
               7. ,   6.2,  26.2,  26.4,  24.6,   3. ,  28.6,  21.6,   4.2,
              24. ,  10.8,   9.2,  15.8,  16.6,  11.2,  26. ,  13. ,  16.4,
               8.4,  18.8,  14.4,  14.8,   6.8,   2.4,  28. ,  21.2,   4. ,
              28.4,  12.2,  18.4,   0.4,   6.4,  23.6,   5. ,  23.8,   5.4,
              27. ,  10.4,  25.8,  25.2,  10.2,   5.2,   9.4,  23.4,  13.6,
              25.4,  27.4,  23.2,  24.4,   5.8,  19. ,  11.8,   0. ,  20.4,
              11.6,  22.8,   0.6,   2.6,  22. ,  18.2,  12.6,  14. ,   7.2])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([20.])
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 5*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 5*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 5*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         6*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  1.80073605,  19.99946065],
            [ 10.00124866,  20.00047199],
            [  1.99946853,  19.99663585],
            [ 15.40199565,  19.9993909 ],
            [  4.79994064,  20.00390473],
            [ 19.99700719,  20.00297693]],

           [[ 14.59857692,  19.99586779],
            [  0.80058943,  20.00370231],
            [ 24.80003408,  20.00244755],
            [  9.60013331,  19.9965659 ],
            [ 25.00055523,  19.99748881],
            [  0.20263547,  19.99880473]],

           [[  4.4003283 ,  20.00369054],
            [  4.59943039,  19.99909919],
            [ 20.20214992,  19.99950981],
            [ 27.60134927,  20.00176336],
            [ 28.19739262,  20.00255584],
            [ 17.00062917,  19.99949943]],

           [[  8.60242916,  20.00043972],
            [ 26.60315944,  19.99636645],
            [ 20.9994148 ,  19.99913688],
            [ 19.60318304,  20.00019944],
            [ 17.19801762,  20.00135054],
            [ 17.59787886,  20.0035856 ]]])
        intensity['Tiffani'] = numpy.array(
          [[[ 10.59819966,  19.99934229],
            [  3.40106895,  19.99785997],
            [ 20.79859316,  20.0017436 ],
            [  3.79719393,  20.00216072],
            [  7.79747924,  20.00006657],
            [ 26.79992062,  20.00138668]],

           [[ 12.00243313,  19.9980034 ],
            [ 13.39872027,  20.00138827],
            [  1.19854564,  19.99769861],
            [ 17.40187949,  19.99813923],
            [ 19.20323417,  19.99828983],
            [ 15.20244644,  19.99986577]],

           [[ 17.8007336 ,  20.00019011],
            [  1.59901148,  19.99762264],
            [  9.79701143,  19.99933936],
            [  8.00368833,  19.99862503],
            [ 14.99854381,  20.00011846],
            [ 21.39907333,  20.00196553]],

           [[ 19.39884423,  20.00143478],
            [ 16.80180232,  19.99987261],
            [ 22.40184336,  20.00033801],
            [  2.80133355,  19.99798902],
            [ 11.00227758,  19.99739315],
            [ 16.19850544,  19.99648102]]])
        intensity['Shannen'] = numpy.array(
          [[[  9.00047497,  20.00037826],
            [  6.60045296,  20.00104538],
            [ 11.40064265,  20.00075668],
            [  5.60214413,  19.99890644],
            [ 18.60286111,  19.99898773],
            [ 13.79616832,  19.99763399]],

           [[ 22.99638969,  19.99722143],
            [  3.60396778,  20.00211497],
            [  0.99700511,  19.99831262],
            [  1.39690613,  20.00252405],
            [ 12.79819823,  20.00117685],
            [  3.20054905,  19.99830147]],

           [[ 21.80168473,  19.99740698],
            [ 27.2039334 ,  19.99702128],
            [ 15.599105  ,  20.00205764],
            [  7.59794001,  20.00165859],
            [ 27.79882702,  20.00127337],
            [  8.80176625,  20.00082943]],

           [[ 24.19736697,  20.00098172],
            [ 13.20114216,  19.99950118],
            [ 20.60201345,  19.9996026 ],
            [  2.1986744 ,  20.00290383],
            [ 25.59828826,  19.99950888],
            [ 19.79813823,  20.0028386 ]]])
        intensity['Jennie'] = numpy.array(
          [[[ 14.20217108,  20.0033587 ],
            [ 22.59805027,  20.00283873],
            [ 17.99665206,  19.9974105 ],
            [ 22.19949628,  19.99818273],
            [  7.39785961,  19.99625588],
            [  6.00331362,  19.99734283]],

           [[ 24.00064077,  19.99981404],
            [ 10.80195591,  20.00334072],
            [  9.20351051,  19.99949528],
            [ 15.7999074 ,  20.00307649],
            [ 16.60233681,  20.00123727],
            [ 11.19918282,  20.00312434]],

           [[ 28.39742588,  20.00174683],
            [ 12.2008734 ,  20.00179645],
            [ 18.40090258,  20.00171476],
            [  0.40188882,  20.00148666],
            [  6.40023209,  20.00248301],
            [ 23.60251176,  19.99941952]],

           [[ 25.40324973,  19.99976357],
            [ 27.39711423,  20.00283844],
            [ 23.19575016,  19.99842876],
            [ 24.4038268 ,  20.00250757],
            [  5.80057021,  20.00273813],
            [ 18.9980687 ,  20.00167966]]])
        intensity['Kirk'] = numpy.array(
          [[[ 15.99575571,  19.99835111],
            [ 12.40241984,  19.9999909 ],
            [  8.20296556,  20.00278722],
            [  6.9963566 ,  19.99829841],
            [  6.19670791,  19.99822603],
            [ 26.20102452,  19.99957096]],

           [[ 26.00289735,  19.99647856],
            [ 13.00244081,  19.99866099],
            [ 16.4010285 ,  19.99812838],
            [  8.40259775,  19.99981953],
            [ 18.79566656,  19.99771783],
            [ 14.39911506,  19.99903998]],

           [[  5.00348795,  19.9985312 ],
            [ 23.80312951,  19.99865002],
            [  5.39652439,  20.00093124],
            [ 26.99706861,  19.99785189],
            [ 10.39996837,  20.00359071],
            [ 25.79882582,  19.99981056]],

           [[ 11.79744882,  20.00049272],
            [  0.        ,  20.0035065 ],
            [ 20.40316936,  20.00397526],
            [ 11.60036898,  20.00065059],
            [ 22.80366565,  20.00172026],
            [  0.59802217,  19.99746033]]])
        intensity['Picard'] = numpy.array(
          [[[ 26.40259541,  19.99898939],
            [ 24.60062288,  19.99813776],
            [  2.99970357,  20.00165375],
            [ 28.60394584,  20.00076093],
            [ 21.59842669,  20.00229376],
            [  4.20078225,  19.99978131]],

           [[ 14.79699727,  20.00170884],
            [  6.79677418,  20.00120388],
            [  2.3975714 ,  20.00221023],
            [ 27.99817555,  19.99997127],
            [ 21.19768976,  20.00082848],
            [  4.00073817,  20.0022574 ]],

           [[ 25.196245  ,  19.9974744 ],
            [ 10.19641873,  19.99907881],
            [  5.20006829,  19.99841985],
            [  9.39996603,  20.00156132],
            [ 23.39687479,  19.99812349],
            [ 13.60142218,  20.00163131]],

           [[  2.59618521,  19.99686673],
            [ 21.99654488,  20.00214557],
            [ 18.19856926,  20.00316727],
            [ 12.59665003,  19.99733969],
            [ 14.00366682,  19.99729137],
            [  7.19686127,  19.99843903]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_wells_media_samples_to_measure(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        p.samples_to_measure = 110
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array(
            [  1.8,  10. ,   2. ,  15.4,   4.8,  20. ,  10.6,   3.4,  20.8,
               3.8,   7.8,  26.8,   9. ,   6.6,  11.4,   5.6,  18.6,  13.8,
              14.6,   0.8,  24.8,   9.6,  25. ,   0.2,  12. ,  13.4,   1.2,
              17.4,  19.2,  15.2,  23. ,   3.6,   1. ,   1.4,  12.8,   3.2,
               4.4,   4.6,  20.2,  27.6,  28.2,  17. ,  17.8,   1.6,   9.8,
               8. ,  15. ,  21.4,  21.8,  27.2,  15.6,   7.6,  27.8,   8.8,
               8.6,  26.6,  21. ,  19.6,  17.2,  17.6,  19.4,  16.8,  22.4,
               2.8,  11. ,  16.2,  24.2,  13.2,  20.6,   2.2,  25.6,  19.8,
              14.2,  22.6,  18. ,  22.2,   7.4,   6. ,  16. ,  12.4,   8.2,
               7. ,   6.2,  26.2,  26.4,  24.6,   3. ,  28.6,  21.6,   4.2,
              24. ,  10.8,   9.2,  15.8,  16.6,  11.2,  26. ,  13. ,  16.4,
               8.4,  18.8,  14.4,  14.8,   6.8,   2.4,  28. ,  21.2,   4. ,
              28.4,  12.2])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([20.])
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 5*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 5*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         5*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 5*numpy.ones((4,6)),
                                       6*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 5*numpy.ones((4,6)),
                                         6*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  1.80073605,  19.99946065],
            [ 10.00124866,  20.00047199],
            [  1.99946853,  19.99663585],
            [ 15.40199565,  19.9993909 ],
            [  4.79994064,  20.00390473],
            [ 19.99700719,  20.00297693]],

           [[ 14.59857692,  19.99586779],
            [  0.80058943,  20.00370231],
            [ 24.80003408,  20.00244755],
            [  9.60013331,  19.9965659 ],
            [ 25.00055523,  19.99748881],
            [  0.20263547,  19.99880473]],

           [[  4.4003283 ,  20.00369054],
            [  4.59943039,  19.99909919],
            [ 20.20214992,  19.99950981],
            [ 27.60134927,  20.00176336],
            [ 28.19739262,  20.00255584],
            [ 17.00062917,  19.99949943]],

           [[  8.60242916,  20.00043972],
            [ 26.60315944,  19.99636645],
            [ 20.9994148 ,  19.99913688],
            [ 19.60318304,  20.00019944],
            [ 17.19801762,  20.00135054],
            [ 17.59787886,  20.0035856 ]]])
        intensity['Tiffani'] = numpy.array(
          [[[ 10.59819966,  19.99934229],
            [  3.40106895,  19.99785997],
            [ 20.79859316,  20.0017436 ],
            [  3.79719393,  20.00216072],
            [  7.79747924,  20.00006657],
            [ 26.79992062,  20.00138668]],

           [[ 12.00243313,  19.9980034 ],
            [ 13.39872027,  20.00138827],
            [  1.19854564,  19.99769861],
            [ 17.40187949,  19.99813923],
            [ 19.20323417,  19.99828983],
            [ 15.20244644,  19.99986577]],

           [[ 17.8007336 ,  20.00019011],
            [  1.59901148,  19.99762264],
            [  9.79701143,  19.99933936],
            [  8.00368833,  19.99862503],
            [ 14.99854381,  20.00011846],
            [ 21.39907333,  20.00196553]],

           [[ 19.39884423,  20.00143478],
            [ 16.80180232,  19.99987261],
            [ 22.40184336,  20.00033801],
            [  2.80133355,  19.99798902],
            [ 11.00227758,  19.99739315],
            [ 16.19850544,  19.99648102]]])
        intensity['Shannen'] = numpy.array(
          [[[  9.00047497,  20.00037826],
            [  6.60045296,  20.00104538],
            [ 11.40064265,  20.00075668],
            [  5.60214413,  19.99890644],
            [ 18.60286111,  19.99898773],
            [ 13.79616832,  19.99763399]],

           [[ 22.99638969,  19.99722143],
            [  3.60396778,  20.00211497],
            [  0.99700511,  19.99831262],
            [  1.39690613,  20.00252405],
            [ 12.79819823,  20.00117685],
            [  3.20054905,  19.99830147]],

           [[ 21.80168473,  19.99740698],
            [ 27.2039334 ,  19.99702128],
            [ 15.599105  ,  20.00205764],
            [  7.59794001,  20.00165859],
            [ 27.79882702,  20.00127337],
            [  8.80176625,  20.00082943]],

           [[ 24.19736697,  20.00098172],
            [ 13.20114216,  19.99950118],
            [ 20.60201345,  19.9996026 ],
            [  2.1986744 ,  20.00290383],
            [ 25.59828826,  19.99950888],
            [ 19.79813823,  20.0028386 ]]])
        intensity['Jennie'] = numpy.array(
          [[[ 14.20217108,  20.00223498],
            [ 22.59805027,  20.00058958],
            [ 17.99665206,  20.00167752],
            [ 22.19949628,  20.00084153],
            [  7.39785961,  20.00203849],
            [  6.00331362,  19.99856786]],

           [[ 24.00064077,  20.00189649],
            [ 10.80195591,  20.00214406],
            [  9.20351051,  19.99840187],
            [ 15.7999074 ,  20.00191156],
            [ 16.60233681,  19.99816379],
            [ 11.19918282,  20.00200722]],

           [[ 28.39742588,  20.00291934],
            [ 12.2008734 ,  19.99713511],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]],

           [[  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]]])
        intensity['Kirk'] = numpy.array(
          [[[ 15.99575571,  20.00069284],
            [ 12.40241984,  20.0034541 ],
            [  8.20296556,  20.00074027],
            [  6.9963566 ,  20.00329299],
            [  6.19670791,  19.99708738],
            [ 26.20102452,  20.00044656]],

           [[ 26.00289735,  20.00207762],
            [ 13.00244081,  20.00237614],
            [ 16.4010285 ,  20.00104058],
            [  8.40259775,  19.99862058],
            [ 18.79566656,  19.99853337],
            [ 14.39911506,  20.00221796]],

           [[  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]],

           [[  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]]])
        intensity['Picard'] = numpy.array(
          [[[ 26.40259541,  19.99898939],
            [ 24.60062288,  19.99813776],
            [  2.99970357,  20.00165375],
            [ 28.60394584,  20.00076093],
            [ 21.59842669,  20.00229376],
            [  4.20078225,  19.99978131]],

           [[ 14.79699727,  20.00170884],
            [  6.79677418,  20.00120388],
            [  2.3975714 ,  20.00221023],
            [ 27.99817555,  19.99997127],
            [ 21.19768976,  20.00082848],
            [  4.00073817,  20.0022574 ]],

           [[  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]],

           [[  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ],
            [  0.        ,   0.        ]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_one_channel_zero(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.zeros(8)
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       1*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 3*numpy.ones((4,6)),
                                          1*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          1*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         1*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       1*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         1*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  3.00122676,   0.        ],
            [  8.99840934,   0.        ],
            [ 12.00207294,   0.        ],
            [ 17.99944501,   0.        ],
            [ 15.99761037,   0.        ],
            [  7.00035189,   0.        ]],

           [[  3.00154443,   0.        ],
            [  9.00201225,   0.        ],
            [ 11.99902147,   0.        ],
            [ 18.00024996,   0.        ],
            [ 16.0007696 ,   0.        ],
            [  7.00105535,   0.        ]],

           [[  3.00022384,   0.        ],
            [  8.99975124,   0.        ],
            [ 12.00117286,   0.        ],
            [ 17.99910341,   0.        ],
            [ 15.99995828,   0.        ],
            [  7.00106816,   0.        ]],

           [[  3.00156701,   0.        ],
            [  8.99793022,   0.        ],
            [ 12.0009139 ,   0.        ],
            [ 18.00039669,   0.        ],
            [ 15.9993679 ,   0.        ],
            [  7.0018766 ,   0.        ]]])
        intensity['Tiffani'] = numpy.array(
          [[[  7.99895766,   0.        ],
            [ 14.0024914 ,   0.        ],
            [ 10.99900043,   0.        ],
            [ 18.99962248,   0.        ],
            [  9.99951414,   0.        ],
            [  6.00115403,   0.        ]],

           [[  8.0010562 ,   0.        ],
            [ 13.99935256,   0.        ],
            [ 10.99870858,   0.        ],
            [ 19.00124298,   0.        ],
            [ 10.00189558,   0.        ],
            [  5.99862513,   0.        ]],

           [[  7.99777081,   0.        ],
            [ 13.99815478,   0.        ],
            [ 11.00174834,   0.        ],
            [ 18.99771176,   0.        ],
            [  9.99846556,   0.        ],
            [  6.00159872,   0.        ]],

           [[  8.00233506,   0.        ],
            [ 14.00234413,   0.        ],
            [ 10.99906616,   0.        ],
            [ 18.99904434,   0.        ],
            [ 10.00191126,   0.        ],
            [  6.0006131 ,   0.        ]]])
        intensity['Shannen'] = numpy.array(
          [[[  3.99959473,   0.        ],
            [ 14.9995695 ,   0.        ],
            [ 16.99977485,   0.        ],
            [  4.99731087,   0.        ],
            [ 20.00119698,   0.        ],
            [ 13.00222986,   0.        ]],

           [[  3.99877874,   0.        ],
            [ 15.00172159,   0.        ],
            [ 17.00156084,   0.        ],
            [  5.00193377,   0.        ],
            [ 19.99879302,   0.        ],
            [ 12.99915308,   0.        ]],

           [[  3.99744805,   0.        ],
            [ 15.00158322,   0.        ],
            [ 17.00193243,   0.        ],
            [  4.99717009,   0.        ],
            [ 19.99848449,   0.        ],
            [ 12.99902022,   0.        ]],

           [[  4.00183995,   0.        ],
            [ 14.9991082 ,   0.        ],
            [ 16.99891201,   0.        ],
            [  4.99909126,   0.        ],
            [ 19.99837409,   0.        ],
            [ 13.00028124,   0.        ]]])
        intensity['Jennie'] = numpy.array(
          [[[  3.00058171,   0.        ],
            [  9.0021011 ,   0.        ],
            [ 11.99940261,   0.        ],
            [ 17.99820965,   0.        ],
            [ 15.99910438,   0.        ],
            [  7.00124665,   0.        ]],

           [[  2.99763105,   0.        ],
            [  8.99872617,   0.        ],
            [ 12.0015751 ,   0.        ],
            [ 18.00033965,   0.        ],
            [ 16.00171785,   0.        ],
            [  7.00011562,   0.        ]],

           [[  3.0015932 ,   0.        ],
            [  8.9985666 ,   0.        ],
            [ 11.99971068,   0.        ],
            [ 17.99855283,   0.        ],
            [ 16.00142793,   0.        ],
            [  6.99788385,   0.        ]],

           [[  2.99994887,   0.        ],
            [  8.99794377,   0.        ],
            [ 12.00220467,   0.        ],
            [ 18.00103111,   0.        ],
            [ 15.99918471,   0.        ],
            [  7.00172089,   0.        ]]])
        intensity['Kirk'] = numpy.array(
   

          [[[  8.00048728,   0.        ],
            [ 14.00167924,   0.        ],
            [ 11.00192195,   0.        ],
            [ 19.00231816,   0.        ],
            [  9.99770114,   0.        ],
            [  5.99900056,   0.        ]],

           [[  7.99990768,   0.        ],
            [ 14.00049248,   0.        ],
            [ 10.99941874,   0.        ],
            [ 18.99755728,   0.        ],
            [  9.99776876,   0.        ],
            [  6.00094267,   0.        ]],

           [[  7.99961532,   0.        ],
            [ 14.00194254,   0.        ],
            [ 10.99842258,   0.        ],
            [ 18.99741554,   0.        ],
            [ 10.00077766,   0.        ],
            [  6.00227918,   0.        ]],

           [[  7.99841476,   0.        ],
            [ 14.00053015,   0.        ],
            [ 11.00067501,   0.        ],
            [ 19.00108849,   0.        ],
            [  9.99853032,   0.        ],
            [  6.00152107,   0.        ]]])
        intensity['Picard'] = numpy.array(
          [[[  3.99903334,   0.        ],
            [ 14.99798174,   0.        ],
            [ 17.00225243,   0.        ],
            [  4.99915397,   0.        ],
            [ 19.99952816,   0.        ],
            [ 12.99899432,   0.        ]],

           [[  4.0018762 ,   0.        ],
            [ 14.99889641,   0.        ],
            [ 17.00139244,   0.        ],
            [  5.00220569,   0.        ],
            [ 20.00196701,   0.        ],
            [ 13.00160209,   0.        ]],

           [[  4.00249787,   0.        ],
            [ 15.00224886,   0.        ],
            [ 16.99932637,   0.        ],
            [  4.99864414,   0.        ],
            [ 20.00237459,   0.        ],
            [ 12.99922181,   0.        ]],

           [[  4.00036777,   0.        ],
            [ 14.99928653,   0.        ],
            [ 16.99741191,   0.        ],
            [  5.00226785,   0.        ],
            [ 19.99866726,   0.        ],
            [ 12.99831301,   0.        ]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_one_channel_only(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        light_520.n_time_steps = 60*8
        p.apply_inducer(light_520, 'rows')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       8*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 3*numpy.ones((4,6)),
                                          8*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          8*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         8*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       8*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         8*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
          [[[  3.00122676,   0.        ],
            [  8.99840934,   0.        ],
            [ 12.00207294,   0.        ],
            [ 17.99944501,   0.        ],
            [ 15.99761037,   0.        ],
            [  7.00035189,   0.        ]],

           [[  3.00154443,   0.        ],
            [  9.00201225,   0.        ],
            [ 11.99902147,   0.        ],
            [ 18.00024996,   0.        ],
            [ 16.0007696 ,   0.        ],
            [  7.00105535,   0.        ]],

           [[  3.00022384,   0.        ],
            [  8.99975124,   0.        ],
            [ 12.00117286,   0.        ],
            [ 17.99910341,   0.        ],
            [ 15.99995828,   0.        ],
            [  7.00106816,   0.        ]],

           [[  3.00156701,   0.        ],
            [  8.99793022,   0.        ],
            [ 12.0009139 ,   0.        ],
            [ 18.00039669,   0.        ],
            [ 15.9993679 ,   0.        ],
            [  7.0018766 ,   0.        ]]])
        intensity['Tiffani'] = numpy.array(
          [[[  7.99895766,   0.        ],
            [ 14.0024914 ,   0.        ],
            [ 10.99900043,   0.        ],
            [ 18.99962248,   0.        ],
            [  9.99951414,   0.        ],
            [  6.00115403,   0.        ]],

           [[  8.0010562 ,   0.        ],
            [ 13.99935256,   0.        ],
            [ 10.99870858,   0.        ],
            [ 19.00124298,   0.        ],
            [ 10.00189558,   0.        ],
            [  5.99862513,   0.        ]],

           [[  7.99777081,   0.        ],
            [ 13.99815478,   0.        ],
            [ 11.00174834,   0.        ],
            [ 18.99771176,   0.        ],
            [  9.99846556,   0.        ],
            [  6.00159872,   0.        ]],

           [[  8.00233506,   0.        ],
            [ 14.00234413,   0.        ],
            [ 10.99906616,   0.        ],
            [ 18.99904434,   0.        ],
            [ 10.00191126,   0.        ],
            [  6.0006131 ,   0.        ]]])
        intensity['Shannen'] = numpy.array(
          [[[  3.99959473,   0.        ],
            [ 14.9995695 ,   0.        ],
            [ 16.99977485,   0.        ],
            [  4.99731087,   0.        ],
            [ 20.00119698,   0.        ],
            [ 13.00222986,   0.        ]],

           [[  3.99877874,   0.        ],
            [ 15.00172159,   0.        ],
            [ 17.00156084,   0.        ],
            [  5.00193377,   0.        ],
            [ 19.99879302,   0.        ],
            [ 12.99915308,   0.        ]],

           [[  3.99744805,   0.        ],
            [ 15.00158322,   0.        ],
            [ 17.00193243,   0.        ],
            [  4.99717009,   0.        ],
            [ 19.99848449,   0.        ],
            [ 12.99902022,   0.        ]],

           [[  4.00183995,   0.        ],
            [ 14.9991082 ,   0.        ],
            [ 16.99891201,   0.        ],
            [  4.99909126,   0.        ],
            [ 19.99837409,   0.        ],
            [ 13.00028124,   0.        ]]])
        intensity['Jennie'] = numpy.array(
          [[[  3.00058171,   0.        ],
            [  9.0021011 ,   0.        ],
            [ 11.99940261,   0.        ],
            [ 17.99820965,   0.        ],
            [ 15.99910438,   0.        ],
            [  7.00124665,   0.        ]],

           [[  2.99763105,   0.        ],
            [  8.99872617,   0.        ],
            [ 12.0015751 ,   0.        ],
            [ 18.00033965,   0.        ],
            [ 16.00171785,   0.        ],
            [  7.00011562,   0.        ]],

           [[  3.0015932 ,   0.        ],
            [  8.9985666 ,   0.        ],
            [ 11.99971068,   0.        ],
            [ 17.99855283,   0.        ],
            [ 16.00142793,   0.        ],
            [  6.99788385,   0.        ]],

           [[  2.99994887,   0.        ],
            [  8.99794377,   0.        ],
            [ 12.00220467,   0.        ],
            [ 18.00103111,   0.        ],
            [ 15.99918471,   0.        ],
            [  7.00172089,   0.        ]]])
        intensity['Kirk'] = numpy.array(
   

          [[[  8.00048728,   0.        ],
            [ 14.00167924,   0.        ],
            [ 11.00192195,   0.        ],
            [ 19.00231816,   0.        ],
            [  9.99770114,   0.        ],
            [  5.99900056,   0.        ]],

           [[  7.99990768,   0.        ],
            [ 14.00049248,   0.        ],
            [ 10.99941874,   0.        ],
            [ 18.99755728,   0.        ],
            [  9.99776876,   0.        ],
            [  6.00094267,   0.        ]],

           [[  7.99961532,   0.        ],
            [ 14.00194254,   0.        ],
            [ 10.99842258,   0.        ],
            [ 18.99741554,   0.        ],
            [ 10.00077766,   0.        ],
            [  6.00227918,   0.        ]],

           [[  7.99841476,   0.        ],
            [ 14.00053015,   0.        ],
            [ 11.00067501,   0.        ],
            [ 19.00108849,   0.        ],
            [  9.99853032,   0.        ],
            [  6.00152107,   0.        ]]])
        intensity['Picard'] = numpy.array(
          [[[  3.99903334,   0.        ],
            [ 14.99798174,   0.        ],
            [ 17.00225243,   0.        ],
            [  4.99915397,   0.        ],
            [ 19.99952816,   0.        ],
            [ 12.99899432,   0.        ]],

           [[  4.0018762 ,   0.        ],
            [ 14.99889641,   0.        ],
            [ 17.00139244,   0.        ],
            [  5.00220569,   0.        ],
            [ 20.00196701,   0.        ],
            [ 13.00160209,   0.        ]],

           [[  4.00249787,   0.        ],
            [ 15.00224886,   0.        ],
            [ 16.99932637,   0.        ],
            [  4.99864414,   0.        ],
            [ 20.00237459,   0.        ],
            [ 12.99922181,   0.        ]],

           [[  4.00036777,   0.        ],
            [ 14.99928653,   0.        ],
            [ 16.99741191,   0.        ],
            [  5.00226785,   0.        ],
            [ 19.99866726,   0.        ],
            [ 12.99831301,   0.        ]]])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000*60*8)
            self.assertEqual(lpa.intensity.shape[0], 2)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_light_signal_wells_media(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make 2 inducers and add them.
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.signal = self.signal
        light_520.signal_init = self.signal_init
        light_520.n_time_steps = self.n_time_steps
        light_520.sampling_time_steps = numpy.array(
            [ 97,  31,   8,  68,  12,  13,  19,  24,  99,  32,  57, 136,  94,  38, 114,  78, 140,  76,
              56,  62,  15,  63,  89,  54,  48,  33,   6, 141,  43, 134,  59,  75, 110,  46,  81,  42,
             112, 125,  22, 129,  10,  70,  93,  25, 105,  58, 111, 107,  40,  83,  44,   3, 138,  69,
              30, 121, 109,  55, 200,  52,  80, 135,  23,  87, 123,  50,  14,  92,  11,   5,  74,  16,
             101,  35, 106,  67,  26, 132, 128,  53,  85, 113,  64, 126,  91,  17,   1,  79, 122, 131,
              45,  27,  61,   9, 118,  41,  65, 137,  34,  37,  51, 124, 119, 133,  18,  60, 102,  88,
               4,  96,   0, 130,  47,  84, 108, 117, 100, 120, 104, 116,  36,   7,  77,  39,  71,  49,
              82,  28,  66, 103, 127, 142,  72,  90,  95,  21,  98,  20, 139,   2, 115,  86,  29,  73])
        p.apply_inducer(light_520, 'wells')

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.array([20.])
        p.apply_inducer(light_660, 'media')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          6*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 4*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 4*numpy.ones((4,6)),
                                         6*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity_red = {}
        intensity_red['Tori'] = numpy.array(
            [[ 19.99946065,  20.00047199,  19.99663585,  19.9993909 ,
               20.00390473,  20.00297693],
             [ 19.99586779,  20.00370231,  20.00244755,  19.9965659 ,
               19.99748881,  19.99880473],
             [ 20.00369054,  19.99909919,  19.99950981,  20.00176336,
               20.00255584,  19.99949943],
             [ 20.00043972,  19.99636645,  19.99913688,  20.00019944,
               20.00135054,  20.0035856 ]])
        intensity_red['Tiffani'] = numpy.array(
            [[ 19.99934229,  19.99785997,  20.0017436 ,  20.00216072,
               20.00006657,  20.00138668],
             [ 19.9980034 ,  20.00138827,  19.99769861,  19.99813923,
               19.99828983,  19.99986577],
             [ 20.00019011,  19.99762264,  19.99933936,  19.99862503,
               20.00011846,  20.00196553],
             [ 20.00143478,  19.99987261,  20.00033801,  19.99798902,
               19.99739315,  19.99648102]])
        intensity_red['Shannen'] = numpy.array(
            [[ 20.00037826,  20.00104538,  20.00075668,  19.99890644,
               19.99898773,  19.99763399],
             [ 19.99722143,  20.00211497,  19.99831262,  20.00252405,
               20.00117685,  19.99830147],
             [ 19.99740698,  19.99702128,  20.00205764,  20.00165859,
               20.00127337,  20.00082943],
             [ 20.00098172,  19.99950118,  19.9996026 ,  20.00290383,
               19.99950888,  20.0028386 ]])
        intensity_red['Jennie'] = numpy.array(
            [[ 20.0033587 ,  20.00283873,  19.9974105 ,  19.99818273,
               19.99625588,  19.99734283],
             [ 19.99981404,  20.00334072,  19.99949528,  20.00307649,
               20.00123727,  20.00312434],
             [ 20.00174683,  20.00179645,  20.00171476,  20.00148666,
               20.00248301,  19.99941952],
             [ 19.99976357,  20.00283844,  19.99842876,  20.00250757,
               20.00273813,  20.00167966]])
        intensity_red['Kirk'] = numpy.array(
            [[ 19.99835111,  19.9999909 ,  20.00278722,  19.99829841,
               19.99822603,  19.99957096],
             [ 19.99647856,  19.99866099,  19.99812838,  19.99981953,
               19.99771783,  19.99903998],
             [ 19.9985312 ,  19.99865002,  20.00093124,  19.99785189,
               20.00359071,  19.99981056],
             [ 20.00049272,  20.0035065 ,  20.00397526,  20.00065059,
               20.00172026,  19.99746033]])
        intensity_red['Picard'] = numpy.array(
            [[ 19.99898939,  19.99813776,  20.00165375,  20.00076093,
               20.00229376,  19.99978131],
             [ 20.00170884,  20.00120388,  20.00221023,  19.99997127,
               20.00082848,  20.0022574 ],
             [ 19.9974744 ,  19.99907881,  19.99841985,  20.00156132,
               19.99812349,  20.00163131],
             [ 19.99686673,  20.00214557,  20.00316727,  19.99733969,
               19.99729137,  19.99843903]])
        # We will only check green intensity in one well per plate
        # The following specifies the coordinates of the chosen well on
        # each plate
        coordinates = {'Tori':    numpy.array([0,5]),
                       'Tiffani': numpy.array([0,0]),
                       'Shannen': numpy.array([2,3]),
                       'Jennie':  numpy.array([1,4]),
                       'Kirk':    numpy.array([3,0]),
                       'Picard':  numpy.array([1,5]),
                       }
        # The following specifies the intensities
        intensity_green = {}
        intensity_green['Tori'] = numpy.array(
          [  5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,   5.00275022,   5.00275022,   5.00275022,
             5.00275022,  11.99960369,  12.31446209,  12.6293205 ,
            12.9441789 ,  13.25204046,  13.56689886,  13.87476041,
            14.18262197,  14.48348667,  14.79134822,  15.09221292,
            15.38608076,  15.67994861,   0.        ])
        intensity_green['Tiffani'] = numpy.array(
          [  4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,   4.99725237,
             4.99725237,   4.99725237,   4.99725237,  12.00011342,
            12.31537632,  12.63063922,  12.9391944 ,  13.2544573 ,
            13.56301248,  13.87156766,  14.18012284,  14.48867802,
            14.79052548,  15.09237294,  15.38751267,  15.68265241,
            15.97108443,  16.25951644,  16.54124074,  16.81625731,
            17.09127389,  17.35958274,   0.        ])
        intensity_green['Shannen'] = numpy.array(
          [  4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,   4.99717009,
             4.99717009,   4.99717009,   4.99717009,  11.99848226,
            12.31492575,  12.62477667,   0.        ])
        intensity_green['Jennie'] = numpy.array(
          [  5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
             5.00288299,   5.00288299,   5.00288299,   5.00288299,
            11.99872893,  12.31268884,  12.62664876,  12.94060867,
            13.25456859,  13.56170329,  13.8756632 ,  14.1827979 ,
            14.4899326 ,  14.79024209,  15.09055157,  15.38403584,
            15.68434532,  15.97100438,  16.25766343,  16.53749727,
            16.8173311 ,  17.09033973,  17.35652313,  17.62270654,
            17.87523951,  18.12777249,  18.37348025,  18.61236279,
            18.84442012,  19.06965223,  19.28805913,  19.49964081,
            19.70439728,  19.90232853,  20.09343457,  20.27089017,
            20.44152056,  20.60532573,  20.76230569,  20.91246043,
            21.04896474,  21.17864384,  21.2946725 ,  21.41070117,
            21.5130794 ,  21.6018072 ,  21.68370979,  21.75878716,
            21.8202141 ,  21.87481582,  21.92259233,  21.95671841,
            21.97719406,  21.9976697 ,  21.9976697 ,  21.9976697 ,
            21.97719406,  21.95671841,  21.92259233,  21.87481582,
            21.8202141 ,  21.75878716,  21.68370979,  21.6018072 ,
            21.5130794 ,  21.41070117,  21.2946725 ,  21.17864384,
            21.04896474,  20.91246043,  20.76230569,  20.60532573,
            20.44152056,  20.27089017,  20.09343457,  19.90232853,
            19.70439728,  19.49964081,  19.28805913,  19.06965223,
            18.84442012,  18.61236279,  18.37348025,  18.12777249,
            17.87523951,  17.62270654,  17.35652313,  17.09033973,
            16.8173311 ,  16.53749727,  16.25766343,  15.97100438,
            15.68434532,  15.38403584,  15.09055157,  14.79024209,
            14.4899326 ,  14.1827979 ,  13.8756632 ,  13.56170329,
            13.25456859,  12.94060867,  12.62664876,  12.31268884,
            11.99872893,  11.68476901,  11.3708091 ,  11.05684918,
            10.74971448,  10.43575457,  10.12861987,   9.82148517,
             9.51435047,   9.20721577,   8.90690629,   8.61342202,
             8.31993775,   8.02645348,   7.73979443,   7.45996059,
             7.18012675,   6.90711813,   0.        ])
        intensity_green['Kirk'] = numpy.array(
          [  5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,   5.00177758,   5.00177758,
             5.00177758,   5.00177758,  12.00017754,  12.31364044,
            12.62710334,  12.94056624,  13.25402914,  13.56749204,
            13.87414053,  14.18078902,  14.48743751,  14.78727159,
            15.08710567,  15.38693975,  15.67995941,  15.97297908,
            16.25918434,  16.53857518,  16.81796603,  17.09054246,
            17.35630449,  17.62206651,  17.88101412,  18.12633292,
            18.37165171,  18.61015609,  18.84866047,  19.07353603,
            19.29159717,  19.50284391,  19.70727624,  19.90489415,
            20.08888325,  20.27287234,  20.44323261,  20.60677847,
            20.76350992,  20.91342696,  21.04971518,  21.17918899,
            21.29503397,  21.41087896,  21.51309512,  21.60168246,
            21.68345539,  21.75841391,  21.81974361,  21.87425889,
            21.92195977,  21.95603183,  21.98328947,  21.99691829,
            21.99691829,  21.99691829,  21.98328947,  21.95603183,
            21.92195977,  21.87425889,  21.81974361,  21.75841391,
            21.68345539,  21.60168246,  21.51309512,  21.41087896,
            21.29503397,  21.17918899,  21.04971518,  20.91342696,
            20.76350992,  20.60677847,  20.44323261,  20.27287234,
            20.08888325,  19.90489415,   0.        ])
        intensity_green['Picard'] = numpy.array(
          [  4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,   4.99853183,   4.99853183,
             4.99853183,   4.99853183,  11.99902667,  12.31143491,
            12.63021882,  12.94262706,  13.2550353 ,  13.56744354,
            13.8734761 ,  14.17950867,  14.48554123,  14.79157379,
            15.09123067,  15.38451187,  15.68416876,  15.97107428,
            16.25797981,  16.53850966,  16.81903951,  17.09319367,
            17.36097217,  17.62237498,  17.87740211,  18.12605357,
            18.37470502,  18.61060512,  18.84650522,  19.06965397,
            19.29280271,  19.50320009,  19.7072218 ,  19.89849215,
            20.0897625 ,  20.2682815 ,  20.44042481,  20.60619245,
            20.76558441,  20.91222501,  21.04611426,  21.1800035 ,
            21.29476571,  21.40952792,  21.51153878,  21.60079827,
            21.68368209,  21.76019023,  21.82394702,  21.87495244,
            21.91958219,  21.95783626,  21.98333897,  21.99609033,
            22.00246601,  21.99609033,  21.98333897,  21.95783626,
            21.91958219,  21.87495244,  21.82394702,  21.76019023,
            21.68368209,  21.60079827,  21.51153878,  21.40952792,
            21.29476571,  21.1800035 ,  21.04611426,  20.91222501,
            20.76558441,  20.60619245,  20.44042481,  20.2682815 ,
            20.0897625 ,  19.89849215,  19.7072218 ,  19.50320009,
            19.29280271,  19.06965397,  18.84650522,  18.61060512,
            18.37470502,  18.12605357,  17.87740211,  17.62237498,
            17.36097217,  17.09319367,  16.81903951,  16.53850966,
            16.25797981,  15.97107428,   0.        ])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000)
            self.assertEqual(lpa.intensity.shape[0], 251)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: red channel
            numpy.testing.assert_almost_equal(
                lpa.intensity[:250,:,:,1],
                numpy.tile(intensity_red[lpa_name], (250,1,1)))
            # Intensity: green channel
            coords_lpa = coordinates[lpa_name]
            numpy.testing.assert_almost_equal(
                lpa.intensity[:,coords_lpa[0],coords_lpa[1],0],
                intensity_green[lpa_name])
            # Intensity: last frame
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_light_signal_rows_cols(self):
        # Make plate array
        p = lpadesign.plate.LPAPlateArray(name='PA1',
                                          array_n_rows=2,
                                          array_n_cols=3,
                                          plate_names=['P{}'.format(i+1)
                                                       for i in range(6)])
        p.resources['LPA'] = ['Tori',
                              'Tiffani',
                              'Shannen',
                              'Jennie',
                              'Kirk',
                              'Picard']
        # Make inducers
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.array([ 3,  9, 12, 18, 16,  7,
                                              8, 14, 11, 19, 10,  6,
                                              4, 15, 17,  5, 20, 13])
        p.apply_inducer(light_520, 'rows')

        light_660 = lpadesign.inducer.StaggeredLightSignal(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        # Write sampling times, signal, etc
        light_660.signal = self.signal
        light_660.signal_init = self.signal_init
        light_660.n_time_steps = self.n_time_steps
        light_660.sampling_time_steps = numpy.array([ 34, 155, 200, 164,
                                                      63,  21, 102, 183])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 3*numpy.ones((4,6)),
                                          7*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          7*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        # Expected intensities
        intensity_green = {}
        intensity_green['Tori'] = numpy.array(
              [[  3.00122676,   8.99840934,  12.00207294,  17.99944501,
                 15.99761037,   7.00035189],
               [  3.00154443,   9.00201225,  11.99902147,  18.00024996,
                 16.0007696 ,   7.00105535],
               [  3.00022384,   8.99975124,  12.00117286,  17.99910341,
                 15.99995828,   7.00106816],
               [  3.00156701,   8.99793022,  12.0009139 ,  18.00039669,
                 15.9993679 ,   7.0018766 ]])
        intensity_green['Tiffani'] = numpy.array(
              [[  7.99895766,  14.0024914 ,  10.99900043,  18.99962248,
                  9.99951414,   6.00115403],
               [  8.0010562 ,  13.99935256,  10.99870858,  19.00124298,
                 10.00189558,   5.99862513],
               [  7.99777081,  13.99815478,  11.00174834,  18.99771176,
                  9.99846556,   6.00159872],
               [  8.00233506,  14.00234413,  10.99906616,  18.99904434,
                 10.00191126,   6.0006131 ]])
        intensity_green['Shannen'] = numpy.array(
              [[  3.99959473,  14.9995695 ,  16.99977485,   4.99731087,
                 20.00119698,  13.00222986],
               [  3.99877874,  15.00172159,  17.00156084,   5.00193377,
                 19.99879302,  12.99915308],
               [  3.99744805,  15.00158322,  17.00193243,   4.99717009,
                 19.99848449,  12.99902022],
               [  4.00183995,  14.9991082 ,  16.99891201,   4.99909126,
                 19.99837409,  13.00028124]])
        intensity_green['Jennie'] = numpy.array(
              [[  3.00058171,   9.0021011 ,  11.99940261,  17.99820965,
                 15.99910438,   7.00124665],
               [  2.99763105,   8.99872617,  12.0015751 ,  18.00033965,
                 16.00171785,   7.00011562],
               [  3.0015932 ,   8.9985666 ,  11.99971068,  17.99855283,
                 16.00142793,   6.99788385],
               [  2.99994887,   8.99794377,  12.00220467,  18.00103111,
                 15.99918471,   7.00172089]])
        intensity_green['Kirk'] = numpy.array(
              [[  8.00048728,  14.00167924,  11.00192195,  19.00231816,
                  9.99770114,   5.99900056],
               [  7.99990768,  14.00049248,  10.99941874,  18.99755728,
                  9.99776876,   6.00094267],
               [  7.99961532,  14.00194254,  10.99842258,  18.99741554,
                 10.00077766,   6.00227918],
               [  7.99841476,  14.00053015,  11.00067501,  19.00108849,
                  9.99853032,   6.00152107]])
        intensity_green['Picard'] = numpy.array(
              [[  3.99903334,  14.99798174,  17.00225243,   4.99915397,
                 19.99952816,  12.99899432],
               [  4.0018762 ,  14.99889641,  17.00139244,   5.00220569,
                 20.00196701,  13.00160209],
               [  4.00249787,  15.00224886,  16.99932637,   4.99864414,
                 20.00237459,  12.99922181],
               [  4.00036777,  14.99928653,  16.99741191,   5.00226785,
                 19.99866726,  12.99831301]])
        # We will only check red intensity in one well per plate
        # The following specifies the coordinates of the chosen well on
        # each plate
        coordinates = {'Tori':    numpy.array([1,1]),
                       'Tiffani': numpy.array([3,2]),
                       'Shannen': numpy.array([0,0]),
                       'Jennie':  numpy.array([3,4]),
                       'Kirk':    numpy.array([0,4]),
                       'Picard':  numpy.array([1,1]),
                       }
        # The following specifies the intensities
        intensity_red = {}
        intensity_red['Tori'] = numpy.array(
          [  5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,   5.00092558,
             5.00092558,   5.00092558,   5.00092558,  12.00388282,
            12.31124868,  12.62692172,  12.94259477,  13.24996062,
            13.56563367,  13.87299952,  14.18036538,  14.48773124,
            14.78678991,  15.09415577,  15.38490726,  15.68396593,
            15.97471742,  16.25716172,  16.53960602,  16.81374314,
            17.08788025,  17.36201737,  17.61954011,  17.87706286,
            18.12627842,  18.37549398,  18.61640236,  18.84900355,
            19.07329755,  19.28928437,  19.496964  ,  19.70464364,
            19.90401608,  20.08677416,  20.26953224,  20.44398313,
            20.61012684,  20.75965617,  20.90918551,  21.05040766,
            21.17501544,  21.29962322,  21.40761663,  21.50730286,
            21.60698908,  21.68175375,  21.75651842,  21.8229759 ,
            21.87281901,  21.92266212,  21.95589086,  21.98081242,
            21.99742679,  21.99742679,  21.99742679,  21.98081242,
            21.95589086,  21.92266212,  21.87281901,  21.8229759 ,
            21.75651842,  21.68175375,  21.60698908,  21.50730286,
            21.40761663,  21.29962322,  21.17501544,  21.05040766,
            20.90918551,  20.75965617,  20.61012684,  20.44398313,
            20.26953224,  20.08677416,  19.90401608,  19.70464364,
            19.496964  ,  19.28928437,  19.07329755,  18.84900355,
            18.61640236,  18.37549398,  18.12627842,  17.87706286,
            17.61954011,  17.36201737,  17.08788025,  16.81374314,
            16.53960602,  16.25716172,  15.97471742,  15.68396593,
            15.38490726,  15.09415577,  14.78678991,  14.48773124,
            14.18036538,  13.87299952,  13.56563367,  13.24996062,
            12.94259477,  12.62692172,  12.31124868,  12.00388282,
            11.68820978,  11.37253674,  11.05686369,  10.74949784,
            10.43382479,  10.12645893,   9.81909308,   9.51172722,
             9.21266855,   8.91360987,   8.6145512 ,   8.31549253,
             8.02474104,   7.74229674,   7.45985244,   7.18571532,
             6.91157821,   6.64574828,   6.37991834,   6.1223956 ,
             5.87318004,   5.62396448,   5.3830561 ,   5.15045491,
             4.92616091,   4.71017409,   4.50249446,   4.29481482,
             4.09544237,   3.9126843 ,   3.72992622,   3.55547533,
             3.38933162,   3.23980228,   3.09027295,   2.9490508 ,
             2.82444302,   2.69983524,   2.59184183,   2.4921556 ,
             2.40077656,   2.31770471,   2.24294004,   2.17648256,
             2.12663945,   2.07679634,   2.04356759,   2.01864604,
             2.00203167,   2.00203167,   2.00203167,   2.01864604,
             2.04356759,   2.07679634,   0.        ])
        intensity_red['Tiffani'] = numpy.array(
          [  5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,   5.00060292,   5.00060292,
             5.00060292,   5.00060292,  11.99709234,  12.31643418,
            12.62851825,  12.94060232,  13.25268639,  13.56477046,
            13.87685454,  14.18168084,  14.48650714,  14.79133344,
            15.08890198,  15.38647051,  15.68403904,  15.97434981,
            16.2574028 ,  16.5404558 ,  16.81625102,  17.09204625,
            17.36058371,  17.62186339,  17.87588531,  18.12990723,
            18.37667138,  18.61617776,  18.84842637,  19.07341722,
            19.29115029,  19.50162559,  19.70484313,  19.90080289,
            20.08950489,  20.27094912,  20.44513558,  20.6048065 ,
            20.76447742,  20.9096328 ,  21.04753041,  21.17817026,
            21.29429456,  21.41041887,  21.51202764,  21.60637864,
            21.6862141 ,  21.75879179,  21.82411171,  21.87491609,
            21.91846271,  21.95475155,  21.98378263,  21.99829817,
            21.99829817,  21.99829817,  21.98378263,  21.95475155,
            21.91846271,  21.87491609,  21.82411171,  21.75879179,
            21.6862141 ,  21.60637864,  21.51202764,  21.41041887,
            21.29429456,  21.17817026,  21.04753041,  20.9096328 ,
            20.76447742,  20.6048065 ,  20.44513558,  20.27094912,
            20.08950489,  19.90080289,  19.70484313,  19.50162559,
            19.29115029,  19.07341722,  18.84842637,  18.61617776,
            18.37667138,  18.12990723,  17.87588531,  17.62186339,
            17.36058371,  17.09204625,  16.81625102,  16.5404558 ,
            16.2574028 ,  15.97434981,  15.68403904,  15.38647051,
            15.08890198,  14.79133344,  14.48650714,  14.18168084,
            13.87685454,  13.56477046,  13.25268639,  12.94060232,
            12.62851825,  12.31643418,  11.99709234,  11.68500826,
            11.37292419,  11.06084012,  10.74875605,  10.43667198,
            10.12458791,   9.8197616 ,   9.5149353 ,   9.210109  ,
             8.91254046,   8.61497193,   8.3174034 ,   8.02709263,
             7.74403964,   7.46098664,   7.18519142,   6.90939619,
             6.64085873,   6.37957905,   6.12555713,   5.87153521,
             5.62477106,   5.38526468,   5.15301607,   4.92802522,
             4.71029215,   4.49981685,   4.29659931,   4.10063955,
             3.91193755,   3.73049332,   3.55630686,   3.38937817,
             3.23696502,   3.09180964,   2.95391203,   2.82327218,
             2.69989011,   2.59102357,   2.4894148 ,   2.39506381,
             2.31522835,   2.24265065,   2.17733073,   2.12652635,
             2.07572196,   2.04669089,   2.01765981,   2.00314427,
             2.00314427,   2.00314427,   2.01765981,   2.04669089,
             2.07572196,   2.12652635,   2.17733073,   2.24265065,
             2.31522835,   2.39506381,   2.4894148 ,   2.59102357,
             2.69989011,   2.82327218,   0.        ])
        intensity_red['Shannen'] = numpy.array(
          [  5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
             5.0006518 ,   5.0006518 ,   5.0006518 ,   5.0006518 ,
            11.9984438 ,  12.31049695,  12.63035143,  12.94240459,
            13.25445774,  13.5665109 ,  13.87076272,  14.18281588,
            14.4870677 ,  14.79131953,  15.08777003,  15.38422052,
            15.68067102,  15.96932019,  16.25796936,  16.5388172 ,
            16.81966503,  17.09271154,  17.35795673,  17.62320191,
            17.88064576,  18.13028828,  18.37212948,  18.61397067,
            18.84801054,  19.07424908,  19.29268629,  19.50332216,
            19.70615671,  19.90118994,  20.08842183,  20.26785239,
            20.43948163,  20.61111086,   0.        ])
        intensity_red['Jennie'] = numpy.array(
          [  4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,   4.99868825,
             4.99868825,   4.99868825,   4.99868825,  12.00164288,
            12.31306275,  12.62448263,  12.94388763,  13.2553075 ,
            13.56672738,  13.87016213,  14.181582  ,  14.48501675,
            14.7884515 ,  15.09188625,  15.38733588,  15.6827855 ,
            15.97025   ,  16.2577145 ,  16.53719388,  16.81667325,
            17.0881675 ,  17.35966175,  17.62317088,  17.87869488,
            18.12623375,  18.37377263,  18.61332638,  18.844895  ,
            19.0684785 ,  19.292062  ,  19.49967525,  19.7072885 ,
            19.8989315 ,  20.0905745 ,  20.27423238,  20.44192   ,
            20.60960763,  20.761325  ,  20.91304238,  21.0487895 ,
            21.1765515 ,  21.29632838,  21.40812013,  21.51192675,
            21.59976313,  21.6875995 ,  21.75946563,  21.82334663,
            21.8792425 ,  21.91916813,  21.95909375,  21.98304913,
            21.99901938,  21.99901938,  21.99901938,  21.98304913,
            21.95909375,  21.91916813,  21.8792425 ,  21.82334663,
            21.75946563,  21.6875995 ,  21.59976313,  21.51192675,
            21.40812013,  21.29632838,  21.1765515 ,  21.0487895 ,
            20.91304238,  20.761325  ,  20.60960763,  20.44192   ,
            20.27423238,  20.0905745 ,  19.8989315 ,  19.7072885 ,
            19.49967525,  19.292062  ,  19.0684785 ,  18.844895  ,
            18.61332638,  18.37377263,  18.12623375,  17.87869488,
            17.62317088,  17.35966175,  17.0881675 ,  16.81667325,
            16.53719388,  16.2577145 ,  15.97025   ,  15.6827855 ,
            15.38733588,  15.09188625,  14.7884515 ,  14.48501675,
            14.181582  ,  13.87016213,  13.56672738,  13.2553075 ,
            12.94388763,  12.62448263,  12.31306275,  12.00164288,
            11.68223788,  11.370818  ,  11.05939813,  10.74797825,
            10.43655838,  10.1251385 ,   9.82170375,   9.51028388,
             9.20684913,   8.9113995 ,   8.61594988,   8.32050025,
             8.02505063,   7.74557125,   7.45810675,   7.17862738,
             6.90713313,   6.643624  ,   6.38011488,   6.12459088,
             5.86906688,   5.62951313,   5.38995938,   5.15839075,
             4.92682213,   4.71122375,   4.49562538,   4.29599725,
             4.09636913,   3.91271125,   3.72905338,   3.55338063,
             3.39367813,   3.23397563,   3.09024338,   2.95449625,
             2.81874913,   2.69897225,   2.59516563,   2.491359  ,
             2.3955375 ,   2.31568625,   2.24382013,   2.17993913,
             2.12404325,   2.0761325 ,   2.044192  ,   2.02023663,
             2.00426638,   1.99628125,   2.00426638,   2.02023663,
             2.044192  ,   2.0761325 ,   2.12404325,   2.17993913,
             2.24382013,   2.31568625,   2.3955375 ,   2.491359  ,
             2.59516563,   2.69897225,   2.81874913,   2.95449625,
             3.09024338,   3.23397563,   3.39367813,   3.55338063,
             3.72905338,   3.91271125,   4.09636913,   4.29599725,
             4.49562538,   4.71122375,   4.92682213,   5.15839075,
             5.38995938,   5.62951313,   5.86906688,   6.12459088,
             6.38011488,   6.643624  ,   0.        ])
        intensity_red['Kirk'] = numpy.array(
          [  4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,   4.99756386,
             4.99756386,   4.99756386,   4.99756386,  12.00371798,
            12.31457123,  12.62542449,  12.94424834,  13.25510159,
            13.56595485,  13.8768081 ,  14.17969076,  14.49054401,
            14.79342667,  15.08833873,  15.39122139,  15.67816286,
            15.97307492,  16.26001639,  16.53898725,  16.81795812,
            17.0889584 ,  17.35995867,  17.62298835,  17.87804743,
            18.12513591,  18.3722244 ,  18.61134228,  18.84248958,
            19.07363687,  19.28884297,  19.50404907,  19.70331397,
            19.90257888,  20.09387319,  20.26922631,  20.44457943,
            20.60399135,  20.76340328,  20.90687401,  21.05034474,
            21.17787428,  21.29743323,  21.40902157,  21.51263933,
            21.60031588,  21.68799244,  21.75972781,  21.82349258,
            21.87928675,  21.91913973,  21.95899272,  21.98290451,
            21.9988457 ,  21.9988457 ,  21.9988457 ,  21.98290451,
            21.95899272,  21.91913973,  21.87928675,  21.82349258,
            21.75972781,  21.68799244,  21.60031588,  21.51263933,
            21.40902157,  21.29743323,   0.        ])
        intensity_red['Picard'] = numpy.array(
          [  5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,   5.00146654,   5.00146654,   5.00146654,
             5.00146654,  12.0018879 ,  12.31192987,  12.63013084,
            12.94017282,  13.25021479,  13.56841576,  13.87029873,
            14.1803407 ,  14.49038267,  14.79226565,  15.09414862,
            15.38787259,  15.68159656,  15.97532054,  16.26088551,
            16.53829149,  16.81569746,  17.09310344,  17.35419141,
            17.62343839,  17.87636736,   0.        ])
        # Load LPA files and compare
        lpa_names = ['Tori',
                     'Tiffani',
                     'Shannen',
                     'Jennie',
                     'Kirk',
                     'Picard']
        for lpa_name in lpa_names:
            lpa = lpaprogram.LPA(name=lpa_name,
                                 layout_names=['520-2-KB', '660-LS'])
            lpa.load_files(os.path.join(self.temp_dir, 'LPA Files/' + lpa_name))
            # Dimensions
            self.assertEqual(lpa.n_rows, 4)
            self.assertEqual(lpa.n_cols, 6)
            self.assertEqual(lpa.n_channels, 2)
            self.assertEqual(lpa.step_size, 60000)
            self.assertEqual(lpa.intensity.shape[0], 251)
            # Dot correction
            numpy.testing.assert_almost_equal(lpa.dc, dc_exp[lpa_name])
            # Grayscale calibration
            gcal_exp = numpy.ones((4, 6, 2))*255
            numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
            # Intensity: green channel
            numpy.testing.assert_almost_equal(
                lpa.intensity[:250,:,:,0],
                numpy.tile(intensity_green[lpa_name], (250,1,1)))
            # Intensity: green channel
            coords_lpa = coordinates[lpa_name]
            numpy.testing.assert_almost_equal(
                lpa.intensity[:,coords_lpa[0],coords_lpa[1],1],
                intensity_red[lpa_name])
            # Intensity: last frame
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))
