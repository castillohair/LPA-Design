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
                                          numpy.zeros((4, 6, 2)))
        numpy.testing.assert_almost_equal(p.lpa.gcal,
                                          numpy.ones((4, 6, 2))*255)
        numpy.testing.assert_almost_equal(p.lpa.intensity,
                                          numpy.zeros((1, 4, 6, 2)))
        self.assertEqual(p.lpa_optimize_dc, [False, False])
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
                                          numpy.zeros((8, 12, 4)))
        numpy.testing.assert_almost_equal(p.lpa.gcal,
                                          numpy.ones((8, 12, 4))*255)
        numpy.testing.assert_almost_equal(p.lpa.intensity,
                                          numpy.zeros((1, 8, 12, 4)))
        self.assertEqual(p.lpa_optimize_dc, [False, False, False, False])
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        light_660.intensities = [5, 10, 15, 20,]
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
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   5.00225094],
                                       [  0.99775886,   5.00043956],
                                       [  2.00071772,   4.99834234],
                                       [  2.99970161,   4.99721923],
                                       [  3.99813213,   5.00120694],
                                       [  4.99752286,   5.00286667]],

                                      [[  0.        ,  10.00025001],
                                       [  0.99889055,  10.00265584],
                                       [  2.00026252,  10.00123339],
                                       [  3.00089391,  10.00158622],
                                       [  3.99957631,   9.99911403],
                                       [  4.99745787,   9.99889423]],

                                      [[  0.        ,  14.99966057],
                                       [  1.00040402,  14.99750859],
                                       [  1.9991104 ,  15.00194049],
                                       [  3.00279193,  14.99714238],
                                       [  4.00120469,  14.9973051 ],
                                       [  5.0028731 ,  15.00218866]],

                                      [[  0.        ,  20.00120669],
                                       [  0.99922729,  19.99863112],
                                       [  2.00206989,  19.9976445 ],
                                       [  3.00099994,  20.00258099],
                                       [  3.99822207,  19.99775917],
                                       [  4.99792806,  19.99978527]]],


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
        light_660.intensities = [5, 10, 15, 20,]
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
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   5.00225094],
                                       [  0.99775886,   5.00043956],
                                       [  2.00071772,   4.99834234],
                                       [  2.99970161,   4.99721923],
                                       [  3.99813213,   5.00120694],
                                       [  4.99752286,   5.00286667]],

                                      [[  0.        ,  10.00025001],
                                       [  0.99889055,  10.00265584],
                                       [  2.00026252,  10.00123339],
                                       [  3.00089391,  10.00158622],
                                       [  3.99957631,   9.99911403],
                                       [  4.99745787,   9.99889423]],

                                      [[  0.        ,  14.99966057],
                                       [  1.00040402,  14.99750859],
                                       [  1.9991104 ,  15.00194049],
                                       [  3.00279193,  14.99714238],
                                       [  4.00120469,  14.9973051 ],
                                       [  5.0028731 ,  15.00218866]],

                                      [[  0.        ,  20.00120669],
                                       [  0.99922729,  19.99863112],
                                       [  2.00206989,  19.9976445 ],
                                       [  3.00099994,  20.00258099],
                                       [  3.99822207,  19.99775917],
                                       [  4.99792806,  19.99978527]]],


                                     [[[  0.        ,   5.00225094],
                                       [  0.99775886,   5.00043956],
                                       [  2.00071772,   4.99834234],
                                       [  2.99970161,   4.99721923],
                                       [  3.99813213,   5.00120694],
                                       [  4.99752286,   5.00286667]],

                                      [[  0.        ,  10.00025001],
                                       [  0.99889055,  10.00265584],
                                       [  2.00026252,  10.00123339],
                                       [  3.00089391,  10.00158622],
                                       [  3.99957631,   9.99911403],
                                       [  4.99745787,   9.99889423]],

                                      [[  0.        ,  14.99966057],
                                       [  1.00040402,  14.99750859],
                                       [  1.9991104 ,  15.00194049],
                                       [  3.00279193,  14.99714238],
                                       [  4.00120469,  14.9973051 ],
                                       [  5.0028731 ,  15.00218866]],

                                      [[  0.        ,  20.00120669],
                                       [  0.99922729,  19.99863112],
                                       [  2.00206989,  19.9976445 ],
                                       [  3.00099994,  20.00258099],
                                       [  3.99822207,  19.99775917],
                                       [  4.99792806,  19.99978527]]]])
        numpy.testing.assert_almost_equal(lpa.intensity, intensity_exp)

    def test_save_rep_setup_files_rows_and_cols_full_dc_optimization(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.lpa_optimize_dc = [True, True]
        p.lpa_optimize_dc_uniform = [False, False]
        p.lpa.set_all_gcal(255)
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
        dc_exp = numpy.array([[[ 1,  4],
                               [ 1,  4],
                               [ 1,  4],
                               [ 1,  4],
                               [ 1,  4],
                               [ 1,  3]],

                              [[ 1,  7],
                               [ 1,  6],
                               [ 1,  7],
                               [ 1,  7],
                               [ 1,  7],
                               [ 1,  6]],

                              [[ 1, 10],
                               [ 1, 10],
                               [ 1,  9],
                               [ 1,  9],
                               [ 1, 10],
                               [ 1, 10]],

                              [[ 1, 12],
                               [ 1, 13],
                               [ 1, 13],
                               [ 1, 13],
                               [ 1, 13],
                               [ 1, 15]]])
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]])
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   9.9996897 ],
                                       [  1.00072838,  10.00087913],
                                       [  2.00071772,  10.00150392],
                                       [  2.99970161,   9.9992243 ],
                                       [  3.9997761 ,  10.00079928],
                                       [  5.00066595,  10.0008044 ]],

                                      [[  0.        ,  20.00050003],
                                       [  1.00034242,  19.99877934],
                                       [  2.00026252,  20.00246677],
                                       [  2.99921931,  19.99751222],
                                       [  3.99957631,  19.99822807],
                                       [  5.00079841,  20.00104781]],

                                      [[  0.        ,  29.99771183],
                                       [  1.00040402,  29.99821614],
                                       [  2.00079316,  30.00223558],
                                       [  2.9997588 ,  29.99923269],
                                       [  3.99950927,  29.99780315],
                                       [  4.99980386,  30.00277375]],

                                      [[  0.        ,  40.00404473],
                                       [  0.99922729,  39.99967047],
                                       [  2.00036745,  39.99850071],
                                       [  2.99934376,  40.00354666],
                                       [  3.99979618,  40.00026915],
                                       [  4.99957863,  40.00438081]]],

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

    def test_save_rep_setup_files_rows_and_cols_uniform_dc_optimization(self):
        # Make plate
        p = lpadesign.plate.LPAPlate(name='P1')
        p.resources['LPA'] = ['Jennie']
        p.lpa_optimize_dc = [True, True]
        p.lpa_optimize_dc_uniform = [True, True]
        p.lpa.set_all_gcal(255)
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
        dc_exp[:,:,1] = 15
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,   9.99728362],
                                       [  1.00072838,  10.00489232],
                                       [  2.00071772,   9.99989751],
                                       [  2.99970161,  10.00241487],
                                       [  3.9997761 ,  10.00241387],
                                       [  5.00066595,  10.00573334]],

                                      [[  0.        ,  20.0053264 ],
                                       [  1.00034242,  20.00122897],
                                       [  2.00026252,  19.99679392],
                                       [  2.99921931,  20.00074663],
                                       [  3.99957631,  19.99744061],
                                       [  5.00079841,  19.99615877]],

                                      [[  0.        ,  30.00575842],
                                       [  1.00040402,  30.00221483],
                                       [  2.00079316,  29.99976748],
                                       [  2.9997588 ,  29.99675873],
                                       [  3.99950927,  30.00578554],
                                       [  4.99980386,  29.99475591]],

                                      [[  0.        ,  39.99670364],
                                       [  0.99922729,  40.0004732 ],
                                       [  2.00036745,  39.99769778],
                                       [  2.99934376,  40.00354666],
                                       [  3.99979618,  40.00185275],
                                       [  4.99957863,  40.00438081]]],

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
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  19.99777535],
                                       [  0.99775886,  20.00175826],
                                       [  2.00071772,  19.99899181],
                                       [  2.99970161,  20.00004389],
                                       [  3.99813213,  19.99917667],
                                       [  4.99752286,  19.99996583]],

                                      [[  6.00179289,  20.00050003],
                                       [  6.99804133,  19.99959588],
                                       [  8.00105007,  20.00246677],
                                       [  9.00268173,  19.99751222],
                                       [  9.99894077,  19.99822807],
                                       [ 10.99707975,  19.99778845]],

                                      [[ 11.99983339,  20.00142497],
                                       [ 12.99849282,  20.00227629],
                                       [ 14.00050384,  20.00066769],
                                       [ 15.00182715,  20.00196242],
                                       [ 15.99803708,  19.99826936],
                                       [ 16.99749156,  19.99730572]],

                                      [[ 17.9996932 ,  20.00120669],
                                       [ 18.99838029,  19.99863112],
                                       [ 20.00026957,  19.9976445 ],
                                       [ 21.00037487,  20.00258099],
                                       [ 21.99966602,  19.99775917],
                                       [ 23.0023532 ,  19.99978527]]],

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
        dc_exp[:,:,1] = 7
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  19.99777535],
                                       [  0.99775886,  20.00175826],
                                       [  2.00071772,  19.99899181],
                                       [  2.99970161,  20.00004389],
                                       [  3.99813213,  19.99917667],
                                       [  4.99752286,  19.99996583]],

                                      [[  6.00179289,  20.00050003],
                                       [  6.99804133,  19.99959588],
                                       [  8.00105007,  20.00246677],
                                       [  9.00268173,  19.99751222],
                                       [  9.99894077,  19.99822807],
                                       [ 10.99707975,  19.99778845]],

                                      [[ 11.99983339,  20.00142497],
                                       [ 12.99849282,  20.00227629],
                                       [ 14.00050384,  20.00066769],
                                       [ 15.00182715,  20.00196242],
                                       [ 15.99803708,  19.99826936],
                                       [ 16.99749156,  19.99730572]],

                                      [[ 17.9996932 ,  20.00120669],
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
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  0.        ],
                                       [  0.99775886,  0.        ],
                                       [  2.00071772,  0.        ],
                                       [  2.99970161,  0.        ],
                                       [  3.99813213,  0.        ],
                                       [  4.99752286,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99889055,  0.        ],
                                       [  2.00026252,  0.        ],
                                       [  3.00089391,  0.        ],
                                       [  3.99957631,  0.        ],
                                       [  4.99745787,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00040402,  0.        ],
                                       [  1.9991104 ,  0.        ],
                                       [  3.00279193,  0.        ],
                                       [  4.00120469,  0.        ],
                                       [  5.0028731 ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99922729,  0.        ],
                                       [  2.00206989,  0.        ],
                                       [  3.00099994,  0.        ],
                                       [  3.99822207,  0.        ],
                                       [  4.99792806,  0.        ]]],


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
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 0
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.ones((4, 6, 2))*255
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Intensity includes a single frame with the appropriate intensities,
        # plus a dark frame.
        intensity_exp = numpy.array([[[[  0.        ,  0.        ],
                                       [  0.99775886,  0.        ],
                                       [  2.00071772,  0.        ],
                                       [  2.99970161,  0.        ],
                                       [  3.99813213,  0.        ],
                                       [  4.99752286,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99889055,  0.        ],
                                       [  2.00026252,  0.        ],
                                       [  3.00089391,  0.        ],
                                       [  3.99957631,  0.        ],
                                       [  4.99745787,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  1.00040402,  0.        ],
                                       [  1.9991104 ,  0.        ],
                                       [  3.00279193,  0.        ],
                                       [  4.00120469,  0.        ],
                                       [  5.0028731 ,  0.        ]],

                                      [[  0.        ,  0.        ],
                                       [  0.99922729,  0.        ],
                                       [  2.00206989,  0.        ],
                                       [  3.00099994,  0.        ],
                                       [  3.99822207,  0.        ],
                                       [  4.99792806,  0.        ]]],


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
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        # Generate staggered signal
        light_520.set_staggered_signal(
            signal=self.signal,
            signal_init=self.signal_init,
            sampling_time_steps=numpy.array([69,  6, 21, 63, 30, 36,
                                             54, 45, 18, 12, 42, 66,
                                             60, 51,  0,  3, 39, 33,
                                             24, 27, 15, 48, 57,  9]),
            n_time_steps=self.n_time_steps)
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
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Check red light intensity
        intensity_red_exp = numpy.array(
                  [[ 19.99777535,  20.00175826,  19.99899181,
                     20.00004389,  19.99917667,  19.99996583],
                   [ 20.00050003,  19.99959588,  20.00246677,
                     19.99751222,  19.99822807,  19.99778845],
                   [ 20.00142497,  20.00227629,  20.00066769,
                     20.00196242,  19.99826936,  19.99730572],
                   [ 20.00120669,  19.99863112,  19.9976445 ,
                     20.00258099,  19.99775917,  19.99978527]])
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

        # Add staggered signal
        light_660 = lpadesign.inducer.LightSignal(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.set_staggered_signal(
            signal=self.signal,
            signal_init=self.signal_init,
            sampling_time_steps=numpy.array([ 72,  20,  30,  10]),
            n_time_steps=self.n_time_steps)
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
        dc_exp[:,:,0] = 4
        dc_exp[:,:,1] = 7
        dc_exp[3,5,1] = 8
        numpy.testing.assert_almost_equal(lpa.dc, dc_exp)
        # Grayscale calibration
        gcal_exp = numpy.array([[[255, 182],
                                 [255, 182],
                                 [255, 192],
                                 [255, 153],
                                 [255, 178],
                                 [255, 171]],

                                [[255, 197],
                                 [255, 174],
                                 [255, 189],
                                 [255, 177],
                                 [255, 196],
                                 [255, 186]],

                                [[255, 175],
                                 [255, 175],
                                 [255, 187],
                                 [255, 182],
                                 [255, 183],
                                 [255, 231]],

                                [[255, 182],
                                 [255, 199],
                                 [255, 215],
                                 [255, 231],
                                 [255, 177],
                                 [255, 222]]], dtype=int)
        numpy.testing.assert_almost_equal(lpa.gcal, gcal_exp)
        # Check green light intensity
        intensity_green_exp = numpy.array(
            [[ 0.        ,  0.99775886,  2.00071772,
               2.99970161,  3.99813213,  4.99752286],
             [ 0.        ,  0.99889055,  2.00026252,
               3.00089391,  3.99957631,  4.99745787],
             [ 0.        ,  1.00040402,  1.9991104 ,
               3.00279193,  4.00120469,  5.0028731 ],
             [ 0.        ,  0.99922729,  2.00206989,
               3.00099994,  3.99822207,  4.99792806]])
        intensity_green_exp.shape=(1,4,6)
        intensity_green_exp = numpy.repeat(intensity_green_exp, 101, 0)
        intensity_green_exp[-1,:,:] = 0.
        numpy.testing.assert_almost_equal(lpa.intensity[:,:,:,0], intensity_green_exp)
        # # Check red light intensity
        intensity_init_red_exp = numpy.array(
            [[ 5.00225094,  5.00043956,  4.99834234,  4.99721923,  5.00120694,
               5.00286667],
             [ 5.00012501,  5.00132792,  4.99778027,  4.997963  ,  4.99955702,
               5.00229905],
             [ 5.0017644 ,  4.99916953,  4.9987272 ,  4.99904746,  5.00096426,
               5.00072955],
             [ 5.00172911,  5.00106257,  5.00222136,  4.99781843,  4.99943979,
               5.00269505]])
        numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,1],
                                          intensity_init_red_exp)
        # Check specific wells of green light over time
        numpy.testing.assert_almost_equal(lpa.intensity[:,0,0,1], numpy.array([
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
             5.00225094,   5.00225094,   5.00225094,   5.00225094,
            11.99754237,  12.87335735,  13.73794393,  14.58568792,
            15.42220351,  16.2250339 ,  16.99979331,  17.73525332,
            18.42579975,  19.07143258,  19.66092344,  20.1942723 ,
            20.66025078,  21.06447308,  21.3957108 ,  21.65957813,
            21.85046088,  21.96274485,  22.00204424,  21.96274485,
            21.85046088,  21.65957813,  21.3957108 ,  21.06447308,
            20.66025078,  20.1942723 ,  19.66092344,  19.07143258,
            18.42579975,  17.73525332,  16.99979331,  16.2250339 ,
            15.42220351,  14.58568792,  13.73794393,  12.87335735,
            11.99754237,  11.12734159,  10.26275501,   9.40939682,
             8.57849543,   7.77566504,   7.00090563,   6.26544562,
             5.57489919,   4.92926636,   4.3397755 ,   3.80642664,
             3.34044816,   2.93622586,   2.60498814,   2.34112081,
             2.15023806,   2.03795409,   1.9986547 ,   2.03795409,
             2.15023806,   2.34112081,   2.60498814,   2.93622586,
             3.34044816,   3.80642664,   4.3397755 ,   4.92926636,
             5.57489919,   6.26544562,   7.00090563,   7.77566504,
             8.57849543,   9.40939682,  10.26275501,  11.12734159,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,1,1,1], numpy.array([
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
             5.00132792,   5.00132792,   5.00132792,   5.00132792,
            11.99747121,  12.87198912,  13.73507542,  14.58673012,
            15.42123741,  16.22716568,  16.99879913,  17.73613776,
            18.42774996,  19.07363574,  19.66236348,  20.19393319,
            20.66262907,  21.0627353 ,  21.39425189,  21.65717885,
            21.84580036,  21.96011642,  22.00012705,  21.96011642,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,2,3,1], numpy.array([
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,   4.99904746,   4.99904746,
             4.99904746,   4.99904746,  12.00117745,  12.87283584,
            13.73872166,  14.58728976,  15.41854015,  16.22670025,
            17.00022491,  17.73334157,  18.42605023,  19.07257831,
            19.66138066,  20.1924573 ,  20.66003564,  21.06411569,
            21.39892487,  21.65869062,  21.8491855 ,  21.96463694,
            21.99927237,  21.96463694,  21.8491855 ,  21.65869062,
            21.39892487,  21.06411569,  20.66003564,  20.1924573 ,
            19.66138066,  19.07257831,  18.42605023,  17.73334157,
             0.        ]))
        numpy.testing.assert_almost_equal(lpa.intensity[:,3,5,1], numpy.array([
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,   5.00269505,   5.00269505,
             5.00269505,   5.00269505,  12.00097065,  12.86957045,
            13.73817025,  14.59027765,  15.42039518,  16.22852284,
            16.99816824,  17.73482883,  18.42750968,  19.07071333,
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
        # Default dot correction and grayscale calibrations
        self.default_dc = {}
        self.default_dc['Tori'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]]])
        self.default_dc['Tiffani'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]]])
        self.default_dc['Shannen'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 8],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]]])
        self.default_dc['Jennie'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 8]]])
        self.default_dc['Kirk'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 8],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]]])
        self.default_dc['Picard'] = numpy.array(
          [[[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]],

           [[4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7],
            [4, 7]]])
        self.default_gcal = {}
        self.default_gcal['Tori'] = numpy.array(
          [[[255, 211],
            [255, 198],
            [255, 169],
            [255, 185],
            [255, 172],
            [255, 187]],

           [[255, 165],
            [255, 170],
            [255, 198],
            [255, 194],
            [255, 172],
            [255, 197]],

           [[255, 172],
            [255, 174],
            [255, 249],
            [255, 187],
            [255, 168],
            [255, 179]],

           [[255, 178],
            [255, 160],
            [255, 177],
            [255, 182],
            [255, 166],
            [255, 185]]])
        self.default_gcal['Tiffani'] = numpy.array(
          [[[255, 190],
            [255, 167],
            [255, 171],
            [255, 205],
            [255, 186],
            [255, 186]],

           [[255, 200],
            [255, 234],
            [255, 217],
            [255, 214],
            [255, 174],
            [255, 237]],

           [[255, 177],
            [255, 178],
            [255, 194],
            [255, 181],
            [255, 208],
            [255, 195]],

           [[255, 168],
            [255, 164],
            [255, 195],
            [255, 196],
            [255, 182],
            [255, 168]]])
        self.default_gcal['Shannen'] = numpy.array(
          [[[255, 183],
            [255, 171],
            [255, 210],
            [255, 230],
            [255, 198],
            [255, 190]],

           [[255, 181],
            [255, 172],
            [255, 171],
            [255, 178],
            [255, 169],
            [255, 206]],

           [[255, 202],
            [255, 185],
            [255, 179],
            [255, 199],
            [255, 196],
            [255, 182]],

           [[255, 209],
            [255, 215],
            [255, 218],
            [255, 184],
            [255, 171],
            [255, 175]]])
        self.default_gcal['Jennie'] = numpy.array(
          [[[255, 182],
            [255, 182],
            [255, 192],
            [255, 153],
            [255, 178],
            [255, 171]],

           [[255, 197],
            [255, 174],
            [255, 189],
            [255, 177],
            [255, 196],
            [255, 186]],

           [[255, 175],
            [255, 175],
            [255, 187],
            [255, 182],
            [255, 183],
            [255, 231]],

           [[255, 182],
            [255, 199],
            [255, 215],
            [255, 231],
            [255, 177],
            [255, 222]]])
        self.default_gcal['Kirk'] = numpy.array(
          [[[255, 177],
            [255, 174],
            [255, 199],
            [255, 163],
            [255, 177],
            [255, 226]],

           [[255, 181],
            [255, 163],
            [255, 206],
            [255, 169],
            [255, 241],
            [255, 191]],

           [[255, 220],
            [255, 194],
            [255, 174],
            [255, 174],
            [255, 185],
            [255, 172]],

           [[255, 187],
            [255, 168],
            [255, 174],
            [255, 183],
            [255, 185],
            [255, 177]]])
        self.default_gcal['Picard'] = numpy.array(
          [[[255, 204],
            [255, 170],
            [255, 164],
            [255, 184],
            [255, 239],
            [255, 185]],

           [[255, 191],
            [255, 173],
            [255, 153],
            [255, 176],
            [255, 198],
            [255, 173]],

           [[255, 172],
            [255, 177],
            [255, 190],
            [255, 178],
            [255, 209],
            [255, 166]],

           [[255, 165],
            [255, 194],
            [255, 179],
            [255, 211],
            [255, 177],
            [255, 192]]])
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
                                              numpy.zeros((4, 6, 2)))
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              numpy.ones((4, 6, 2))*255)
            numpy.testing.assert_almost_equal(lpa.intensity,
                                              numpy.zeros((1, 4, 6, 2)))
        self.assertEqual(p.lpa_optimize_dc, [False, False])
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
                                              numpy.zeros((8, 12, 4)))
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              numpy.ones((8, 12, 4))*255)
            numpy.testing.assert_almost_equal(lpa.intensity,
                                              numpy.zeros((1, 8, 12, 4)))
        self.assertEqual(p.lpa_optimize_dc, [False, False, False, False])
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = (numpy.arange(8) + 1)*2
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
        light_660.intensities = numpy.array([ 3,  6,  7,  4,
                                             20, 10,  8,  5])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  3.00122676,   2.99993841],
                [  8.99991737,   2.99794158],
                [ 11.99681118,   3.00081555],
                [ 18.0011686 ,   3.00212692],
                [ 15.99761037,   2.99902717],
                [  6.99685346,   2.99994521]],

               [[  2.99810623,   6.00176326],
                [  8.99739346,   5.99778782],
                [ 12.00244878,   6.00174279],
                [ 18.00024996,   6.00254972],
                [ 15.99731744,   6.00168822],
                [  6.99767809,   6.00214017]],

               [[  3.00022384,   6.9991199 ],
                [  8.99975124,   6.99973227],
                [ 11.99766066,   6.9989673 ],
                [ 17.99910341,   7.00196682],
                [ 15.99995828,   7.00142775],
                [  6.99953988,   6.99984825]],

               [[  2.99812879,   3.99996345],
                [  8.99960675,   3.9994464 ],
                [ 12.00266152,   3.99963854],
                [ 17.99874603,   3.99825033],
                [ 16.00284225,   4.00033686],
                [  6.99674702,   3.99763015]]])
        intensity['Tiffani'] = numpy.array(
              [[[  8.00231152,   2.9972239 ],
                [ 14.00078232,   3.00081107],
                [ 10.9973392 ,   2.99890526],
                [ 18.99791588,   3.00031262],
                [  9.99951414,   2.9972437 ],
                [  5.99951839,   3.00134543]],

               [[  7.99935854,   5.99751561],
                [ 13.99764136,   6.00059965],
                [ 11.00035043,   5.99823131],
                [ 18.99789002,   5.99804147],
                [  9.9985177 ,   6.00095257],
                [  6.00191927,   6.00151011]],

               [[  7.99777081,   6.99926497],
                [ 14.00325801,   6.99952912],
                [ 11.00174834,   7.00204774],
                [ 18.9993485 ,   7.00231907],
                [  9.99677463,   6.99865504],
                [  6.00329841,   7.00171874]],

               [[  8.00233506,   3.99892678],
                [ 14.00065974,   3.99722616],
                [ 11.00071396,   4.00159234],
                [ 19.00237926,   4.00061735],
                [ 10.00015931,   4.00102918],
                [  5.99746317,   3.99771201]]])
        intensity['Shannen'] = numpy.array(
              [[[  3.99959473,   3.00084998],
                [ 14.9995695 ,   2.99832778],
                [ 16.99977485,   3.00029057],
                [  4.99731087,   3.00159828],
                [ 20.00119698,   2.99890179],
                [ 13.00222986,   2.99853704]],

               [[  3.99877874,   6.00153483],
                [ 15.00172159,   5.99839257],
                [ 17.00156084,   5.99817341],
                [  5.00193377,   6.00262744],
                [ 19.99879302,   5.99794133],
                [ 12.99915308,   6.00088144]],

               [[  3.99744805,   7.0001409 ],
                [ 15.00158322,   6.99748279],
                [ 17.00193243,   6.99956455],
                [  4.99717009,   6.99980974],
                [ 19.99848449,   6.99960287],
                [ 12.99902022,   7.002053  ]],

               [[  4.00183995,   3.99812764],
                [ 14.9991082 ,   4.00061139],
                [ 16.99891201,   3.9972682 ],
                [  4.99909126,   3.99836339],
                [ 19.99837409,   4.00041014],
                [ 13.00028124,   4.00245218]]])
        intensity['Jennie'] = numpy.array(
              [[[  2.99912583,  19.99777535],
                [  8.99764682,  20.00175826],
                [ 11.99776804,  19.99899181],
                [ 17.99820965,  20.00004389],
                [ 15.99910438,  19.99917667],
                [  7.00281819,  19.99996583]],

               [[  2.99763105,  10.00025001],
                [  9.00162992,  10.00265584],
                [ 12.0015751 ,  10.00123339],
                [ 17.99866504,  10.00158622],
                [ 15.99830524,   9.99911403],
                [  7.00178589,   9.99889423]],

               [[  3.0015932 ,   7.99831694],
                [  8.99687673,   7.99979088],
                [ 12.00139344,   7.99911529],
                [ 17.99855283,   8.00078497],
                [ 15.99803708,   8.00154281],
                [  6.99788385,   7.99779979]],

               [[  3.00278168,   5.00172911],
                [  8.9995765 ,   5.00106257],
                [ 11.99879979,   5.00222136],
                [ 17.99937493,   4.99781843],
                [ 15.99918471,   4.99943979],
                [  6.99841975,   5.00269505]]])
        intensity['Kirk'] = numpy.array(
              [[[  8.00222689,  20.00259377],
                [ 14.00167924,  19.99913528],
                [ 11.00023583,  19.99893815],
                [ 18.99875767,  20.00172606],
                [  9.99770114,  20.00010147],
                [  6.00045698,  20.00118137]],

               [[  7.99990768,   9.99837761],
                [ 14.00212595,  10.00179512],
                [ 10.99941874,  10.00171563],
                [ 18.99930418,  10.00088069],
                [ 10.0012366 ,   9.99752688],
                [  6.00094267,   9.99755712]],

               [[  7.99961532,   8.00272052],
                [ 13.99675855,   7.99759002],
                [ 10.99987913,   7.99787232],
                [ 18.99917603,   7.99743728],
                [ 10.00077766,   7.99979266],
                [  5.99721396,   8.00108235]],

               [[  8.00011836,   4.99749055],
                [ 14.00053015,   4.99750903],
                [ 11.00067501,   5.00233098],
                [ 19.00284346,   5.00101186],
                [  9.99685973,   5.00073175],
                [  6.00315949,   4.99855046]]])
        intensity['Picard'] = numpy.array(
              [[[  4.00235755,  19.99779507],
                [ 14.99934731,  20.00051679],
                [ 17.0005672 ,  20.00111186],
                [  4.99915397,  19.99834075],
                [ 19.9978661 ,  19.99880597],
                [ 13.00066794,  19.9982064 ]],

               [[  3.998447  ,   9.9973452 ],
                [ 14.99719296,  10.00232516],
                [ 17.00297499,  10.00110511],
                [  5.00220569,   9.99941175],
                [ 19.99677574,   9.99865072],
                [ 13.00000817,   9.99759386]],

               [[  3.99913725,   7.99928607],
                [ 14.99733492,   8.00083728],
                [ 17.00088095,   7.99867843],
                [  4.99864414,   8.00173826],
                [ 20.00237459,   8.00148802],
                [ 12.99922181,   8.00103773]],

               [[  4.00209281,   4.99827065],
                [ 15.00272752,   4.99943195],
                [ 17.00258931,   5.00148601],
                [  5.00226785,   4.99792436],
                [ 20.00037086,   4.99776795],
                [ 12.99994126,   5.00113872]]])
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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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
        p.lpa_optimize_dc = [True, True]
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
        light_660.intensities = numpy.array([ 3,  6,  7,  4,
                                             20, 10,  8,  5])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.array(
              [[[1, 1],
                [2, 1],
                [2, 1],
                [3, 1],
                [3, 1],
                [1, 1]],

               [[1, 2],
                [2, 2],
                [2, 2],
                [3, 2],
                [3, 2],
                [2, 2]],

               [[1, 3],
                [2, 3],
                [2, 3],
                [3, 3],
                [3, 3],
                [2, 3]],

               [[1, 2],
                [2, 2],
                [2, 2],
                [3, 2],
                [3, 2],
                [1, 2]]])
        dc_exp['Tiffani'] = numpy.array(
              [[[2, 1],
                [3, 1],
                [2, 1],
                [3, 1],
                [2, 1],
                [1, 1]],

               [[2, 2],
                [2, 2],
                [2, 2],
                [3, 2],
                [2, 2],
                [1, 2]],

               [[2, 3],
                [3, 3],
                [2, 3],
                [3, 3],
                [2, 3],
                [1, 3]],

               [[2, 2],
                [3, 2],
                [2, 2],
                [3, 2],
                [2, 2],
                [1, 2]]])
        dc_exp['Shannen'] = numpy.array(
              [[[1, 1],
                [3, 1],
                [3, 1],
                [1, 1],
                [4, 1],
                [2, 1]],

               [[1, 2],
                [3, 2],
                [3, 2],
                [1, 2],
                [4, 2],
                [2, 2]],

               [[1, 3],
                [3, 3],
                [3, 3],
                [1, 3],
                [3, 3],
                [2, 3]],

               [[1, 2],
                [3, 2],
                [3, 2],
                [1, 2],
                [4, 2],
                [2, 2]]])
        dc_exp['Jennie'] = numpy.array(
              [[[1, 7],
                [2, 7],
                [2, 7],
                [3, 7],
                [3, 7],
                [2, 6]],

               [[1, 4],
                [2, 3],
                [2, 4],
                [3, 4],
                [3, 4],
                [2, 3]],

               [[1, 3],
                [2, 3],
                [2, 3],
                [3, 3],
                [3, 3],
                [2, 3]],

               [[1, 2],
                [2, 2],
                [2, 2],
                [3, 2],
                [3, 2],
                [2, 2]]])
        dc_exp['Kirk'] = numpy.array(
              [[[2, 7],
                [3, 7],
                [2, 7],
                [3, 7],
                [2, 7],
                [2, 7]],

               [[2, 4],
                [3, 4],
                [2, 4],
                [3, 4],
                [2, 4],
                [1, 4]],

               [[2, 3],
                [2, 3],
                [2, 3],
                [3, 3],
                [2, 3],
                [1, 3]],

               [[2, 2],
                [3, 2],
                [2, 2],
                [3, 2],
                [2, 2],
                [1, 2]]])
        dc_exp['Picard'] = numpy.array(
              [[[1, 7],
                [3, 7],
                [3, 7],
                [1, 7],
                [3, 7],
                [2, 7]],

               [[1, 4],
                [3, 4],
                [3, 4],
                [1, 4],
                [3, 4],
                [2, 4]],

               [[1, 3],
                [3, 3],
                [3, 3],
                [1, 3],
                [3, 3],
                [2, 3]],

               [[1, 2],
                [3, 2],
                [3, 2],
                [1, 2],
                [3, 2],
                [2, 2]]])
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  2.99975557,   2.99993841],
                [  8.99991737,   3.0003034 ],
                [ 12.00031902,   3.00001873],
                [ 17.99944501,   2.99976614],
                [ 15.99761037,   2.99982499],
                [  7.00035189,   2.99994521]],

               [[  2.99982533,   6.00016364],
                [  9.00047265,   6.0001613 ],
                [ 11.99902147,   6.00015797],
                [ 18.00024996,   6.00018744],
                [ 16.0007696 ,   5.99930439],
                [  7.00105535,   5.99974283]],

               [[  3.00022384,   7.00072483],
                [  8.99975124,   6.99973227],
                [ 12.00117286,   7.00053816],
                [ 17.99910341,   6.99888191],
                [ 15.99995828,   7.00061524],
                [  6.99953988,   6.99984825]],

               [[  2.9998479 ,   3.99996345],
                [  8.99960675,   3.9994464 ],
                [ 11.99916628,   3.99963854],
                [ 18.00039669,   3.99985476],
                [ 15.9993679 ,   4.00033686],
                [  7.00016674,   3.99922984]]])
        intensity['Tiffani'] = numpy.array(
              [[[  7.99895766,   2.9996894 ],
                [ 14.0024914 ,   3.00000828],
                [ 11.00066165,   2.99971055],
                [ 18.99962248,   3.00031262],
                [  9.99951414,   2.99963131],
                [  5.99951839,   2.99973353]],

               [[  7.99935854,   5.99990221],
                [ 14.00106377,   6.00059965],
                [ 11.00035043,   5.99981229],
                [ 19.00124298,   5.9996312 ],
                [  9.9985177 ,   6.00015063],
                [  6.0002722 ,   6.00070959]],

               [[  8.00111996,   7.00085463],
                [ 13.99815478,   7.00112646],
                [ 10.99871755,   6.99888975],
                [ 18.99771176,   7.00074498],
                [ 10.00015649,   6.99944728],
                [  5.99989903,   6.9993258 ]],

               [[  7.99900906,   4.00051365],
                [ 14.00234413,   4.00043421],
                [ 11.00071396,   3.99921375],
                [ 18.99904434,   3.99981803],
                [ 10.00015931,   4.00023201],
                [  6.0006131 ,   4.0000883 ]]])
        intensity['Shannen'] = numpy.array(
              [[[  3.99959473,   3.00005018],
                [ 14.99796355,   2.99994413],
                [ 16.99977485,   3.00029057],
                [  5.00061597,   3.00000718],
                [ 20.00119698,   2.99970256],
                [ 12.99882969,   3.00015963]],

               [[  4.00048471,   5.99994712],
                [ 14.9984303 ,   5.99997944],
                [ 17.00156084,   6.00052533],
                [  5.00025075,   6.00022093],
                [ 19.99879302,   5.9994852 ],
                [ 12.99915308,   5.99930268]],

               [[  4.00028311,   7.00093709],
                [ 14.99988774,   7.00069707],
                [ 17.00025239,   7.0011619 ],
                [  5.00046637,   6.99901295],
                [ 20.00015159,   7.00040283],
                [ 12.99902022,   6.99886481]],

               [[  4.00016132,   3.99972306],
                [ 15.00221618,   4.00061139],
                [ 17.00218607,   3.99965748],
                [  5.0007444 ,   3.99996338],
                [ 19.99837409,   4.00041014],
                [ 13.00028124,   4.00007966]]])
        intensity['Jennie'] = numpy.array(
              [[[  3.00058171,  19.99777535],
                [  9.00061634,  20.00175826],
                [ 12.00103719,  19.99899181],
                [ 17.99820965,  20.00004389],
                [ 15.99910438,  19.99917667],
                [  6.9996751 ,  20.00160881]],

               [[  2.99926375,  10.00025001],
                [  8.99872617,   9.99938967],
                [ 12.0015751 ,  10.00042298],
                [ 18.00033965,  10.00077761],
                [ 16.00171785,  10.00068895],
                [  6.99844534,  10.00052391]],

               [[  2.99995835,   7.99992626],
                [  9.00025647,   7.99899114],
                [ 12.00139344,   7.99911529],
                [ 17.99855283,   8.00078497],
                [ 16.00142793,   8.00074457],
                [  7.0009531 ,   8.00020514]],

               [[  2.99994887,   5.00009775],
                [  8.9995765 ,   4.9994571 ],
                [ 11.99879979,   5.00061551],
                [ 18.00103111,   4.99943375],
                [ 15.99918471,   4.99943979],
                [  6.99841975,   4.99994632]]])
        intensity['Kirk'] = numpy.array(
              [[[  7.99874766,  20.00259377],
                [ 14.00167924,  19.99913528],
                [ 11.00023583,  19.99893815],
                [ 19.00231816,  20.00172606],
                [ 10.00115816,  20.00010147],
                [  6.00045698,  20.00118137]],

               [[  7.99990768,   9.99917246],
                [ 14.00049248,   9.99942034],
                [ 10.99941874,  10.00014722],
                [ 18.99755728,   9.99929148],
                [ 10.0012366 ,  10.00138068],
                [  5.99919414,  10.00073095]],

               [[  7.99961532,   7.99997422],
                [ 14.00021454,   7.99994572],
                [ 10.99987913,   8.0002626 ],
                [ 18.99741554,   8.00057384],
                [ 10.00077766,   7.99900287],
                [  6.00059077,   8.00108235]],

               [[  8.00011836,   5.00064254],
                [ 14.00053015,   4.99988616],
                [ 11.00067501,   4.99995683],
                [ 19.00108849,   5.00021183],
                [ 10.00020092,   5.00073175],
                [  5.99988265,   5.00011986]]])
        intensity['Picard'] = numpy.array(
              [[[  4.00069545,  19.99779507],
                [ 14.99798174,  20.00051679],
                [ 17.00225243,  20.00111186],
                [  4.99915397,  19.99834075],
                [ 19.99952816,  19.99880597],
                [ 13.00066794,  19.9982064 ]],

               [[  4.0001616 ,  10.00053313],
                [ 14.99889641,  10.0015344 ],
                [ 17.00139244,   9.99952641],
                [  5.00051803,  10.00101499],
                [ 20.00196701,   9.99865072],
                [ 13.00000817,  10.00158555]],

               [[  4.00081756,   8.00087876],
                [ 15.00224886,   8.00083728],
                [ 16.99932637,   8.00110448],
                [  5.00021604,   8.00013951],
                [ 20.00237459,   7.99989932],
                [ 12.99922181,   7.99943912]],

               [[  4.00036777,   4.99987163],
                [ 14.99928653,   4.99943195],
                [ 16.99741191,   5.00068947],
                [  5.00061366,   4.99950749],
                [ 19.99866726,   5.00016113],
                [ 12.99994126,   5.00035275]]])
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
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
            # Intensity: frame 1
            numpy.testing.assert_almost_equal(lpa.intensity[0,:,:,:],
                                             intensity[lpa_name])
            # Intensity: frame 2
            numpy.testing.assert_array_equal(lpa.intensity[-1,:,:,:],
                                             numpy.zeros((4,6,2)))

    def test_save_rep_setup_files_rows_and_cols_uniform_dc_optimization(self):
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
        p.lpa_optimize_dc = [True, True]
        p.lpa_optimize_dc_uniform = [True, True]
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
        light_660.intensities = numpy.array([ 3,  6,  7,  4,
                                             20, 10,  8,  5])
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected DC values
        dc_exp = {}
        dc_exp['Tori'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       3*numpy.ones((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 3*numpy.ones((4,6)),
                                          3*numpy.ones((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          3*numpy.ones((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 3*numpy.ones((4,6)),
                                       7*numpy.ones((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 3*numpy.ones((4,6)),
                                         7*numpy.ones((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  3.00122676,   2.99993841],
                [  8.99840934,   2.99951613],
                [ 12.00207294,   3.00001873],
                [ 17.99944501,   3.00055306],
                [ 15.99761037,   2.99902717],
                [  7.00035189,   2.99915741]],

               [[  3.00154443,   6.00096345],
                [  9.00201225,   6.0001613 ],
                [ 11.99902147,   6.00015797],
                [ 18.00024996,   6.00018744],
                [ 16.0007696 ,   6.000099  ],
                [  7.00105535,   6.00054194]],

               [[  3.00022384,   7.00072483],
                [  8.99975124,   6.99973227],
                [ 12.00117286,   7.00053816],
                [ 17.99910341,   6.99888191],
                [ 15.99995828,   7.00061524],
                [  7.00106816,   6.99984825]],

               [[  3.00156701,   4.00075489],
                [  8.99793022,   4.00023774],
                [ 12.0009139 ,   3.99884275],
                [ 18.00039669,   3.99985476],
                [ 15.9993679 ,   4.00112838],
                [  7.0018766 ,   4.00002968]]])
        intensity['Tiffani'] = numpy.array(
              [[[  7.99895766,   3.00051123],
                [ 14.0024914 ,   3.00081107],
                [ 10.99900043,   3.00051584],
                [ 18.99962248,   3.00031262],
                [  9.99951414,   2.99883544],
                [  6.00115403,   3.00053948]],

               [[  8.0010562 ,   5.99990221],
                [ 13.99935256,   6.00059965],
                [ 10.99870858,   5.99981229],
                [ 19.00124298,   5.9996312 ],
                [ 10.00189558,   6.00015063],
                [  5.99862513,   5.99910854]],

               [[  7.99777081,   7.00085463],
                [ 13.99815478,   7.00112646],
                [ 11.00174834,   6.99888975],
                [ 18.99771176,   7.00074498],
                [  9.99846556,   6.99944728],
                [  6.00159872,   6.9993258 ]],

               [[  8.00233506,   3.99892678],
                [ 14.00234413,   3.99883019],
                [ 10.99906616,   4.00079948],
                [ 18.99904434,   3.99981803],
                [ 10.00191126,   4.00102918],
                [  6.0006131 ,   3.9992962 ]]])
        intensity['Shannen'] = numpy.array(
              [[[  3.99959473,   2.99925038],
                [ 14.9995695 ,   2.99913596],
                [ 16.99977485,   3.00109022],
                [  4.99731087,   3.00000718],
                [ 20.00119698,   3.00050334],
                [ 13.00222986,   3.00097092]],

               [[  3.99877874,   5.99915327],
                [ 15.00172159,   6.00077288],
                [ 17.00156084,   5.99974136],
                [  5.00193377,   5.99941876],
                [ 19.99879302,   6.00025713],
                [ 12.99915308,   6.00088144]],

               [[  3.99744805,   7.00093709],
                [ 15.00158322,   7.00069707],
                [ 17.00193243,   7.0011619 ],
                [  4.99717009,   6.99901295],
                [ 19.99848449,   7.00040283],
                [ 12.99902022,   6.99886481]],

               [[  4.00183995,   3.99892535],
                [ 14.9991082 ,   3.99991002],
                [ 16.99891201,   3.99965748],
                [  4.99909126,   4.00076337],
                [ 19.99837409,   3.99880032],
                [ 13.00028124,   4.00007966]]])
        intensity['Jennie'] = numpy.array(
              [[[  3.00058171,  19.99777535],
                [  9.0021011 ,  20.00175826],
                [ 11.99940261,  19.99899181],
                [ 17.99820965,  20.00004389],
                [ 15.99910438,  19.99917667],
                [  7.00124665,  19.99996583]],

               [[  2.99763105,  10.00025001],
                [  8.99872617,  10.00265584],
                [ 12.0015751 ,  10.00123339],
                [ 18.00033965,  10.00158622],
                [ 16.00171785,   9.99911403],
                [  7.00011562,   9.99889423]],

               [[  3.0015932 ,   7.99831694],
                [  8.9985666 ,   7.99979088],
                [ 11.99971068,   7.99911529],
                [ 17.99855283,   8.00078497],
                [ 16.00142793,   8.00154281],
                [  6.99788385,   7.99779979]],

               [[  2.99994887,   5.00172911],
                [  8.99794377,   5.00106257],
                [ 12.00220467,   5.00222136],
                [ 18.00103111,   4.99781843],
                [ 15.99918471,   4.99943979],
                [  7.00172089,   4.99788477]]])
        intensity['Kirk'] = numpy.array(
              [[[  8.00048728,  20.00259377],
                [ 14.00167924,  19.99913528],
                [ 11.00192195,  19.99893815],
                [ 19.00231816,  20.00172606],
                [  9.99770114,  20.00010147],
                [  5.99900056,  20.00118137]],

               [[  7.99990768,   9.99837761],
                [ 14.00049248,  10.00179512],
                [ 10.99941874,  10.00171563],
                [ 18.99755728,  10.00088069],
                [  9.99776876,   9.99752688],
                [  6.00094267,   9.99755712]],

               [[  7.99961532,   8.00203395],
                [ 14.00194254,   7.99759002],
                [ 10.99842258,   7.99787232],
                [ 18.99741554,   7.99743728],
                [ 10.00077766,   7.99979266],
                [  6.00227918,   8.00108235]],

               [[  7.99841476,   4.99749055],
                [ 14.00053015,   4.99750903],
                [ 11.00067501,   5.00233098],
                [ 19.00108849,   5.00101186],
                [  9.99853032,   5.00073175],
                [  6.00152107,   4.99855046]]])
        intensity['Picard'] = numpy.array(
              [[[  3.99903334,  19.99779507],
                [ 14.99798174,  20.00051679],
                [ 17.00225243,  20.00111186],
                [  4.99915397,  19.99834075],
                [ 19.99952816,  19.99880597],
                [ 12.99899432,  19.9982064 ]],

               [[  4.0018762 ,   9.9973452 ],
                [ 14.99889641,  10.00232516],
                [ 17.00139244,  10.00110511],
                [  5.00220569,   9.99941175],
                [ 20.00196701,   9.99865072],
                [ 13.00160209,   9.99759386]],

               [[  4.00249787,   7.99928607],
                [ 15.00224886,   8.00083728],
                [ 16.99932637,   7.99867843],
                [  4.99864414,   8.00173826],
                [ 20.00237459,   8.00148802],
                [ 12.99922181,   8.00103773]],

               [[  4.00036777,   4.99827065],
                [ 14.99928653,   4.99943195],
                [ 16.99741191,   5.00148601],
                [  5.00226785,   4.99792436],
                [ 19.99866726,   4.99776795],
                [ 12.99831301,   5.00113872]]])
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
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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
            [ 10. ,  10.9,  10.7,  14.1,  17.9,  13.2,  11.8,   4.9,   1.2,
              15.6,   6.5,   9.1,   9.3,   5.4,  10.3,  12.1,   4.3,   0.3,
               6.5,   7.4,  19. ,  16.7,   6.4,   3.5,   7.9,   9.7,   4.6,
               3.9,   2.6,   1.1,  18.9,   1.9,  15.9,   2.2,   4.4,  14.4,
               6. ,   1.7,  19.4,   8.6,   6.4,   3.9,  17.7,  11.1,  19.6,
               5. ,  18.8,  14.6,  15.4,  11.8,   6.4,   2.3,   9.3,  14.5,
               7.7,  12. ,   7.4,  10.7,  17.9,  17.8,  17.6,  10.8,   1.9,
              18.3,   1.1,  12.2,   0. ,   0.6,  17. ,  19.8,   2.6,   6.6,
               8.4,   5.8,   3.8,   7.8,   5.9,  15.1,  16.7,   9.6,  10.9,
               7.1,  15.5,  11.8,  19.1,  11.2,   0.5,   6.5,   8. ,   5.6,
               9.2,  16.5,   3.5,   0. ,  18. ,  11.8,  17.2,  10.3,   8. ,
              16.1,  14.1,  17.3,   4.8,   3.4,   7.1,  16.8,   8.6,  17.8,
              14.9,   8.3,  18.4,   7. ,  17. ,  13. ,  12.2,   9.3,   5.4,
              14.1,  13.1,   2.5,  15.4,  13.5,   0.2,   5.1,  12.6,   7.4,
               0.8,  10.9,   0. ,  16.7,  16.2,  17.7,  17.1,  19. ,  13.3,
              18.1,   9.5,   3.2,  16.4,   3.3,   6.8,  15.6,  13.3,   8.3])
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

        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  9.99820443,  19.99958943],
                [ 10.90003397,  19.99913601],
                [ 10.69891055,  20.00171851],
                [ 14.098962  ,  20.00132631],
                [ 17.89786084,  19.99909922],
                [ 13.20306248,  20.00147294]],

               [[  6.49818897,  19.99841264],
                [  7.40237299,  19.99816418],
                [ 19.00102114,  20.00211138],
                [ 16.70227276,  19.99747509],
                [  6.40030784,  20.00191923],
                [  3.49883905,  19.99781091]],

               [[  6.00044769,  19.99748541],
                [  1.69906808,  20.00081103],
                [ 19.40136929,  20.00176201],
                [  8.60116065,  20.00176336],
                [  6.40277929,  19.99757919],
                [  3.9001803 ,  19.99796773]],

               [[  7.70161523,  19.99981724],
                [ 11.99724029,  19.99723198],
                [  7.40292286,  19.99819272],
                [ 10.70284935,  20.00248267],
                [ 17.89983571,  20.00168432],
                [ 17.80306205,  19.99934857]]])
        intensity['Tiffani'] = numpy.array(
              [[[ 11.79888178,  20.00258636],
                [  4.90164108,  19.99978761],
                [  1.20273015,  20.00021778],
                [ 15.59836069,  20.00208413],
                [  6.49901488,  20.00019494],
                [  9.10068711,  19.9995668 ]],

               [[  7.89749914,  19.99728742],
                [  9.69909917,  20.00199883],
                [  4.59716137,  19.99779332],
                [  3.90284927,  20.00274499],
                [  2.60096575,  20.00130404],
                [  1.10024206,  19.99943004]],

               [[ 17.69690994,  20.00187407],
                [ 11.09782012,  19.99785596],
                [ 19.59705365,  20.0003242 ],
                [  5.00189602,  19.99875548],
                [ 18.80313497,  19.99774174],
                [ 14.5969215 ,  20.00012481]],

               [[ 17.60114593,  20.00018795],
                [ 10.80031644,  19.99735898],
                [  1.8982658 ,  20.00241166],
                [ 18.30204587,  19.99749148],
                [  1.10022776,  19.99956568],
                [ 12.2028216 ,  19.9996494 ]]])
        intensity['Shannen'] = numpy.array(
              [[[  9.29725489,  19.99820176],
                [  5.40241454,  19.99828058],
                [ 10.30105404,  20.00007126],
                [ 12.1032754 ,  19.99766124],
                [  4.30113131,  20.00015249],
                [  0.29921449,  20.00160505]],

               [[ 18.90207699,  19.99955913],
                [  1.90236838,  20.00019596],
                [ 15.89810857,  19.99756991],
                [  2.20138942,  20.00127122],
                [  4.40179307,  19.9985413 ],
                [ 14.39691148,  19.99741246]],

               [[ 15.40009772,  19.99721778],
                [ 11.80052206,  20.00245081],
                [  6.39756509,  19.9979572 ],
                [  2.30080786,  20.00104999],
                [  9.30239609,  20.00206516],
                [ 14.50172025,  20.00188048]],

               [[  0.        ,  20.00180615],
                [  0.60294799,  19.99744601],
                [ 16.99891201,  19.99749098],
                [ 19.79798842,  19.99741692],
                [  2.59831092,  20.0020507 ],
                [  6.60268737,  20.00118913]]])
        intensity['Jennie'] = numpy.array(
              [[[  8.39755231,  19.99777535],
                [  5.80244287,  20.00175826],
                [  3.79874836,  19.99899181],
                [  7.79922418,  20.00004389],
                [  5.89856006,  19.99917667],
                [ 15.09943384,  19.99996583]],

               [[  9.20187833,  20.00050003],
                [ 16.49911652,  19.99959588],
                [  3.50045941,  20.00246677],
                [  0.        ,  19.99751222],
                [ 17.99809339,  19.99822807],
                [ 11.79881096,  19.99778845]],

               [[ 14.89679589,  20.00142497],
                [  8.3006496 ,  20.00227629],
                [ 18.40258534,  20.00066769],
                [  7.00044825,  20.00196242],
                [ 17.0017291 ,  19.99826936],
                [ 13.00133157,  19.99730572]],

               [[  0.79885324,  20.00120669],
                [ 10.90006762,  19.99863112],
                [  0.        ,  19.9976445 ],
                [ 16.70092904,  20.00258099],
                [ 16.20066991,  19.99775917],
                [ 17.70072013,  19.99978527]]])
        intensity['Kirk'] = numpy.array(
              [[[ 16.7002996 ,  20.00259377],
                [  9.60208399,  19.99913528],
                [ 10.89906873,  19.99893815],
                [  7.09961072,  20.00172606],
                [ 15.50127659,  20.00010147],
                [ 11.80284062,  20.00118137]],

               [[ 17.20203969,  20.00231915],
                [ 10.29741041,  19.99804908],
                [  8.00124942,  19.99794184],
                [ 16.09944716,  20.00176138],
                [ 14.10021776,  20.00044907],
                [ 17.30341744,  20.00066844]],

               [[ 12.19926422,  19.9985624 ],
                [  9.29661247,  20.00221999],
                [  5.40089404,  20.00025812],
                [ 14.09798014,  20.00182666],
                [ 13.10325981,  20.00224593],
                [  2.49883915,  20.00270586]],

               [[ 17.09735687,  20.00099418],
                [ 18.99926806,  20.00112937],
                [ 13.29816112,  19.99824456],
                [ 18.09727759,  19.99844722],
                [  9.50236265,  19.99734582],
                [  3.19818977,  19.99969474]]])
        intensity['Picard'] = numpy.array(
              [[[ 19.10095225,  19.99779507],
                [ 11.19761907,  20.00051679],
                [  0.49882711,  20.00111186],
                [  6.49820293,  19.99834075],
                [  7.99781679,  19.99880597],
                [  5.60327449,  19.9982064 ]],

               [[  4.80087976,  20.00026927],
                [  3.40009054,  19.999115  ],
                [  7.10250854,  20.00221023],
                [ 16.80228064,  19.9988235 ],
                [  8.59674471,  19.99730144],
                [ 17.80089397,  20.00077609]],

               [[ 15.39835873,  20.00100239],
                [ 13.49694624,  20.0020932 ],
                [  0.19898617,  19.99952647],
                [  5.09924579,  19.99875004],
                [ 12.59871788,  20.00093982],
                [  7.39945054,  20.00259433]],

               [[ 16.40168036,  19.99868603],
                [  3.30335285,  19.99772781],
                [  6.79965508,  20.00036825],
                [ 15.60231163,  19.9972384 ],
                [ 13.30177987,  20.00223999],
                [  8.2975577 ,  19.99905307]]])
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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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
            [ 10. ,  10.9,  10.7,  14.1,  17.9,  13.2,  11.8,   4.9,   1.2,
              15.6,   6.5,   9.1,   9.3,   5.4,  10.3,  12.1,   4.3,   0.3,
               6.5,   7.4,  19. ,  16.7,   6.4,   3.5,   7.9,   9.7,   4.6,
               3.9,   2.6,   1.1,  18.9,   1.9,  15.9,   2.2,   4.4,  14.4,
               6. ,   1.7,  19.4,   8.6,   6.4,   3.9,  17.7,  11.1,  19.6,
               5. ,  18.8,  14.6,  15.4,  11.8,   6.4,   2.3,   9.3,  14.5,
               7.7,  12. ,   7.4,  10.7,  17.9,  17.8,  17.6,  10.8,   1.9,
              18.3,   1.1,  12.2,   0. ,   0.6,  17. ,  19.8,   2.6,   6.6,
               8.4,   5.8,   3.8,   7.8,   5.9,  15.1,  16.7,   9.6,  10.9,
               7.1,  15.5,  11.8,  19.1,  11.2,   0.5,   6.5,   8. ,   5.6,
               9.2,  16.5,   3.5,   0. ,  18. ,  11.8,  17.2,  10.3,   8. ,
              16.1,  14.1,  17.3,   4.8,   3.4,   7.1,  16.8,   8.6,  17.8,
              14.9,   8.3])
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

        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  9.99820443,  19.99958943],
                [ 10.90003397,  19.99913601],
                [ 10.69891055,  20.00171851],
                [ 14.098962  ,  20.00132631],
                [ 17.89786084,  19.99909922],
                [ 13.20306248,  20.00147294]],

               [[  6.49818897,  19.99841264],
                [  7.40237299,  19.99816418],
                [ 19.00102114,  20.00211138],
                [ 16.70227276,  19.99747509],
                [  6.40030784,  20.00191923],
                [  3.49883905,  19.99781091]],

               [[  6.00044769,  19.99748541],
                [  1.69906808,  20.00081103],
                [ 19.40136929,  20.00176201],
                [  8.60116065,  20.00176336],
                [  6.40277929,  19.99757919],
                [  3.9001803 ,  19.99796773]],

               [[  7.70161523,  19.99981724],
                [ 11.99724029,  19.99723198],
                [  7.40292286,  19.99819272],
                [ 10.70284935,  20.00248267],
                [ 17.89983571,  20.00168432],
                [ 17.80306205,  19.99934857]]])
        intensity['Tiffani'] = numpy.array(
              [[[ 11.79888178,  20.00258636],
                [  4.90164108,  19.99978761],
                [  1.20273015,  20.00021778],
                [ 15.59836069,  20.00208413],
                [  6.49901488,  20.00019494],
                [  9.10068711,  19.9995668 ]],

               [[  7.89749914,  19.99728742],
                [  9.69909917,  20.00199883],
                [  4.59716137,  19.99779332],
                [  3.90284927,  20.00274499],
                [  2.60096575,  20.00130404],
                [  1.10024206,  19.99943004]],

               [[ 17.69690994,  20.00187407],
                [ 11.09782012,  19.99785596],
                [ 19.59705365,  20.0003242 ],
                [  5.00189602,  19.99875548],
                [ 18.80313497,  19.99774174],
                [ 14.5969215 ,  20.00012481]],

               [[ 17.60114593,  20.00018795],
                [ 10.80031644,  19.99735898],
                [  1.8982658 ,  20.00241166],
                [ 18.30204587,  19.99749148],
                [  1.10022776,  19.99956568],
                [ 12.2028216 ,  19.9996494 ]]])
        intensity['Shannen'] = numpy.array(
              [[[  9.29725489,  19.99820176],
                [  5.40241454,  19.99828058],
                [ 10.30105404,  20.00007126],
                [ 12.1032754 ,  19.99766124],
                [  4.30113131,  20.00015249],
                [  0.29921449,  20.00160505]],

               [[ 18.90207699,  19.99955913],
                [  1.90236838,  20.00019596],
                [ 15.89810857,  19.99756991],
                [  2.20138942,  20.00127122],
                [  4.40179307,  19.9985413 ],
                [ 14.39691148,  19.99741246]],

               [[ 15.40009772,  19.99721778],
                [ 11.80052206,  20.00245081],
                [  6.39756509,  19.9979572 ],
                [  2.30080786,  20.00104999],
                [  9.30239609,  20.00206516],
                [ 14.50172025,  20.00188048]],

               [[  0.        ,  20.00180615],
                [  0.60294799,  19.99744601],
                [ 16.99891201,  19.99749098],
                [ 19.79798842,  19.99741692],
                [  2.59831092,  20.0020507 ],
                [  6.60268737,  20.00118913]]])
        intensity['Jennie'] = numpy.array(
              [[[  8.39755231,  19.99777535],
                [  5.80244287,  20.00175826],
                [  3.79874836,  19.99899181],
                [  7.79922418,  20.00004389],
                [  5.89856006,  19.99917667],
                [ 15.09943384,  19.99996583]],

               [[  9.20187833,  20.00050003],
                [ 16.49911652,  19.99959588],
                [  3.50045941,  20.00246677],
                [  0.        ,  19.99751222],
                [ 17.99809339,  19.99822807],
                [ 11.79881096,  19.99778845]],

               [[ 14.89679589,  20.00142497],
                [  8.3006496 ,  20.00227629],
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
              [[[ 16.7002996 ,  20.00259377],
                [  9.60208399,  19.99913528],
                [ 10.89906873,  19.99893815],
                [  7.09961072,  20.00172606],
                [ 15.50127659,  20.00010147],
                [ 11.80284062,  20.00118137]],

               [[ 17.20203969,  20.00231915],
                [ 10.29741041,  19.99804908],
                [  8.00124942,  19.99794184],
                [ 16.09944716,  20.00176138],
                [ 14.10021776,  20.00044907],
                [ 17.30341744,  20.00066844]],

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
              [[[ 19.10095225,  19.99779507],
                [ 11.19761907,  20.00051679],
                [  0.49882711,  20.00111186],
                [  6.49820293,  19.99834075],
                [  7.99781679,  19.99880597],
                [  5.60327449,  19.9982064 ]],

               [[  4.80087976,  20.00026927],
                [  3.40009054,  19.999115  ],
                [  7.10250854,  20.00221023],
                [ 16.80228064,  19.9988235 ],
                [  8.59674471,  19.99730144],
                [ 17.80089397,  20.00077609]],

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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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

        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  3.00122676,   0.        ],
                [  8.99991737,   0.        ],
                [ 11.99681118,   0.        ],
                [ 18.0011686 ,   0.        ],
                [ 15.99761037,   0.        ],
                [  6.99685346,   0.        ]],

               [[  2.99810623,   0.        ],
                [  8.99739346,   0.        ],
                [ 12.00244878,   0.        ],
                [ 18.00024996,   0.        ],
                [ 15.99731744,   0.        ],
                [  6.99767809,   0.        ]],

               [[  3.00022384,   0.        ],
                [  8.99975124,   0.        ],
                [ 11.99766066,   0.        ],
                [ 17.99910341,   0.        ],
                [ 15.99995828,   0.        ],
                [  6.99953988,   0.        ]],

               [[  2.99812879,   0.        ],
                [  8.99960675,   0.        ],
                [ 12.00266152,   0.        ],
                [ 17.99874603,   0.        ],
                [ 16.00284225,   0.        ],
                [  6.99674702,   0.        ]]])
        intensity['Tiffani'] = numpy.array(
              [[[  8.00231152,   0.        ],
                [ 14.00078232,   0.        ],
                [ 10.9973392 ,   0.        ],
                [ 18.99791588,   0.        ],
                [  9.99951414,   0.        ],
                [  5.99951839,   0.        ]],

               [[  7.99935854,   0.        ],
                [ 13.99764136,   0.        ],
                [ 11.00035043,   0.        ],
                [ 18.99789002,   0.        ],
                [  9.9985177 ,   0.        ],
                [  6.00191927,   0.        ]],

               [[  7.99777081,   0.        ],
                [ 14.00325801,   0.        ],
                [ 11.00174834,   0.        ],
                [ 18.9993485 ,   0.        ],
                [  9.99677463,   0.        ],
                [  6.00329841,   0.        ]],

               [[  8.00233506,   0.        ],
                [ 14.00065974,   0.        ],
                [ 11.00071396,   0.        ],
                [ 19.00237926,   0.        ],
                [ 10.00015931,   0.        ],
                [  5.99746317,   0.        ]]])
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
              [[[  2.99912583,   0.        ],
                [  8.99764682,   0.        ],
                [ 11.99776804,   0.        ],
                [ 17.99820965,   0.        ],
                [ 15.99910438,   0.        ],
                [  7.00281819,   0.        ]],

               [[  2.99763105,   0.        ],
                [  9.00162992,   0.        ],
                [ 12.0015751 ,   0.        ],
                [ 17.99866504,   0.        ],
                [ 15.99830524,   0.        ],
                [  7.00178589,   0.        ]],

               [[  3.0015932 ,   0.        ],
                [  8.99687673,   0.        ],
                [ 12.00139344,   0.        ],
                [ 17.99855283,   0.        ],
                [ 15.99803708,   0.        ],
                [  6.99788385,   0.        ]],

               [[  3.00278168,   0.        ],
                [  8.9995765 ,   0.        ],
                [ 11.99879979,   0.        ],
                [ 17.99937493,   0.        ],
                [ 15.99918471,   0.        ],
                [  6.99841975,   0.        ]]])
        intensity['Kirk'] = numpy.array(
              [[[  8.00222689,   0.        ],
                [ 14.00167924,   0.        ],
                [ 11.00023583,   0.        ],
                [ 18.99875767,   0.        ],
                [  9.99770114,   0.        ],
                [  6.00045698,   0.        ]],

               [[  7.99990768,   0.        ],
                [ 14.00212595,   0.        ],
                [ 10.99941874,   0.        ],
                [ 18.99930418,   0.        ],
                [ 10.0012366 ,   0.        ],
                [  6.00094267,   0.        ]],

               [[  7.99961532,   0.        ],
                [ 13.99675855,   0.        ],
                [ 10.99987913,   0.        ],
                [ 18.99917603,   0.        ],
                [ 10.00077766,   0.        ],
                [  5.99721396,   0.        ]],

               [[  8.00011836,   0.        ],
                [ 14.00053015,   0.        ],
                [ 11.00067501,   0.        ],
                [ 19.00284346,   0.        ],
                [  9.99685973,   0.        ],
                [  6.00315949,   0.        ]]])
        intensity['Picard'] = numpy.array(
              [[[  4.00235755,   0.        ],
                [ 14.99934731,   0.        ],
                [ 17.0005672 ,   0.        ],
                [  4.99915397,   0.        ],
                [ 19.9978661 ,   0.        ],
                [ 13.00066794,   0.        ]],

               [[  3.998447  ,   0.        ],
                [ 14.99719296,   0.        ],
                [ 17.00297499,   0.        ],
                [  5.00220569,   0.        ],
                [ 19.99677574,   0.        ],
                [ 13.00000817,   0.        ]],

               [[  3.99913725,   0.        ],
                [ 14.99733492,   0.        ],
                [ 17.00088095,   0.        ],
                [  4.99864414,   0.        ],
                [ 20.00237459,   0.        ],
                [ 12.99922181,   0.        ]],

               [[  4.00209281,   0.        ],
                [ 15.00272752,   0.        ],
                [ 17.00258931,   0.        ],
                [  5.00226785,   0.        ],
                [ 20.00037086,   0.        ],
                [ 12.99994126,   0.        ]]])
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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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
        dc_exp['Tori'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       numpy.zeros((4,6))], axis=2)
        dc_exp['Tiffani'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          numpy.zeros((4,6))], axis=2)
        dc_exp['Shannen'] = numpy.stack([ 4*numpy.ones((4,6)),
                                          numpy.zeros((4,6))], axis=2)
        dc_exp['Jennie'] = numpy.stack([ 4*numpy.ones((4,6)),
                                         numpy.zeros((4,6))], axis=2)
        dc_exp['Kirk'] = numpy.stack([ 4*numpy.ones((4,6)),
                                       numpy.zeros((4,6))], axis=2)
        dc_exp['Picard'] = numpy.stack([ 4*numpy.ones((4,6)),
                                         numpy.zeros((4,6))], axis=2)
        # Expected intensities of frame 1
        intensity = {}
        intensity['Tori'] = numpy.array(
              [[[  3.00122676,   0.        ],
                [  8.99991737,   0.        ],
                [ 11.99681118,   0.        ],
                [ 18.0011686 ,   0.        ],
                [ 15.99761037,   0.        ],
                [  6.99685346,   0.        ]],

               [[  2.99810623,   0.        ],
                [  8.99739346,   0.        ],
                [ 12.00244878,   0.        ],
                [ 18.00024996,   0.        ],
                [ 15.99731744,   0.        ],
                [  6.99767809,   0.        ]],

               [[  3.00022384,   0.        ],
                [  8.99975124,   0.        ],
                [ 11.99766066,   0.        ],
                [ 17.99910341,   0.        ],
                [ 15.99995828,   0.        ],
                [  6.99953988,   0.        ]],

               [[  2.99812879,   0.        ],
                [  8.99960675,   0.        ],
                [ 12.00266152,   0.        ],
                [ 17.99874603,   0.        ],
                [ 16.00284225,   0.        ],
                [  6.99674702,   0.        ]]])
        intensity['Tiffani'] = numpy.array(
              [[[  8.00231152,   0.        ],
                [ 14.00078232,   0.        ],
                [ 10.9973392 ,   0.        ],
                [ 18.99791588,   0.        ],
                [  9.99951414,   0.        ],
                [  5.99951839,   0.        ]],

               [[  7.99935854,   0.        ],
                [ 13.99764136,   0.        ],
                [ 11.00035043,   0.        ],
                [ 18.99789002,   0.        ],
                [  9.9985177 ,   0.        ],
                [  6.00191927,   0.        ]],

               [[  7.99777081,   0.        ],
                [ 14.00325801,   0.        ],
                [ 11.00174834,   0.        ],
                [ 18.9993485 ,   0.        ],
                [  9.99677463,   0.        ],
                [  6.00329841,   0.        ]],

               [[  8.00233506,   0.        ],
                [ 14.00065974,   0.        ],
                [ 11.00071396,   0.        ],
                [ 19.00237926,   0.        ],
                [ 10.00015931,   0.        ],
                [  5.99746317,   0.        ]]])
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
              [[[  2.99912583,   0.        ],
                [  8.99764682,   0.        ],
                [ 11.99776804,   0.        ],
                [ 17.99820965,   0.        ],
                [ 15.99910438,   0.        ],
                [  7.00281819,   0.        ]],

               [[  2.99763105,   0.        ],
                [  9.00162992,   0.        ],
                [ 12.0015751 ,   0.        ],
                [ 17.99866504,   0.        ],
                [ 15.99830524,   0.        ],
                [  7.00178589,   0.        ]],

               [[  3.0015932 ,   0.        ],
                [  8.99687673,   0.        ],
                [ 12.00139344,   0.        ],
                [ 17.99855283,   0.        ],
                [ 15.99803708,   0.        ],
                [  6.99788385,   0.        ]],

               [[  3.00278168,   0.        ],
                [  8.9995765 ,   0.        ],
                [ 11.99879979,   0.        ],
                [ 17.99937493,   0.        ],
                [ 15.99918471,   0.        ],
                [  6.99841975,   0.        ]]])
        intensity['Kirk'] = numpy.array(
              [[[  8.00222689,   0.        ],
                [ 14.00167924,   0.        ],
                [ 11.00023583,   0.        ],
                [ 18.99875767,   0.        ],
                [  9.99770114,   0.        ],
                [  6.00045698,   0.        ]],

               [[  7.99990768,   0.        ],
                [ 14.00212595,   0.        ],
                [ 10.99941874,   0.        ],
                [ 18.99930418,   0.        ],
                [ 10.0012366 ,   0.        ],
                [  6.00094267,   0.        ]],

               [[  7.99961532,   0.        ],
                [ 13.99675855,   0.        ],
                [ 10.99987913,   0.        ],
                [ 18.99917603,   0.        ],
                [ 10.00077766,   0.        ],
                [  5.99721396,   0.        ]],

               [[  8.00011836,   0.        ],
                [ 14.00053015,   0.        ],
                [ 11.00067501,   0.        ],
                [ 19.00284346,   0.        ],
                [  9.99685973,   0.        ],
                [  6.00315949,   0.        ]]])
        intensity['Picard'] = numpy.array(
              [[[  4.00235755,   0.        ],
                [ 14.99934731,   0.        ],
                [ 17.0005672 ,   0.        ],
                [  4.99915397,   0.        ],
                [ 19.9978661 ,   0.        ],
                [ 13.00066794,   0.        ]],

               [[  3.998447  ,   0.        ],
                [ 14.99719296,   0.        ],
                [ 17.00297499,   0.        ],
                [  5.00220569,   0.        ],
                [ 19.99677574,   0.        ],
                [ 13.00000817,   0.        ]],

               [[  3.99913725,   0.        ],
                [ 14.99733492,   0.        ],
                [ 17.00088095,   0.        ],
                [  4.99864414,   0.        ],
                [ 20.00237459,   0.        ],
                [ 12.99922181,   0.        ]],

               [[  4.00209281,   0.        ],
                [ 15.00272752,   0.        ],
                [ 17.00258931,   0.        ],
                [  5.00226785,   0.        ],
                [ 20.00037086,   0.        ],
                [ 12.99994126,   0.        ]]])
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
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        sampling_time_steps = numpy.array(
            [ 97,  31,   8,  68,  12,  13,  19,  24,  99,  32,  57, 136,  94,  38, 114,  78, 140,  76,
              56,  62,  15,  63,  89,  54,  48,  33,   6, 141,  43, 134,  59,  75, 110,  46,  81,  42,
             112, 125,  22, 129,  10,  70,  93,  25, 105,  58, 111, 107,  40,  83,  44,   3, 138,  69,
              30, 121, 109,  55, 200,  52,  80, 135,  23,  87, 123,  50,  14,  92,  11,   5,  74,  16,
             101,  35, 106,  67,  26, 132, 128,  53,  85, 113,  64, 126,  91,  17,   1,  79, 122, 131,
              45,  27,  61,   9, 118,  41,  65, 137,  34,  37,  51, 124, 119, 133,  18,  60, 102,  88,
               4,  96,   0, 130,  47,  84, 108, 117, 100, 120, 104, 116,  36,   7,  77,  39,  71,  49,
              82,  28,  66, 103, 127, 142,  72,  90,  95,  21,  98,  20, 139,   2, 115,  86,  29,  73])
        light_520.set_staggered_signal(
            signal=self.signal,
            signal_init=self.signal_init,
            sampling_time_steps=sampling_time_steps,
            n_time_steps=self.n_time_steps)
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

        # Expected intensities of frame 1
        intensity_red = {}
        intensity_red['Tori'] = numpy.array(
              [[ 19.99958943,  19.99913601,  20.00171851,  20.00132631,
                 19.99909922,  20.00147294],
               [ 19.99841264,  19.99816418,  20.00211138,  19.99747509,
                 20.00191923,  19.99781091],
               [ 19.99748541,  20.00081103,  20.00176201,  20.00176336,
                 19.99757919,  19.99796773],
               [ 19.99981724,  19.99723198,  19.99819272,  20.00248267,
                 20.00168432,  19.99934857]])
        intensity_red['Tiffani'] = numpy.array(
              [[ 20.00258636,  19.99978761,  20.00021778,  20.00208413,
                 20.00019494,  19.9995668 ],
               [ 19.99728742,  20.00199883,  19.99779332,  20.00274499,
                 20.00130404,  19.99943004],
               [ 20.00187407,  19.99785596,  20.0003242 ,  19.99875548,
                 19.99774174,  20.00012481],
               [ 20.00018795,  19.99735898,  20.00241166,  19.99749148,
                 19.99956568,  19.9996494 ]])
        intensity_red['Shannen'] = numpy.array(
              [[ 19.99820176,  19.99828058,  20.00007126,  19.99766124,
                 20.00015249,  20.00160505],
               [ 19.99955913,  20.00019596,  19.99756991,  20.00127122,
                 19.9985413 ,  19.99741246],
               [ 19.99721778,  20.00245081,  19.9979572 ,  20.00104999,
                 20.00206516,  20.00188048],
               [ 20.00180615,  19.99744601,  19.99749098,  19.99741692,
                 20.0020507 ,  20.00118913]])
        intensity_red['Jennie'] = numpy.array(
              [[ 19.99777535,  20.00175826,  19.99899181,  20.00004389,
                 19.99917667,  19.99996583],
               [ 20.00050003,  19.99959588,  20.00246677,  19.99751222,
                 19.99822807,  19.99778845],
               [ 20.00142497,  20.00227629,  20.00066769,  20.00196242,
                 19.99826936,  19.99730572],
               [ 20.00120669,  19.99863112,  19.9976445 ,  20.00258099,
                 19.99775917,  19.99978527]])
        intensity_red['Kirk'] = numpy.array(
              [[ 20.00259377,  19.99913528,  19.99893815,  20.00172606,
                 20.00010147,  20.00118137],
               [ 20.00231915,  19.99804908,  19.99794184,  20.00176138,
                 20.00044907,  20.00066844],
               [ 19.9985624 ,  20.00221999,  20.00025812,  20.00182666,
                 20.00224593,  20.00270586],
               [ 20.00099418,  20.00112937,  19.99824456,  19.99844722,
                 19.99734582,  19.99969474]])
        intensity_red['Picard'] = numpy.array(
              [[ 19.99779507,  20.00051679,  20.00111186,  19.99834075,
                 19.99880597,  19.9982064 ],
               [ 20.00026927,  19.999115  ,  20.00221023,  19.9988235 ,
                 19.99730144,  20.00077609],
               [ 20.00100239,  20.0020932 ,  19.99952647,  19.99875004,
                 20.00093982,  20.00259433],
               [ 19.99868603,  19.99772781,  20.00036825,  19.9972384 ,
                 20.00223999,  19.99905307]])
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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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

        light_660 = lpadesign.inducer.LightSignal(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.set_staggered_signal(
            signal=self.signal,
            signal_init=self.signal_init,
            sampling_time_steps=numpy.array([ 34, 155, 200, 164,
                                              63,  21, 102, 183]),
            n_time_steps=self.n_time_steps)
        p.apply_inducer(light_660, 'cols')

        # Attempt to generate rep setup files
        p.save_rep_setup_files(path=self.temp_dir)

        # Expected intensities
        intensity_green = {}
        intensity_green['Tori'] = numpy.array(
              [[  3.00122676,   8.99991737,  11.99681118,  18.0011686 ,
                 15.99761037,   6.99685346],
               [  2.99810623,   8.99739346,  12.00244878,  18.00024996,
                 15.99731744,   6.99767809],
               [  3.00022384,   8.99975124,  11.99766066,  17.99910341,
                 15.99995828,   6.99953988],
               [  2.99812879,   8.99960675,  12.00266152,  17.99874603,
                 16.00284225,   6.99674702]])
        intensity_green['Tiffani'] = numpy.array(
              [[  8.00231152,  14.00078232,  10.9973392 ,  18.99791588,
                  9.99951414,   5.99951839],
               [  7.99935854,  13.99764136,  11.00035043,  18.99789002,
                  9.9985177 ,   6.00191927],
               [  7.99777081,  14.00325801,  11.00174834,  18.9993485 ,
                  9.99677463,   6.00329841],
               [  8.00233506,  14.00065974,  11.00071396,  19.00237926,
                 10.00015931,   5.99746317]])
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
              [[  2.99912583,   8.99764682,  11.99776804,  17.99820965,
                 15.99910438,   7.00281819],
               [  2.99763105,   9.00162992,  12.0015751 ,  17.99866504,
                 15.99830524,   7.00178589],
               [  3.0015932 ,   8.99687673,  12.00139344,  17.99855283,
                 15.99803708,   6.99788385],
               [  3.00278168,   8.9995765 ,  11.99879979,  17.99937493,
                 15.99918471,   6.99841975]])
        intensity_green['Kirk'] = numpy.array(
              [[  8.00222689,  14.00167924,  11.00023583,  18.99875767,
                  9.99770114,   6.00045698],
               [  7.99990768,  14.00212595,  10.99941874,  18.99930418,
                 10.0012366 ,   6.00094267],
               [  7.99961532,  13.99675855,  10.99987913,  18.99917603,
                 10.00077766,   5.99721396],
               [  8.00011836,  14.00053015,  11.00067501,  19.00284346,
                  9.99685973,   6.00315949]])
        intensity_green['Picard'] = numpy.array(
              [[  4.00235755,  14.99934731,  17.0005672 ,   4.99915397,
                 19.9978661 ,  13.00066794],
               [  3.998447  ,  14.99719296,  17.00297499,   5.00220569,
                 19.99677574,  13.00000817],
               [  3.99913725,  14.99733492,  17.00088095,   4.99864414,
                 20.00237459,  12.99922181],
               [  4.00209281,  15.00272752,  17.00258931,   5.00226785,
                 20.00037086,  12.99994126]])
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
                 5.00092558,   5.00092558,   5.00092558,  12.00111376,
                12.3167868 ,  12.62692172,  12.94259477,  13.25272969,
                13.5628646 ,  13.87299952,  14.18313444,  14.48773124,
                14.79232804,  15.09138671,  15.38490726,  15.68396593,
                15.97194835,  16.25993078,  16.54237508,  16.81928126,
                17.09064931,  17.35647925,  17.62230918,  17.87706286,
                18.13181654,  18.37549398,  18.61363329,  18.84623448,
                19.07329755,  19.28928437,  19.49973306,  19.70464364,
                19.90401608,  20.09231229,  20.26953224,  20.44121407,
                20.60735778,  20.76242524,  20.91195457,  21.05040766,
                21.1777845 ,  21.29962322,  21.41038569,  21.51007192,
                21.60422002,  21.68729187,  21.75928748,  21.82020684,
                21.87558807,  21.91989306,  21.9531218 ,  21.98081242,
                21.99742679,  21.99742679,  21.99742679,  21.98081242,
                21.9531218 ,  21.91989306,  21.87558807,  21.82020684,
                21.75928748,  21.68729187,  21.60422002,  21.51007192,
                21.41038569,  21.29962322,  21.1777845 ,  21.05040766,
                20.91195457,  20.76242524,  20.60735778,  20.44121407,
                20.26953224,  20.09231229,  19.90401608,  19.70464364,
                19.49973306,  19.28928437,  19.07329755,  18.84623448,
                18.61363329,  18.37549398,  18.13181654,  17.87706286,
                17.62230918,  17.35647925,  17.09064931,  16.81928126,
                16.54237508,  16.25993078,  15.97194835,  15.68396593,
                15.38490726,  15.09138671,  14.79232804,  14.48773124,
                14.18313444,  13.87299952,  13.5628646 ,  13.25272969,
                12.94259477,  12.62692172,  12.3167868 ,  12.00111376,
                11.68544072,  11.36976767,  11.05963275,  10.74395971,
                10.43382479,  10.12368987,   9.81909308,   9.51449628,
                 9.20989948,   8.91084081,   8.61178214,   8.31826159,
                 8.03027917,   7.74229674,   7.45985244,   7.18294626,
                 6.91157821,   6.64021015,   6.37991834,   6.11962654,
                 5.87041098,   5.62673354,   5.38859423,   5.15599304,
                 4.92892997,   4.71294315,   4.49695633,   4.29758388,
                 4.09821144,   3.90991524,   3.72715716,   3.55547533,
                 3.39486974,   3.23426416,   3.09027295,   2.95181986,
                 2.82444302,   2.7026043 ,   2.59184183,   2.4921556 ,
                 2.3980075 ,   2.31493565,   2.24294004,   2.17648256,
                 2.12110132,   2.07679634,   2.04356759,   2.0214151 ,
                 2.00480073,   1.99926261,   2.00480073,   2.0214151 ,
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
                 5.00060292,   5.00060292,  11.99922697,  12.31558032,
                12.62638361,  12.94273696,  13.25354025,  13.56434354,
                13.87514682,  14.18040005,  14.48565328,  14.79090652,
                15.09060969,  15.3847628 ,  15.67891591,  15.97306902,
                16.25612202,  16.53917502,  16.81667795,  17.08863083,
                17.36058371,  17.62143647,  17.87673917,  18.12649181,
                18.37624445,  18.61489698,  18.84799945,  19.0700018 ,
                19.29200415,  19.50290638,  19.70270849,  19.90251061,
                20.0912126 ,  20.26881448,  20.4408663 ,  20.60736806,
                20.76276971,  20.91262129,  21.0458227 ,  21.17902411,
                21.29557535,  21.40657652,  21.51202764,  21.60082858,
                21.68407946,  21.76178028,  21.82283093,  21.87833151,
                21.92273198,  21.95603234,  21.97823257,  21.99488275,
                22.00043281,  21.99488275,  21.97823257,  21.95603234,
                21.92273198,  21.87833151,  21.82283093,  21.76178028,
                21.68407946,  21.60082858,  21.51202764,  21.40657652,
                21.29557535,  21.17902411,  21.0458227 ,  20.91262129,
                20.76276971,  20.60736806,  20.4408663 ,  20.26881448,
                20.0912126 ,  19.90251061,  19.70270849,  19.50290638,
                19.29200415,  19.0700018 ,  18.84799945,  18.61489698,
                18.37624445,  18.12649181,  17.87673917,  17.62143647,
                17.36058371,  17.08863083,  16.81667795,  16.53917502,
                16.25612202,  15.97306902,  15.67891591,  15.3847628 ,
                15.09060969,  14.79090652,  14.48565328,  14.18040005,
                13.87514682,  13.56434354,  13.25354025,  12.94273696,
                12.62638361,  12.31558032,  11.99922697,  11.68842369,
                11.37207034,  11.06126705,  10.7449137 ,  10.43411041,
                10.12885718,   9.81805389,   9.51280066,   9.20754743,
                 8.90784426,   8.61369115,   8.31953804,   8.03093498,
                 7.74233193,   7.45927893,   7.181776  ,   6.90982312,
                 6.6434203 ,   6.37701748,   6.12171478,   5.87196214,
                 5.62775955,   5.38910703,   5.15600456,   4.92845215,
                 4.71199986,   4.50109763,   4.29574546,   4.09594334,
                 3.90724135,   3.72963947,   3.55758765,   3.39108588,
                 3.23568424,   3.09138271,   2.95263124,   2.82497989,
                 2.7028786 ,   2.59187743,   2.49197637,   2.39762537,
                 2.31437449,   2.24222373,   2.17562302,   2.12567249,
                 2.08127202,   2.04242161,   2.02022138,   2.0035712 ,
                 1.99802114,   2.0035712 ,   2.02022138,   2.04242161,
                 2.08127202,   2.12567249,   2.17562302,   2.24222373,
                 2.31437449,   2.39762537,   2.49197637,   2.59187743,
                 2.7028786 ,   2.82497989,   0.        ])
        intensity_red['Shannen'] = numpy.array(
              [  4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                 4.99955044,   4.99955044,   4.99955044,   4.99955044,
                11.99780133,  12.31132297,  12.63044321,  12.93836625,
                13.25188789,  13.56540953,  13.87333257,  14.18125561,
                14.48917865,  14.79150309,  15.08822893,  15.38495477,
                15.6816806 ,  15.97280784,  16.25833648,  16.53826651,
                16.81819655,  17.09252798,  17.35566222,  17.61879645,
                17.87633208,  18.12826912,  18.37460755,  18.61534738,
                18.84489001,  19.06883404,  19.28717946,  19.49992629,
                19.70707452,  19.90302554,  20.08777937,  20.27253319,
                20.44049121,  20.60844923,   0.        ])
        intensity_red['Jennie'] = numpy.array(
              [  4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,   4.99943979,
                 4.99943979,   4.99943979,   4.99943979,  11.99976402,
                12.31569315,  12.62607965,  12.94200877,  13.25239528,
                13.56278179,  13.87316829,  14.1835548 ,  14.48839869,
                14.78769996,  15.09254385,  15.38630251,  15.68006117,
                15.97381982,  16.25649325,  16.53916667,  16.81629748,
                17.08788567,  17.35947387,  17.61997683,  17.88047979,
                18.12989752,  18.37377263,  18.61210512,  18.844895  ,
                19.07214227,  19.2883043 ,  19.49892371,  19.70400051,
                19.90353469,  20.09198364,  20.26934736,  20.44116846,
                20.60744695,  20.7626402 ,  20.91229084,  21.05085624,
                21.17833641,  21.30027397,  21.41112629,  21.51089338,
                21.60511786,  21.6882571 ,  21.76031111,  21.82127989,
                21.87670605,  21.92104698,  21.95430268,  21.98201576,
                21.99310099,  21.99864361,  21.99310099,  21.98201576,
                21.95430268,  21.92104698,  21.87670605,  21.82127989,
                21.76031111,  21.6882571 ,  21.60511786,  21.51089338,
                21.41112629,  21.30027397,  21.17833641,  21.05085624,
                20.91229084,  20.7626402 ,  20.60744695,  20.44116846,
                20.26934736,  20.09198364,  19.90353469,  19.70400051,
                19.49892371,  19.2883043 ,  19.07214227,  18.844895  ,
                18.61210512,  18.37377263,  18.12989752,  17.88047979,
                17.61997683,  17.35947387,  17.08788567,  16.81629748,
                16.53916667,  16.25649325,  15.97381982,  15.68006117,
                15.38630251,  15.09254385,  14.78769996,  14.48839869,
                14.1835548 ,  13.87316829,  13.56278179,  13.25239528,
                12.94200877,  12.62607965,  12.31569315,  11.99976402,
                11.6838349 ,  11.3734484 ,  11.05751927,  10.74713277,
                10.43674626,  10.12635976,   9.81597325,   9.51112936,
                 9.21182809,   8.91252681,   8.61322554,   8.31946688,
                 8.03125084,   7.7430348 ,   7.46036137,   7.18323057,
                 6.91164237,   6.64005418,   6.37955122,   6.12459088,
                 5.86963053,   5.62575542,   5.38742292,   5.15463304,
                 4.92738578,   4.71122375,   4.50060434,   4.29552754,
                 4.09599335,   3.9075444 ,   3.73018069,   3.55835959,
                 3.3920811 ,   3.23688785,   3.08723721,   2.95421442,
                 2.82119163,   2.70479669,   2.59394437,   2.48863466,
                 2.39441019,   2.31681356,   2.23921694,   2.17824816,
                 2.122822  ,   2.07848107,   2.04522537,   2.01751229,
                 2.00642706,   2.00088444,   2.00642706,   2.01751229,
                 2.04522537,   2.07848107,   2.122822  ,   2.17824816,
                 2.23921694,   2.31681356,   2.39441019,   2.48863466,
                 2.59394437,   2.70479669,   2.82119163,   2.95421442,
                 3.08723721,   3.23688785,   3.3920811 ,   3.55835959,
                 3.73018069,   3.9075444 ,   4.09599335,   4.29552754,
                 4.50060434,   4.71122375,   4.92738578,   5.15463304,
                 5.38742292,   5.62575542,   5.86963053,   6.12459088,
                 6.37955122,   6.64005418,   0.        ])
        intensity_red['Kirk'] = numpy.array(
              [  5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,   5.0014085 ,
                 5.0014085 ,   5.0014085 ,   5.0014085 ,  12.00006088,
                12.31541518,  12.62523694,  12.94059124,  13.25594554,
                13.5657673 ,  13.87558907,  14.1798783 ,  14.48416754,
                14.78845677,  15.092746  ,  15.38597018,  15.67919435,
                15.97241852,  16.26011016,  16.54226926,  16.81889584,
                17.08998989,  17.3555514 ,  17.62111291,  17.87560936,
                18.13010581,  18.3735372 ,  18.61143606,  18.84380238,
                19.07063617,  19.29193743,  19.50217363,  19.7068773 ,
                19.9005159 ,  20.08862197,  20.27119551,  20.44270399,
                20.60867994,  20.76359082,  20.90743664,  21.04574993,
                21.17853068,  21.30024638,  21.41089701,  21.51048258,
                21.60453561,  21.68752358,  21.75944649,  21.82030434,
                21.87562966,  21.91988991,  21.9530851 ,  21.98074776,
                21.99734535,  21.99734535,  21.99734535,  21.98074776,
                21.9530851 ,  21.91988991,  21.87562966,  21.82030434,
                21.75944649,  21.68752358,  21.60453561,  21.51048258,
                21.41089701,  21.30024638,   0.        ])
        intensity_red['Picard'] = numpy.array(
              [  4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,   4.99839492,   4.99839492,   4.99839492,
                 4.99839492,  12.00057606,  12.31608936,  12.62606734,
                12.94158064,  13.25155862,  13.56707192,  13.87151458,
                14.18149256,  14.48593522,  14.79037788,  15.08928521,
                15.38819255,  15.68156457,  15.96940126,  16.25723796,
                16.53953933,  16.81630539,  17.09307144,  17.35876685,
                17.61892694,  17.87908703,   0.        ])
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
            numpy.testing.assert_almost_equal(lpa.dc,
                                              self.default_dc[lpa_name])
            # Grayscale calibration
            numpy.testing.assert_almost_equal(lpa.gcal,
                                              self.default_gcal[lpa_name])
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

