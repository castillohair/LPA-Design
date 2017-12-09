# -*- coding: UTF-8 -*-
"""
Unit tests for plate classes

"""

import collections
import itertools
import os
import random
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
        # something with p.lpa
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
        # something with p.lpa
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
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'time step units should be specified'
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'number of time steps should be specified'
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = 1000*60

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = 'min'

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step size
        light_660.time_step_size = None

        # Attempt to generate rep setup files
        errmsg = 'all time step units should be the same'
        with self.assertRaisesRegexp(ValueError, errmsg):
            p.save_rep_setup_files(path=self.temp_dir)

        # Specify time step units
        light_660.time_step_units = None

        # Attempt to generate rep setup files
        errmsg = 'all number of time steps should be the same'
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        light_520.sampling_time_steps = numpy.arange(24)*3
        p.apply_inducer(light_520, 'wells')
        # Shuffle
        random.seed(1)
        light_520.shuffle()

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
        light_660.sampling_time_steps = numpy.array([10., 20., 30., 72.])
        p.apply_inducer(light_660, 'cols')
        # Shuffle
        random.seed(1)
        light_660.shuffle()

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
