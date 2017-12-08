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

