# -*- coding: UTF-8 -*-
"""
Unit tests for inducer classes

"""

import itertools
import os
import random
import six
import shutil
import unittest

import numpy
import pandas

import lpadesign

class TestLPAInducerBase(unittest.TestCase):
    """
    Tests for the LPAInducerBase class.

    """
    def setUp(self):
        # Directory where to save temporary files
        self.temp_dir = "test/temp_lpa_inducer_base"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        ind = lpadesign.inducer.LPAInducerBase(name='Red Light')

    def test_default_attributes(self):
        ind = lpadesign.inducer.LPAInducerBase(name='Red Light')
        self.assertEqual(ind.name, 'Red Light')
        # Default parameters
        self.assertEqual(ind.units, u'µmol/(m^2*s)')
        self.assertIsNone(ind.led_layout)
        self.assertIsNone(ind.led_channel)
        # Time step attributes
        self.assertIsNone(ind.time_step_size)
        self.assertIsNone(ind.time_step_units)
        self.assertIsNone(ind.n_time_steps)
        # Shuffling attributes
        self.assertTrue(ind.shuffling_enabled)
        self.assertIsNone(ind.shuffled_idx)
        self.assertEqual(ind.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(ind._doses_table, pandas.DataFrame)
        self.assertTrue(ind._doses_table.empty)
        self.assertIsInstance(ind.doses_table, pandas.DataFrame)
        self.assertTrue(ind.doses_table.empty)

    def test_non_default_attributes(self):
        ind = lpadesign.inducer.LPAInducerBase(name='Red Light',
                                               units=u'µmol/(m^2*s)',
                                               led_layout='660nm',
                                               led_channel=2)
        self.assertEqual(ind.name, 'Red Light')
        # From parameters
        self.assertEqual(ind.units, u'µmol/(m^2*s)')
        self.assertEqual(ind.led_layout, '660nm')
        self.assertEqual(ind.led_channel, 2)
        # Time step attributes
        self.assertIsNone(ind.time_step_size)
        self.assertIsNone(ind.time_step_units)
        self.assertIsNone(ind.n_time_steps)
        # Shuffling attributes
        self.assertTrue(ind.shuffling_enabled)
        self.assertIsNone(ind.shuffled_idx)
        self.assertEqual(ind.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(ind._doses_table, pandas.DataFrame)
        self.assertTrue(ind._doses_table.empty)
        self.assertIsInstance(ind.doses_table, pandas.DataFrame)
        self.assertTrue(ind.doses_table.empty)

    def test_shuffle_not_implemented(self):
        ind = lpadesign.inducer.LPAInducerBase(name='Red Light')
        with self.assertRaises(NotImplementedError):
            ind.get_lpa_intensity(10)

    def test_no_effect_functions(self):
        ind = lpadesign.inducer.LPAInducerBase(name='Red Light')
        ind.save_exp_setup_instructions()
        ind.save_exp_setup_files()
        ind.save_rep_setup_instructions()
        ind.save_rep_setup_files()

class TestLightInducer(unittest.TestCase):
    """
    Tests for the LightInducer class.

    """
    def setUp(self):
        # Directory where to save temporary files
        self.temp_dir = "test/temp_light_inducer"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        light_520 = lpadesign.inducer.LightInducer(name='520nm Light')

    def test_default_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(name='520nm Light')
        # Default main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, u'µmol/(m^2*s)')
        self.assertIsNone(light_520.led_layout)
        self.assertIsNone(light_520.led_channel)
        self.assertEqual(light_520.id_prefix, '5')
        self.assertEqual(light_520.id_offset, 0)
        # Time step attributes
        self.assertEqual(light_520.time_step_size, 60000)
        self.assertEqual(light_520.time_step_units, 'min')
        self.assertIsNone(light_520.n_time_steps)
        # Shuffling attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertIsNone(light_520.shuffled_idx)
        self.assertEqual(light_520.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(light_520._doses_table, pandas.DataFrame)
        self.assertTrue(light_520._doses_table.empty)
        self.assertIsInstance(light_520.doses_table, pandas.DataFrame)
        self.assertTrue(light_520.doses_table.empty)
        # Headers
        self.assertEqual(light_520._intensities_header,
                         u"520nm Light Intensity (µmol/(m^2*s))")

    def test_custom_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G',
            id_offset=24)
        # Main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, 'W/m^2')
        self.assertEqual(light_520.led_layout, '520-2-KB')
        self.assertEqual(light_520.led_channel, 1)
        self.assertEqual(light_520.id_prefix, 'G')
        self.assertEqual(light_520.id_offset, 24)
        # Time step attributes
        self.assertEqual(light_520.time_step_size, 60000)
        self.assertEqual(light_520.time_step_units, 'min')
        self.assertIsNone(light_520.n_time_steps)
        # Shuffling attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertIsNone(light_520.shuffled_idx)
        self.assertEqual(light_520.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(light_520._doses_table, pandas.DataFrame)
        self.assertTrue(light_520._doses_table.empty)
        self.assertIsInstance(light_520.doses_table, pandas.DataFrame)
        self.assertTrue(light_520.doses_table.empty)
        # Headers
        self.assertEqual(light_520._intensities_header,
                         u"520nm Light Intensity (W/m^2)")

    def test_intensities_assignment(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.linspace(0,1,11)
        # Test doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.linspace(0,1,11))

    def test_intensities_assignment_custom_id(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G',
            id_offset=24)
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.linspace(0,1,11)
        # Test doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (W/m^2)': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1 + 24) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.linspace(0,1,11))

    def test_set_gradient_linear(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=0, max=1, n=21)
        # Check doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,21)},
            index=['G{:03d}'.format(i + 1) for i in range(21)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.linspace(0,1,21))

    def test_set_gradient_linear_repeat(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=0, max=1, n=12, n_repeat=3)
        # Check doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.repeat(numpy.linspace(0,1,4), 3)},
            index=['G{:03d}'.format(i + 1) for i in range(12)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.repeat(numpy.linspace(0,1,4), 3))

    def test_set_gradient_linear_repeat_error(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        with self.assertRaises(ValueError):
            light_520.set_gradient(min=0, max=1, n=11, n_repeat=3)

    def test_set_gradient_log(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=1e-6, max=1e-3, n=10, scale='log')
        # Check doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.logspace(-6,-3,10)},
            index=['G{:03d}'.format(i + 1) for i in range(10)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.logspace(-6,-3,10))

    def test_set_gradient_log_repeat(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=1e-6, max=1e-3, n=12, scale='log', n_repeat=3)
        # Check doses table
        conc = numpy.repeat(numpy.logspace(-6,-3,4), 3)
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': conc},
            index=['G{:03d}'.format(i + 1) for i in range(12)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities, conc)

    def test_set_gradient_log_zero(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=1e-6, max=1e-3, n=10, scale='log', use_zero=True)
        # Check doses table
        conc = numpy.append([0], numpy.logspace(-6,-3,9))
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': conc},
            index=['G{:03d}'.format(i + 1) for i in range(10)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities, conc)

    def test_set_gradient_log_zero_repeat(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities from gradient
        light_520.set_gradient(min=1e-6,
                               max=1e-3,
                               n=12,
                               scale='log',
                               use_zero=True,
                               n_repeat=2)
        # Check doses table
        conc = numpy.repeat(numpy.append([0], numpy.logspace(-6,-3,5)), 2)
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': conc},
            index=['G{:03d}'.format(i + 1) for i in range(12)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities, conc)

    def test_set_gradient_scale_error(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        with self.assertRaises(ValueError):
            light_520.set_gradient(min=1e-6, max=1e-3, n=10, scale='symlog')

    def test_get_lpa_intensities_error(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities
        light_520.intensities = numpy.linspace(0,1,11)
        # Calling get_lpa_intensity without setting number of steps should raise
        # an exception.
        errmsg = 'number of time steps should be indicated'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            light_520.get_lpa_intensity(0)

    def test_get_lpa_intensities(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities
        light_520.intensities = numpy.arange(10)*5.
        # Set number of time steps
        light_520.n_time_steps = 20
        # Check output of get_lpa_intensity
        for i in range(10):
            numpy.testing.assert_almost_equal(
                light_520.get_lpa_intensity(i),
                numpy.ones(20)*5*i)

    def test_get_lpa_intensities_with_shuffling(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Set intensities
        light_520.intensities = numpy.arange(10)*5.
        # Set number of time steps
        light_520.n_time_steps = 20
        # Shuffle intensities
        random.seed(1)
        light_520.shuffle()
        # Check output of get_lpa_intensity
        # Values have been obtained from shuffling with seed 1
        # Shuffling results are different in python 2 and 3
        if six.PY2:
            values = numpy.array([8, 0, 3, 4, 5, 2, 9, 6, 7, 1])*5
        elif six.PY3:
            values = numpy.array([6, 8, 9, 7, 5, 3, 0, 4, 1, 2])*5
        for i in range(10):
            numpy.testing.assert_almost_equal(
                light_520.get_lpa_intensity(i),
                numpy.ones(20)*values[i])

    def test_shuffle(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)
        # Shuffle
        random.seed(1)
        light_520.shuffle()
        # The following indices give the correct shuffled intensities array
        # after setting the random seed to one.
        # Shuffling results are different in python 2 and 3
        if six.PY2:
            shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
        else:
            shuffling_ind = [6, 8, 10, 7, 5, 3, 0, 4, 1, 9, 2]
        # Check concentrations
        intensities = numpy.linspace(0,1,11)
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          intensities[shuffling_ind])
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_520.doses_table,
                                               df.iloc[shuffling_ind])

    def test_shuffle_disabled(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)
        # Disable shuffling
        light_520.shuffling_enabled = False
        # Shuffle
        random.seed(1)
        light_520.shuffle()
        # Check intensities
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          numpy.linspace(0,1,11))
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)

    def test_sync_shuffling_fail(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,12)

        with self.assertRaises(ValueError):
            light_520.sync_shuffling(light_660)

    def test_sync_shuffling_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,11)

        # Sync shuffling
        light_520.sync_shuffling(light_660)
        # Check attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertFalse(light_660.shuffling_enabled)
        self.assertEqual(light_520.shuffling_sync_list, [light_660])
        self.assertEqual(light_660.shuffling_sync_list, [])

    def test_sync_shuffling_no_shuffling_in_dependent(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,11)

        # Sync shuffling
        light_520.sync_shuffling(light_660)

        # Shuffle dependent inducer
        random.seed(1)
        light_660.shuffle()
        # Check intensities
        numpy.testing.assert_almost_equal(light_660.intensities,
                                          numpy.linspace(0,1,11) + 2)
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'660nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11) + 2},
            index=['R{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_660._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_660.doses_table, df)

    def test_sync_shuffling(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,11)

        # Sync shuffling
        light_520.sync_shuffling(light_660)

        # Shuffle both inducers by calling the first one
        random.seed(1)
        light_520.shuffle()

        # The following indices give the correct shuffled intensities array
        # after setting the random seed to one.
        # Shuffling results are different in python 2 and 3
        if six.PY2:
            shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
        else:
            shuffling_ind = [6, 8, 10, 7, 5, 3, 0, 4, 1, 9, 2]
        # Check intensities for independent inducer
        intensities = numpy.linspace(0,1,11)
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          intensities[shuffling_ind])
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_520.doses_table,
                                               df.iloc[shuffling_ind])
        # Check intensities for dependent inducer
        intensities = numpy.linspace(2,3,11)
        numpy.testing.assert_almost_equal(light_660.intensities,
                                          intensities[shuffling_ind])
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'660nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11) + 2},
            index=['R{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_660._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_660.doses_table,
                                               df.iloc[shuffling_ind])

    def test_sync_shuffling_2(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=0,
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            led_channel=1,
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,11)

        # Sync shuffling
        light_520.sync_shuffling(light_660)

        # Shuffle both inducers by calling the first one
        random.seed(1)
        light_520.shuffle()
        # Attempt to shuffle the dependent inducer to make sure it doesn't ruin
        # the shuffling.
        light_660.shuffle()
        # The following indices give the correct shuffled intensities array
        # after setting the random seed to one.
        # Shuffling results are different in python 2 and 3
        if six.PY2:
            shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
        else:
            shuffling_ind = [6, 8, 10, 7, 5, 3, 0, 4, 1, 9, 2]
        # Check intensities for independent inducer
        intensities = numpy.linspace(0,1,11)
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          intensities[shuffling_ind])
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'520nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11)},
            index=['G{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_520.doses_table,
                                               df.iloc[shuffling_ind])
        # Check intensities for dependent inducer
        intensities = numpy.linspace(2,3,11)
        numpy.testing.assert_almost_equal(light_660.intensities,
                                          intensities[shuffling_ind])
        # Check unshuffled doses table
        df = pandas.DataFrame(
            {u'660nm Light Intensity (µmol/(m^2*s))': numpy.linspace(0,1,11) + 2},
            index=['R{:03d}'.format(i + 1) for i in range(11)])
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_660._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_660.doses_table,
                                               df.iloc[shuffling_ind])

class TestLightSignal(unittest.TestCase):
    """
    Tests for the LightSignal class.

    """
    def setUp(self):
        # Directory where to save temporary files
        self.temp_dir = "test/temp_light_signal"
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

    def tearDown(self):
        # Delete temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_create(self):
        light_520 = lpadesign.inducer.LightSignal(name='520nm Light')

    def test_default_attributes(self):
        light_520 = lpadesign.inducer.LightSignal(name='520nm Light')
        # Default main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, u'µmol/(m^2*s)')
        self.assertIsNone(light_520.led_layout)
        self.assertIsNone(light_520.led_channel)
        self.assertEqual(light_520.id_prefix, '5')
        self.assertEqual(light_520.id_offset, 0)
        # Time step attributes
        self.assertEqual(light_520.time_step_size, 60000)
        self.assertEqual(light_520.time_step_units, 'min')
        self.assertEqual(light_520.n_time_steps, 0)
        # Shuffling attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertIsNone(light_520.shuffled_idx)
        self.assertEqual(light_520.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(light_520._doses_table, pandas.DataFrame)
        self.assertTrue(light_520._doses_table.empty)
        self.assertIsInstance(light_520.doses_table, pandas.DataFrame)
        self.assertTrue(light_520.doses_table.empty)
        # Headers
        self.assertEqual(light_520._signal_labels_header,
                         u"520nm Light Signal Label")
        self.assertEqual(light_520._intensities_headers, [])

    def test_custom_attributes(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G',
            id_offset=24)
        # Main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, 'W/m^2')
        self.assertEqual(light_520.led_layout, '520-2-KB')
        self.assertEqual(light_520.led_channel, 1)
        self.assertEqual(light_520.id_prefix, 'G')
        self.assertEqual(light_520.id_offset, 24)
        # Time step attributes
        self.assertEqual(light_520.time_step_size, 60000)
        self.assertEqual(light_520.time_step_units, 'min')
        self.assertEqual(light_520.n_time_steps, 0)
        # Shuffling attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertIsNone(light_520.shuffled_idx)
        self.assertEqual(light_520.shuffling_sync_list, [])
        # Doses table
        self.assertIsInstance(light_520._doses_table, pandas.DataFrame)
        self.assertTrue(light_520._doses_table.empty)
        self.assertIsInstance(light_520.doses_table, pandas.DataFrame)
        self.assertTrue(light_520.doses_table.empty)
        # Headers
        self.assertEqual(light_520._signal_labels_header,
                         u"520nm Light Signal Label")
        self.assertEqual(light_520._intensities_headers, [])

    def test_intensities_assignment(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])

        # Test number of time steps
        self.assertEqual(light_520.n_time_steps, 2)
        # Test doses table
        df = pandas.DataFrame()
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 0 min'] = \
                numpy.linspace(0,1,11)
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 1 min'] = \
                numpy.linspace(10,20,11)
        df[u'520nm Light Signal Label'] = [""]*11
        df.index=['G{:03d}'.format(i + 1) for i in range(11)]
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.array([numpy.linspace(0,1,11),
                                                      numpy.linspace(10,20,11)]))
        # Test signal labels attribute
        numpy.testing.assert_array_equal(light_520.signal_labels, ['']*11)

    def test_n_time_steps_writing(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])

        # Test number of time steps
        self.assertEqual(light_520.n_time_steps, 2)

        # Writing to n_time_steps should have no effect
        light_520.n_time_steps = 5
        self.assertEqual(light_520.n_time_steps, 2)

    def test_intensities_assignment_custom_id(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G',
            id_offset=24)
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])

        # Test number of time steps
        self.assertEqual(light_520.n_time_steps, 2)
        # Test doses table
        df = pandas.DataFrame()
        df[u'520nm Light Intensity (W/m^2) at t = 0 min'] = \
                numpy.linspace(0,1,11)
        df[u'520nm Light Intensity (W/m^2) at t = 1 min'] = \
                numpy.linspace(10,20,11)
        df[u'520nm Light Signal Label'] = [""]*11
        df.index=['G{:03d}'.format(i + 1 +24) for i in range(11)]
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.array([numpy.linspace(0,1,11),
                                                      numpy.linspace(10,20,11)]))
        # Test signal labels attribute
        numpy.testing.assert_array_equal(light_520.signal_labels, ['']*11)

    def test_intensities_and_signal_labels_assignment(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])
        # Writing signal labels should add to the doses table
        light_520.signal_labels = ["Signal {}".format(i+1)
                                   for i in range(11)]

        # Test number of time steps
        self.assertEqual(light_520.n_time_steps, 2)
        # Test doses table
        df = pandas.DataFrame()
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 0 min'] = \
                numpy.linspace(0,1,11)
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 1 min'] = \
                numpy.linspace(10,20,11)
        df[u'520nm Light Signal Label'] = \
            ["Signal {}".format(i+1) for i in range(11)]
        df.index=['G{:03d}'.format(i + 1) for i in range(11)]
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test intensities attribute
        numpy.testing.assert_array_equal(light_520.intensities,
                                         numpy.array([numpy.linspace(0,1,11),
                                                      numpy.linspace(10,20,11)]))
        # Test signal labels attribute
        numpy.testing.assert_array_equal(
            light_520.signal_labels,
            ["Signal {}".format(i+1) for i in range(11)])

    def test_set_staggered_signal_error_signal_length(self):
        # Define LightSignal object
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')

        # Calling set_staggered_signal with a signal with less than 8 elements
        # should trigger an error.
        errmsg = 'signal should have at least 8 elements'
        with six.assertRaisesRegex(self, ValueError, errmsg):
            light_520.set_staggered_signal(
                signal=numpy.arange(7, dtype=float),
                signal_init=0,
                sampling_time_steps=numpy.array([0, 2, 4, 6, 8]),
                n_time_steps=15)

        light_520.set_staggered_signal(
            signal=numpy.arange(8, dtype=float),
            signal_init=0,
            sampling_time_steps=numpy.array([0, 2, 4, 6, 8]),
            n_time_steps=15)

    def test_set_staggered_signal_1(self):
        # Define signal and sampling times
        signal = numpy.arange(10, dtype=float)
        signal_init = 6
        sampling_times = numpy.array([0, 2, 4, 6, 8])

        # Define LightSignal object
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')

        # Generate staggered signal
        light_520.set_staggered_signal(signal=signal,
                                       signal_init=signal_init,
                                       sampling_time_steps=sampling_times,
                                       n_time_steps=15)

        # Test time steps
        self.assertEqual(light_520.n_time_steps, 15)

        # Test intensities
        intensities_exp = numpy.array(
            [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3, 4, 5],
             [6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=float).T
        numpy.testing.assert_array_equal(light_520.intensities, intensities_exp)

        # Test signal labels
        signal_labels_exp = ["Sampling time: {} min".format(ts)
                             for ts in sampling_times]
        numpy.testing.assert_array_equal(light_520.signal_labels,
                                         signal_labels_exp)

    def test_set_staggered_signal_2(self):
        # Define signal and sampling times
        signal = numpy.arange(18, dtype=float)
        signal_init = 3.
        sampling_times = numpy.array([0, 18, 8, 4, 16, 12])

        # Define LightSignal object
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')

        # Generate staggered signal
        light_520.set_staggered_signal(signal=signal,
                                       signal_init=signal_init,
                                       sampling_time_steps=sampling_times,
                                       n_time_steps=18)

        # Test time steps
        self.assertEqual(light_520.n_time_steps, 18)

        # Test intensities
        intensities_exp = numpy.array(
            [[3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3,  3,  3,  3],
             [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
             [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  0,  1,  2,  3,  4,  5,  6,  7],
             [3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  0,  1,  2,  3],
             [3, 3, 0, 1, 2, 3, 4, 5, 6, 7,  8,  9, 10, 11, 12, 13, 14, 15],
             [3, 3, 3, 3, 3, 3, 0, 1, 2, 3,  4,  5,  6,  7,  8,  9, 10, 11]],
            dtype=float).T
        numpy.testing.assert_array_equal(light_520.intensities, intensities_exp)

        # Test signal labels
        signal_labels_exp = ["Sampling time: {} min".format(ts)
                             for ts in sampling_times]
        numpy.testing.assert_array_equal(light_520.signal_labels,
                                         signal_labels_exp)

    def test_set_staggered_signal_no_signal_labels(self):
        # Define signal and sampling times
        signal = numpy.arange(10, dtype=float)
        signal_init = 6
        sampling_times = numpy.array([0, 2, 4, 6, 8])

        # Define LightSignal object
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')

        # Generate staggered signal
        light_520.set_staggered_signal(signal=signal,
                                       signal_init=signal_init,
                                       sampling_time_steps=sampling_times,
                                       n_time_steps=15,
                                       set_signal_labels=False)

        # Test time steps
        self.assertEqual(light_520.n_time_steps, 15)

        # Test intensities
        intensities_exp = numpy.array(
            [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3],
             [6, 6, 6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3, 4, 5],
             [6, 6, 6, 6, 6, 6, 6, 0, 1, 2, 3, 4, 5, 6, 7]], dtype=float).T
        numpy.testing.assert_array_equal(light_520.intensities, intensities_exp)

        # Test signal labels
        signal_labels_exp = [""]*5
        numpy.testing.assert_array_equal(light_520.signal_labels,
                                         signal_labels_exp)

        # Test doses table
        doses_table_exp = pandas.DataFrame()
        for i in range(15):
            doses_table_exp\
                [u'520nm Light Intensity (µmol/(m^2*s)) at t = {} min'.\
                    format(i)] = intensities_exp[i, :]
        doses_table_exp[u'520nm Light Signal Label'] = signal_labels_exp
        doses_table_exp.index=['G{:03d}'.format(i + 1) for i in range(5)]
        doses_table_exp.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table,
                                               doses_table_exp)
        pandas.util.testing.assert_frame_equal(light_520.doses_table,
                                               doses_table_exp)

    def test_get_lpa_intensities(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])
        # Writing signal labels should add to the doses table
        light_520.signal_labels = ["Signal {}".format(i+1)
                                   for i in range(11)]

        # Check output of get_lpa_intensity
        values = numpy.array([numpy.linspace(0,1,11), numpy.linspace(10,20,11)])
        for i in range(11):
            numpy.testing.assert_almost_equal(light_520.get_lpa_intensity(i),
                                              values[:,i])

    def test_get_lpa_intensities_with_shuffling(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])
        # Writing signal labels should add to the doses table
        light_520.signal_labels = ["Signal {}".format(i+1)
                                   for i in range(11)]
        # Shuffle
        random.seed(1)
        light_520.shuffle()

        # Check output of get_lpa_intensity
        values = numpy.array([numpy.linspace(0,1,11), numpy.linspace(10,20,11)])
        if six.PY2:
            values = values[:, [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]]
        elif six.PY3:
            values = values[:, [6, 8, 10, 7, 5, 3, 0, 4, 1, 9, 2]]
        for i in range(11):
            numpy.testing.assert_almost_equal(light_520.get_lpa_intensity(i),
                                              values[:,i])

    def test_shuffle(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])
        light_520.signal_labels = ['Signal {}'.format(i+1) for i in range(11)]
        # Shuffle
        random.seed(1)
        light_520.shuffle()
        # The following indices give the correct shuffled intensities array
        # after setting the random seed to one.
        # Shuffling results are different in python 2 and 3
        if six.PY2:
            shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
        else:
            shuffling_ind = [6, 8, 10, 7, 5, 3, 0, 4, 1, 9, 2]
        # Check intensities
        intensities_exp = numpy.array([numpy.linspace(0,1,11),
                                       numpy.linspace(10,20,11)])
        intensities_exp_sh = intensities_exp[:, shuffling_ind]
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          intensities_exp_sh)
        # Test signal labels attribute
        signal_labels_exp = ['Signal {}'.format(i+1) for i in range(11)]
        signal_labels_exp_sh = [signal_labels_exp[i] for i in shuffling_ind]
        numpy.testing.assert_array_equal(
            light_520.signal_labels,
            signal_labels_exp_sh)

        # Check unshuffled doses table
        df = pandas.DataFrame()
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 0 min'] = \
                intensities_exp[0]
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 1 min'] = \
                intensities_exp[1]
        df[u'520nm Light Signal Label'] = signal_labels_exp
        df.index=['G{:03d}'.format(i + 1) for i in range(11)]
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_520.doses_table,
                                               df.iloc[shuffling_ind, :])

    def test_shuffle_disabled(self):
        light_520 = lpadesign.inducer.LightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Disable shuffling
        light_520.shuffling_enabled = False
        # Writing intensities should generate a corresponding _doses_table
        light_520.intensities = numpy.array([numpy.linspace(0,1,11),
                                             numpy.linspace(10,20,11)])
        light_520.signal_labels = ['Signal {}'.format(i+1) for i in range(11)]
        # Shuffle
        random.seed(1)
        light_520.shuffle()

        # Check intensities
        intensities_exp = numpy.array([numpy.linspace(0,1,11),
                                       numpy.linspace(10,20,11)])
        numpy.testing.assert_almost_equal(light_520.intensities,
                                          intensities_exp)
        # Test signal labels attribute
        signal_labels_exp = ['Signal {}'.format(i+1) for i in range(11)]
        numpy.testing.assert_array_equal(
            light_520.signal_labels,
            signal_labels_exp)

        # Check unshuffled doses table
        df = pandas.DataFrame()
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 0 min'] = \
                intensities_exp[0]
        df[u'520nm Light Intensity (µmol/(m^2*s)) at t = 1 min'] = \
                intensities_exp[1]
        df[u'520nm Light Signal Label'] = signal_labels_exp
        df.index=['G{:03d}'.format(i + 1) for i in range(11)]
        df.index.name='ID'
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        # Check shuffled doses table
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
