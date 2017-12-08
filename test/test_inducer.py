# -*- coding: UTF-8 -*-
"""
Unit tests for inducer classes

"""

import itertools
import os
import random
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
        with self.assertRaisesRegexp(ValueError, errmsg):
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
        values = numpy.array([8, 0, 3, 4, 5, 2, 9, 6, 7, 1])*5
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
        shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
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
        shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
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
        shuffling_ind = [10, 5, 0, 4, 9, 7, 3, 2, 6, 8, 1]
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

class TestStaggeredLightSignal(unittest.TestCase):
    """
    Tests for the LightInducer class.

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
        light_520 = lpadesign.inducer.StaggeredLightSignal(name='520nm Light')

    def test_default_attributes(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(name='520nm Light')
        # Default main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, u'µmol/(m^2*s)')
        self.assertIsNone(light_520.led_layout)
        self.assertIsNone(light_520.led_channel)
        self.assertEqual(light_520.id_prefix, '5')
        self.assertEqual(light_520.id_offset, 0)
        # Light signal attributes
        numpy.testing.assert_almost_equal(light_520.signal,
                                          numpy.array([]))
        self.assertEqual(light_520.signal_init, 0)
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
        self.assertEqual(light_520._sampling_time_steps_header,
                         u"520nm Light Sampling Time (min)")
        self.assertEqual(light_520._signal_file_name_header,
                         u"520nm Light Signal File")
        self.assertEqual(light_520._signal_file_name,
                         u"520nm Light Signal.xlsx")

    def test_custom_attributes(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
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
        # Light signal attributes
        numpy.testing.assert_almost_equal(light_520.signal,
                                          numpy.array([]))
        self.assertEqual(light_520.signal_init, 0)
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
        self.assertEqual(light_520._sampling_time_steps_header,
                         u"520nm Light Sampling Time (min)")
        self.assertEqual(light_520._signal_file_name_header,
                         u"520nm Light Signal File")
        self.assertEqual(light_520._signal_file_name,
                         u"520nm Light Signal.xlsx")

    def test_sampling_time_steps_assignment(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Writing sampling times should generate a corresponding _doses_table
        light_520.sampling_time_steps = numpy.arange(10)*2
        # Test doses table
        df = pandas.DataFrame(
            index=['G{:03d}'.format(i + 1) for i in range(10)])
        df.index.name='ID'
        df['520nm Light Signal File'] = "520nm Light Signal.xlsx"
        df['520nm Light Sampling Time (min)'] = numpy.arange(10)*2
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test sampling times attribute
        numpy.testing.assert_array_equal(light_520.sampling_time_steps,
                                         numpy.arange(10, dtype=int)*2)

    def test_sampling_time_steps_assignment_custom_id(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G',
            id_offset=24)
        # Writing sampling times should generate a corresponding _doses_table
        light_520.sampling_time_steps = numpy.arange(10)*2
        # Test doses table
        df = pandas.DataFrame(
            index=['G{:03d}'.format(i + 1 + 24) for i in range(10)])
        df.index.name='ID'
        df['520nm Light Signal File'] = "520nm Light Signal.xlsx"
        df['520nm Light Sampling Time (min)'] = numpy.arange(10)*2
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Test sampling times attribute
        numpy.testing.assert_array_equal(light_520.sampling_time_steps,
                                         numpy.arange(10, dtype=int)*2)

    def test_set_step_no_sampling_time_steps(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Call set step without setting sampling times
        errmsg = 'n_time_steps or sampling time steps should be specified'
        with self.assertRaisesRegexp(ValueError, errmsg):
            light_520.set_step(0, 50)

    def test_set_step_1(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Write sampling times
        light_520.sampling_time_steps = numpy.arange(10)*2
        # Call set step
        light_520.set_step(0, 50)
        # Check signal
        self.assertEqual(light_520.signal_init, 0)
        numpy.testing.assert_array_equal(light_520.signal, numpy.ones(18)*50)

    def test_set_step_2(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Write sampling times
        light_520.sampling_time_steps = numpy.array([0, 15, 30])
        # Call set step
        light_520.set_step(30., 10.)
        # Check signal
        self.assertEqual(light_520.signal_init, 30.)
        numpy.testing.assert_array_equal(light_520.signal, numpy.ones(30)*10)

    def test_save_exp_setup_files(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Write signal
        light_520.signal = numpy.array([3, 3, 4, 2, 2, 2, 1, 5])
        light_520.signal_init = 2
        # Save file
        light_520.save_exp_setup_files(path=self.temp_dir)

        # Load file with signal values
        signal_df = pandas.read_excel(
            os.path.join(self.temp_dir, "520nm Light Signal.xlsx"))
        # Expected file contents
        signal_exp = pandas.DataFrame(columns=[
            u"Time (min)",
            u"520nm Light Intensity (µmol/(m^2*s))"])
        signal_exp[u"Time (min)"] = \
            numpy.array([-numpy.inf, 0, 1, 2, 3, 4, 5, 6, 7])
        signal_exp[u"520nm Light Intensity (µmol/(m^2*s))"] = \
            numpy.array([2, 3, 3, 4, 2, 2, 2, 1, 5])
        # Check
        pandas.testing.assert_frame_equal(signal_df, signal_exp)

    def test_get_lpa_intensity_1(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Write sampling times, signal, etc
        light_520.n_time_steps = 10
        light_520.sampling_time_steps = [0, 2, 7, 8]
        light_520.signal = numpy.array([3, 3, 4, 2, 2, 2, 1, 5])
        light_520.signal_init = 2
        # Check lpa intensities
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(0),
            numpy.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(1),
            numpy.array([2, 2, 2, 2, 2, 2, 2, 2, 3, 3], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(2),
            numpy.array([2, 2, 2, 3, 3, 4, 2, 2, 2, 1], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(3),
            numpy.array([2, 2, 3, 3, 4, 2, 2, 2, 1, 5], dtype=float))

    def test_get_lpa_intensity_2(self):
        light_520 = lpadesign.inducer.StaggeredLightSignal(
            name='520nm Light',
            led_layout='520-2-KB',
            led_channel=1,
            id_prefix='G')
        # Write sampling times, signal, etc
        light_520.n_time_steps = 5
        light_520.sampling_time_steps = [0, 2, 7, 8]
        light_520.signal = numpy.array([3, 3, 4, 2, 2, 2, 1, 5])
        light_520.signal_init = 2
        # Check lpa intensities
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(0),
            numpy.array([2, 2, 2, 2, 2], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(1),
            numpy.array([2, 2, 2, 3, 3], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(2),
            numpy.array([4, 2, 2, 2, 1], dtype=float))
        numpy.testing.assert_array_equal(
            light_520.get_lpa_intensity(3),
            numpy.array([2, 2, 2, 1, 5], dtype=float))
