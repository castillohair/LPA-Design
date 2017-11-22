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

class TestLightInducer(unittest.TestCase):
    """
    Tests for the ChemicalInducer class.

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

    def test_default_dose_table_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(name='520nm Light')
        # Default main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, u'µmol/(m^2*s)')
        self.assertEqual(light_520.id_prefix, '5')
        self.assertEqual(light_520.id_offset, 0)
        self.assertIsNone(light_520.led_layout)

        # Default doses table
        df = pandas.DataFrame()
        pandas.util.testing.assert_frame_equal(light_520._doses_table, df)
        pandas.util.testing.assert_frame_equal(light_520.doses_table, df)
        # Headers
        self.assertEqual(light_520._intensities_header,
                         u"520nm Light Intensity (µmol/(m^2*s))")

    def test_default_custom_table_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
            id_prefix='G',
            id_offset=24)
        # Default main attributes
        self.assertEqual(light_520.name, '520nm Light')
        self.assertEqual(light_520.units, 'W/m^2')
        self.assertEqual(light_520.led_layout, '520-2-KB')
        self.assertEqual(light_520.id_prefix, 'G')
        self.assertEqual(light_520.id_offset, 24)

    def test_default_shuffling_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(name='520nm Light')
        self.assertTrue(light_520.shuffling_enabled)
        self.assertIsNone(light_520.shuffled_idx)
        self.assertEqual(light_520.shuffling_sync_list, [])

    def test_intensities_assignment(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
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

    def test_concentrations_assignment_custom_id(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            units='W/m^2',
            led_layout='520-2-KB',
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
            id_prefix='G')
        with self.assertRaises(ValueError):
            light_520.set_gradient(min=0, max=1, n=11, n_repeat=3)

    def test_set_gradient_log(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
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
            id_prefix='G')
        with self.assertRaises(ValueError):
            light_520.set_gradient(min=1e-6, max=1e-3, n=10, scale='symlog')

    def test_set_vol_from_shots_1(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            id_prefix='G')
        # Call set_vol_from_shots. It should not crash.
        light_520.set_vol_from_shots(n_shots=5)

    def test_set_vol_from_shots_2(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            id_prefix='G')
        # Call set_vol_from_shots. It should not crash.
        light_520.set_vol_from_shots(n_shots=5, n_replicates=3)

    def test_shuffle(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
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
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,12)

        with self.assertRaises(ValueError):
            light_520.sync_shuffling(light_660)

    def test_sync_shuffling_attributes(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
            id_prefix='R')
        light_660.intensities = numpy.linspace(2,3,11)

        # Sync shuffling
        light_520.sync_shuffling(light_660)
        # Check attributes
        self.assertTrue(light_520.shuffling_enabled)
        self.assertFalse(light_660.shuffling_enabled)
        self.assertEqual(light_520.shuffling_sync_list, [light_660])

    def test_sync_shuffling_no_shuffling_in_dependent(self):
        light_520 = lpadesign.inducer.LightInducer(
            name='520nm Light',
            led_layout='520-2-KB',
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
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
            id_prefix='G')
        light_520.intensities = numpy.linspace(0,1,11)

        light_660 = lpadesign.inducer.LightInducer(
            name='660nm Light',
            led_layout='660-LS',
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
