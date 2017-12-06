# -*- coding: UTF-8 -*-
import numpy
import lpaprogram
import platedesign
import lpadesign

# lpaprogram requires LED calibration data
lpaprogram.LED_CALIBRATION_PATH = "../supporting_files/led-calibration"

# Experiment
exp = platedesign.experiment.Experiment()
exp.n_replicates = 5
exp.plate_resources['LPA'] = ['Jennie',
                              'Picard',
                              'Kirk',
                              'Shannen',
                              'Sisko',
                              ]
exp.randomize_inducers = True
exp.randomize_plate_resources = False
exp.measurement_template = '../supporting_files/template_FlowCal.xlsx'
exp.measurement_order = 'LPA'
exp.replicate_measurements = ['Date', 'Run by']
exp.plate_measurements = ['Final OD600', 'Incubation time (min)']

# Light sensing strain will use a 2x2 plate array. It will have xylose and
# alternating red and green light across 12 columns, and iptg across 8 rows.
# Shuffling of light inducers should be synchronized with xylose.

iptg = platedesign.inducer.ChemicalInducer(name='IPTG', units=u'µM')
iptg.stock_conc = 1e6
iptg.shot_vol = 5.
iptg.set_gradient(min=0.5,
                  max=500,
                  n=8,
                  scale='log',
                  use_zero=True)
exp.add_inducer(iptg)

xyl = platedesign.inducer.ChemicalInducer(name='Xylose', units='%')
xyl.stock_conc = 50.
xyl.shot_vol = 5.
xyl.set_gradient(min=5e-3,
                 max=0.5,
                 n=12,
                 n_repeat=2,
                 scale='log',
                 use_zero=True)
exp.add_inducer(xyl)

light_520 = lpadesign.inducer.LightInducer(name='520nm Light',
                                           led_layout='520-2-KB',
                                           led_channel=0,
                                           id_prefix='G')
light_520.intensities = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])*50
exp.add_inducer(light_520)
xyl.sync_shuffling(light_520)

# 660nm (red) light: constant throughout all wells
light_660 = lpadesign.inducer.LightInducer(name='660nm Light',
                                           led_layout='660-LS',
                                           led_channel=1,
                                           id_prefix='R')
light_660.intensities = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])*20
exp.add_inducer(light_660)
xyl.sync_shuffling(light_660)

# LPA array for light-sensing strain
platearray = lpadesign.plate.LPAPlateArray(
    'PA1',
    array_n_rows=2,
    array_n_cols=2,
    plate_names=['P1', 'P2', 'P3', 'P4'],
    plate_n_rows=4,
    plate_n_cols=6)
platearray.cell_strain_name = 'Light-Sensing Strain'
platearray.total_media_vol = 16000.*4
platearray.apply_inducer(inducer=light_520, apply_to='rows')
platearray.apply_inducer(inducer=light_660, apply_to='rows')
platearray.apply_inducer(inducer=xyl, apply_to='rows')
platearray.apply_inducer(inducer=iptg, apply_to='cols')
exp.add_plate(platearray)

# Plate for autofluorescence control strain
plate = platedesign.plate.Plate('P5', n_rows=4, n_cols=6)
plate.cell_strain_name = 'Autofluorescence Control Strain'
plate.samples_to_measure = 4
plate.total_media_vol = 16000.
exp.add_plate(plate)

# Add common settings to plates
for plate in exp.plates:
    plate.sample_media_vol = 500.
    plate.cell_setup_method = 'fixed_volume'
    plate.cell_predilution = 100
    plate.cell_predilution_vol = 1000
    plate.cell_shot_vol = 5

# Light program time in minutes
# Only needs to be specified in one light inducer
light_520.n_time_steps = 8*60

exp.generate()
