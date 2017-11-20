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
exp.plate_locations = ['Jennie',
                       'Picard',
                       'Kirk',
                       'Shannen',
                       'Sisko',
                       ]
exp.randomize_inducers = True
exp.randomize_plates = False
exp.measurement_template = '../supporting_files/template_FlowCal.xlsx'
exp.replicate_measurements = ['Date', 'Run by']
exp.plate_measurements = ['Final OD600', 'Incubation time (min)']

# Inducers
# 520nm (green) light: log gradient
light_520 = lpadesign.inducer.LightInducer(name='520nm Light',
                                           led_layout='520-2-KB',
                                           id_prefix='G')
light_520.set_gradient(min=0.1,
                       max=100.,
                       n=24,
                       scale='log',
                       use_zero=True)
exp.add_inducer(light_520)

# 660nm (red) light: constant throughout all wells
light_660 = lpadesign.inducer.LightInducer(name='660nm Light',
                                           led_layout='660-LS',
                                           id_prefix='R')
light_660.intensities = numpy.ones(24)*20.
exp.add_inducer(light_660)

# Plate for light-sensitive strain
plate = lpadesign.plate.LPAPlate(name='P1')
plate.cell_strain_name = 'Light sensing strain 1'
plate.total_media_vol = 16000.
plate.sample_media_vol = 500.
plate.cell_setup_method = 'fixed_volume'
plate.cell_predilution = 100
plate.cell_predilution_vol = 1000
plate.cell_shot_vol = 5
plate.apply_inducer(inducer=light_520, apply_to='wells', led_channel=0)
plate.apply_inducer(inducer=light_660, apply_to='wells', led_channel=1)
exp.add_plate(plate)

exp.generate()