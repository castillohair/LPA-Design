# -*- coding: UTF-8 -*-
import numpy
import lpaprogram
import platedesign
import lpadesign

# lpaprogram requires LED calibration data
lpaprogram.LED_CALIBRATION_PATH = "../supporting_files/LPA Calibration Data"

# Experiment
exp = platedesign.experiment.Experiment()
exp.n_replicates = 3
exp.plate_resources['LPA'] = ['Jennie',
                              'Picard',
                              'Kirk',
                              'Shannen',
                              ]
exp.randomize_inducers = True
exp.randomize_plate_resources = True
exp.measurement_template = '../supporting_files/template_FlowCal.xlsx'
exp.measurement_order = 'LPA'
exp.replicate_measurements = ['Date', 'Run by']
exp.plate_measurements = ['Final OD600', 'Incubation time (min)']

# Inducers
sampling_times = [  0,   5,  10,  15,  20,  25,
                   30,  40,  50,  60,  70,  80,
                   90, 100, 110, 120, 130, 140,
                  150, 160, 170, 180, 190, 200,]
# 520nm (green) light: step
light_520 = lpadesign.inducer.StaggeredLightSignal(name='520nm Light',
                                                   led_layout='520-2-KB',
                                                   led_channel=0,
                                                   id_prefix='G')
light_520.sampling_time_steps = sampling_times
light_520.set_step(initial=0, final=50)
# Light program time in minutes
# Only needs to be specified in one light inducer
light_520.n_time_steps = 8*60

exp.add_inducer(light_520)

# 660nm (red) light: constant
light_660 = lpadesign.inducer.LightInducer(name='660nm Light',
                                           led_layout='660-LS',
                                           led_channel=1,
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
plate.cell_predilution_vol = 300
plate.cell_shot_vol = 5
plate.apply_inducer(inducer=light_520, apply_to='wells')
plate.apply_inducer(inducer=light_660, apply_to='wells')
exp.add_plate(plate)

exp.generate()