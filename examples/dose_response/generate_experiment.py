import numpy
import platedesign
import lpadesign

# lpadesign requires LED calibration data
lpadesign.LED_CALIBRATION_PATH = "../test/test_lpa_files/led-calibration"

# Experiment
exp = platedesign.experiment.Experiment()
exp.n_replicates = 5
exp.randomize = True
exp.measurement_template = '../supporting_files/template_FlowCal.xlsx'
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
lpa = lpadesign.plate.LPA(name='Jennie')
lpa.cell_strain_name = 'Light sensing strain 1'
lpa.media_vol = 16000.
lpa.sample_vol = 500.
lpa.cell_setup_method = 'fixed_volume'
lpa.cell_predilution = 100
lpa.cell_predilution_vol = 1000
lpa.cell_shot_vol = 5
lpa.apply_inducer(inducer=light_520, apply_to='wells')
lpa.apply_inducer(inducer=light_660, apply_to='wells')
exp.add_plate(lpa)

exp.generate()