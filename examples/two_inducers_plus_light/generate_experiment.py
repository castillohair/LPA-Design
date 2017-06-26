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

# Light sensing strain will use a 2x2 plate array. It will have xylose and
# alternating red and green light across 12 columns, and iptg across 8 rows.
# Shuffling of light inducers should be synchronized with xylose.

iptg = platedesign.inducer.ChemicalInducer(name='IPTG', units='uM')
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
                                           id_prefix='G')
light_520.intensity = numpy.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])*50
light_520.sync_shuffling(xyl)
exp.add_inducer(light_520)

# 660nm (red) light: constant throughout all wells
light_660 = lpadesign.inducer.LightInducer(name='660nm Light',
                                           led_layout='660-LS',
                                           id_prefix='R')
light_660.intensity = numpy.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0])*20
light_660.sync_shuffling(xyl)
exp.add_inducer(light_660)

# LPA array for light-sensing strain
lpaarray = lpadesign.plate.LPAArray(
    'PA1',
    array_n_rows=2,
    array_n_cols=2,
    lpa_names=['Jennie', 'Picard', 'Kirk', 'Shannen'],
    lpa_n_rows=4,
    lpa_n_cols=6)
lpaarray.cell_strain_name = 'Light-Sensing Strain'
lpaarray.media_vol = 16000.*4
lpaarray.apply_inducer(inducer=xyl, apply_to='rows')
lpaarray.apply_inducer(inducer=iptg, apply_to='cols')
exp.add_plate(lpaarray)

# Plate for autofluorescence control strain
plate = platedesign.plate.Plate('P5', n_rows=4, n_cols=6)
plate.cell_strain_name = 'Autofluorescence Control Strain'
plate.samples_to_measure = 4
plate.media_vol = 16000.
exp.add_plate(plate)

# Add common settings to plates
for plate in exp.plates:
    plate.sample_vol = 500.
    plate.cell_setup_method = 'fixed_od600'
    plate.cell_predilution = 100
    plate.cell_predilution_vol = 1000
    plate.cell_initial_od600 = 1e-5

exp.generate()
