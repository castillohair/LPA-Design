"""
Module that contains the LPA and LPAArray classes.

"""

import platedesign
import platedesign.plate

class LPA(platedesign.plate.Plate):
    """
    Object that represents a plate in an LPA.

    This class inherits from ``Plate`` in ``platedesign``. As such, it has
    the ability to manage all the chemical inducers in ``platedesign``. In
    addition, it can manage light inducers in ``lpadesign.inducers``. This
    class generates the replicate setup files of light inducers.

    """
    pass

class LPAArray(platedesign.plate.PlateArray):
    pass