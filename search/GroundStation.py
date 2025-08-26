import numpy as np
from helper.constants import *
from helper.coordinate_transforms import geodetic_to_ecef, enu_matrix

# ----------------- GroundStation -----------------
class GroundStation:
    def __init__(self, lat_deg, lon_deg, h_m=0.0):
        self.lat_deg = float(lat_deg)
        self.lon_deg = float(lon_deg)
        self.h_m = float(h_m)
        self._ecef = geodetic_to_ecef(self.lat_deg, self.lon_deg, self.h_m)  # km
        self._enu  = enu_matrix(self.lat_deg, self.lon_deg)                  # 3x3

        # Up vector in ECEF (unit), equals ENU's third row
        self._up_ecef = self._enu[2]