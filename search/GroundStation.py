import numpy as np

# ----------------- Earth / time helpers -----------------
A_WGS84 = 6378.137            # km
F_WGS84 = 1.0 / 298.257223563
E2_WGS84 = F_WGS84 * (2.0 - F_WGS84)  # first eccentricity^2

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """WGS-84 geodetic -> ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h_km = h_m * 1e-3
    sinp, cosp = np.sin(lat), np.cos(lat)
    N = A_WGS84 / np.sqrt(1.0 - E2_WGS84 * sinp*sinp)
    x = (N + h_km) * cosp * np.cos(lon)
    y = (N + h_km) * cosp * np.sin(lon)
    z = (N * (1.0 - E2_WGS84) + h_km) * sinp
    return np.array([x, y, z])

def enu_matrix(lat_deg, lon_deg):
    """ECEF -> ENU rotation at site (rows are E,N,U unit vectors)."""
    lat = np.radians(lat_deg); lon = np.radians(lon_deg)
    sinp, cosp = np.sin(lat), np.cos(lat)
    sinl, cosl = np.sin(lon), np.cos(lon)
    E = np.array([-sinl,            cosl,           0.0])
    N = np.array([-sinp*cosl, -sinp*sinl,     cosp])
    U = np.array([ cosp*cosl,  cosp*sinl,     sinp])  # surface-normal (Up) for geodetic lat
    return np.vstack((E, N, U))

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