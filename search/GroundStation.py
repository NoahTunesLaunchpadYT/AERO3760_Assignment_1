import numpy as np

# ----------------- Earth / time helpers -----------------
R_e = 6378.137                # Radius at the equator (km)
R_p = 6356.7523               # Polar radius (km)
f = 0.003352813               # Flattening

def geodetic_to_ecef(lat_deg, lon_deg, h_m):
    """geodetic -> ECEF (km)."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    h_km = h_m * 1e-3
    sinp, cosp = np.sin(lat), np.cos(lat)
    R_φ = R_e / np.sqrt(1.0 - (2.0*f - f**2) * sinp*sinp)

    R_c = R_φ + h_km
    R_s = (1-f)**2 * R_φ + h_km

    x = (R_c) * cosp * np.cos(lon)
    y = (R_c) * cosp * np.sin(lon)
    z = R_s * sinp
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