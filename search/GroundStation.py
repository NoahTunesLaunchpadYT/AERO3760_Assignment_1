import numpy as np
import matplotlib.pyplot as plt

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

def gmst_from_jd(jd):
    """Greenwich Mean Sidereal Angle (radians) from Julian Date."""
    jd = np.asarray(jd, float)
    T = (jd - 2451545.0) / 36525.0
    theta_deg = 280.46061837 + 360.98564736629*(jd - 2451545.0) + 0.000387933*T*T - (T**3)/38710000.0
    return np.radians(theta_deg % 360.0)

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

    def eci_at(self, jd):
        """
        Station position in ECI (km) and Up vector in ECI (unit),
        both at Julian Date(s) jd.
        """
        theta = gmst_from_jd(jd)
        ct, st = np.cos(theta), np.sin(theta)

        # Rotate ECEF -> ECI by Rz(theta)
        R = self._ecef
        if np.ndim(theta) == 0:
            Rz = np.array([[ct, -st, 0.0],
                           [st,  ct, 0.0],
                           [0.0, 0.0, 1.0]])
            r_eci = Rz @ R
            up_eci = Rz @ self._up_ecef
            return r_eci, up_eci
        else:
            x_eci =  ct*R[0] - st*R[1]
            y_eci =  st*R[0] + ct*R[1]
            z_eci =  np.full_like(theta, R[2], dtype=float)
            r_eci = np.vstack((x_eci, y_eci, z_eci)).T

            ux =  ct*self._up_ecef[0] - st*self._up_ecef[1]
            uy =  st*self._up_ecef[0] + ct*self._up_ecef[1]
            uz =  np.full_like(theta, self._up_ecef[2], dtype=float)
            up_eci = np.vstack((ux, uy, uz)).T
            # normalise numerically just in case
            up_eci = up_eci / np.linalg.norm(up_eci, axis=1, keepdims=True)
            return r_eci, up_eci

    def elevation_from_los(self, los_eci, jd):
        """
        Elevation angle from line-of-sight vector(s) in ECI at times jd.
        Uses ENU at site (rotated to time-varying ECI).
        """
        # Build E,N,U in ECI by rotating ECEF basis rows with GMST
        theta = gmst_from_jd(jd)
        ct, st = np.cos(theta), np.sin(theta)
        # Rotate each basis row (E,N,U) from ECEF to ECI
        ENU = self._enu
        def rot_row(row):
            x =  ct*row[0] - st*row[1]
            y =  st*row[0] + ct*row[1]
            z =  row[2]
            return np.array([x, y, z])

        E_eci = rot_row(ENU[0])
        N_eci = rot_row(ENU[1])
        U_eci = rot_row(ENU[2])

        # If jd is array: broadcast rows
        if np.ndim(theta) != 0:
            E_eci = np.tile(E_eci, (los_eci.shape[0],1))
            N_eci = np.tile(N_eci, (los_eci.shape[0],1))
            U_eci = np.tile(U_eci, (los_eci.shape[0],1))

        # Components in ENU
        e_comp = np.sum(los_eci * E_eci, axis=-1)
        n_comp = np.sum(los_eci * N_eci, axis=-1)
        u_comp = np.sum(los_eci * U_eci, axis=-1)
        elev = np.arctan2(u_comp, np.sqrt(e_comp*e_comp + n_comp*n_comp))
        return elev  # radians

    def plot_celestial_track(self, traj, min_el_deg=None, show_horizon=True, ra_wrap='auto'):
        """
        Plot apparent track on the celestial sphere as seen from this station.
        - traj.JD must be absolute Julian Dates, traj.R in ECI km
        - If min_el_deg is set, only plot samples with elevation >= that
        - show_horizon: draws a simple horizon band (Dec of zenith - 90° .. +90° w.r.t. Up)
        - ra_wrap: 'auto', '0_360', or '-180_180'
        """
        JD = np.asarray(traj.JD)
        R_sat = np.asarray(traj.R)

        # Station ECI and LOS vectors
        r_site_eci, up_eci = self.eci_at(JD)  # (n,3) and (n,3)
        rho = R_sat - r_site_eci
        uhat = rho / np.linalg.norm(rho, axis=1, keepdims=True)  # unit LOS

        # Equatorial coordinates (ICRF-like): Dec from z, RA from x,y
        dec = np.degrees(np.arcsin(uhat[:,2]))
        ra  = np.degrees(np.arctan2(uhat[:,1], uhat[:,0]))  # [-180,180]
        if ra_wrap == '0_360':
            ra = (ra + 360.0) % 360.0
        elif ra_wrap == '-180_180':
            ra = ((ra + 180.0) % 360.0) - 180.0
        else:
            # default: 0..360
            ra = (ra + 360.0) % 360.0

        mask = np.ones(ra.shape, dtype=bool)
        if min_el_deg is not None:
            elev = np.degrees(self.elevation_from_los(uhat, JD))
            mask &= (elev >= float(min_el_deg))

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.scatter(ra[mask], dec[mask], s=3, alpha=0.75)

        # Optional horizon indicator: compute where U (zenith) lies and mark dec=0 visibility band?
        if show_horizon:
            # The celestial horizon (Dec as seen from site) isn’t a constant Dec line,
            # but we can show zenith track for context.
            zenith_dec = np.degrees(np.arcsin(up_eci[:,2]))
            zenith_ra  = np.degrees(np.arctan2(up_eci[:,1], up_eci[:,0]))
            zenith_ra  = (zenith_ra + 360.0) % 360.0
            ax.plot(zenith_ra, zenith_dec, lw=1.0, alpha=0.5, label='Zenith track')

        ax.set_xlabel('Right Ascension [deg]')
        ax.set_ylabel('Declination [deg]')
        ax.set_ylim([-90, 90])
        ax.set_xlim([360, 0])  # RA increases to the left (astronomy convention)
        ax.grid(True, alpha=0.4)
        title = f'Celestial track from lat {self.lat_deg:.2f}°, lon {self.lon_deg:.3f}°'
        if min_el_deg is not None:
            title += f' (elev ≥ {min_el_deg:.1f}°)'
        ax.set_title(title)
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.show()
