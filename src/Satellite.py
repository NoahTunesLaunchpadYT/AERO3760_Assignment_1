from helper.coordinate_transforms import *

class Satellite:
    """Represents a satellite in orbit.
    """
    def __init__(self, state, start_jd=None):
        # state should be (r_km, v_km_s) as two length-3 arrays
        r = np.array(state[0], dtype=float)
        v = np.array(state[1], dtype=float)
        self.states = [ (r, v) ]
        self.start_jd = start_jd

    @classmethod
    def from_state_vector(cls, r_km, v_km_s):
        return cls((r_km, v_km_s))

    @classmethod
    def from_keplerian(cls, a_km, e, inc_deg, raan_deg, aop_deg, ta_deg, mu=MU_EARTH):
        r, v = kepler_to_rv(a_km, e, inc_deg, raan_deg, aop_deg, ta_deg, mu)
        return cls((r, v))

    @classmethod
    def from_peri_apo(cls, ta_deg, rp_km, ra_km, raan_deg, inc_deg, aop_deg, mu=MU_EARTH):
        a = 0.5 * (rp_km + ra_km)
        e = (ra_km - rp_km) / (ra_km + rp_km)
        r, v = kepler_to_rv(a, e, inc_deg, raan_deg, aop_deg, ta_deg, mu)
        return cls((r, v))

    def current_state(self):
        return self.states[-1]
    
    def get_orbital_period(self):
        a_km, e, inc_deg, raan_deg, aop_deg, ta_deg = self.keplerian_elements()
        return 2 * np.pi * np.sqrt(a_km**3 / MU_EARTH)

    def __repr__(self):
        r, v = self.current_state()
        return f"Satellite(states={len(self.states)}, r={r}, v={v})"

    def keplerian_elements(self, mu=MU_EARTH, degrees: bool = True):
        """
        Return the classical Keplerian elements from the current state vector.

        Returns
        -------
        (a_km, e, inc, raan, aop, ta)
            a_km : semi-major axis [km]
            e    : eccentricity [-]
            inc  : inclination [deg or rad]
            raan : right ascension of ascending node Ω [deg or rad]
            aop  : argument of perigee ω [deg or rad]
            ta   : true anomaly ta [deg or rad]
        """
        r, v = self.current_state()
        r = np.asarray(r, dtype=float)
        v = np.asarray(v, dtype=float)

        # magnitudes
        rmag = np.linalg.norm(r)
        vmag = np.linalg.norm(v)

        # specific angular momentum
        h = np.cross(r, v)
        hmag = np.linalg.norm(h)

        # node vector
        k_hat = np.array([0.0, 0.0, 1.0])
        n = np.cross(k_hat, h)
        nmag = np.linalg.norm(n)

        # eccentricity vector
        e_vec = (np.cross(v, h) / mu) - (r / rmag)
        e = np.linalg.norm(e_vec)

        # specific orbital energy
        energy = vmag**2 / 2.0 - mu / rmag

        # semi-major axis (parabolic => a = inf)
        if np.isclose(e, 1.0, atol=1e-10):
            a = np.inf
        else:
            a = -mu / (2.0 * energy)

        # inclination
        inc = np.arccos(np.clip(h[2] / hmag, -1.0, 1.0))

        # tolerances for singular cases
        eps_e = 1e-10
        eps_n = 1e-12

        # RAAN Ω
        if nmag > eps_n:
            raan = np.arctan2(n[1], n[0])  # (-π, π]
            if raan < 0:
                raan += 2.0 * np.pi
        else:
            # equatorial: Ω undefined
            raan = 0.0

        # argument of perigee ω and true anomaly ν
        if e > eps_e and nmag > eps_n:
            # general case
            # ω = angle between n and e_vec
            cos_aop = np.clip(np.dot(n, e_vec) / (nmag * e), -1.0, 1.0)
            aop = np.arccos(cos_aop)
            if e_vec[2] < 0:
                aop = 2.0 * np.pi - aop

            # ν = angle between e_vec and r
            cos_ta = np.clip(np.dot(e_vec, r) / (e * rmag), -1.0, 1.0)
            # sign via (r·v)
            ta = np.arccos(cos_ta)
            if np.dot(r, v) < 0:
                ta = 2.0 * np.pi - ta

        elif e <= eps_e and nmag > eps_n:
            # circular inclined: ω undefined; use argument of latitude u instead
            # u = angle between n and r
            cos_u = np.clip(np.dot(n, r) / (nmag * rmag), -1.0, 1.0)
            u = np.arccos(cos_u)
            if r[2] < 0:
                u = 2.0 * np.pi - u
            aop = 0.0
            ta = u

        elif e > eps_e and nmag <= eps_n:
            # equatorial, non-circular: Ω undefined; use longitude of periapsis ϖ instead
            # ϖ = angle from +X to e_vec in equatorial plane
            varpi = np.arctan2(e_vec[1], e_vec[0])
            if varpi < 0:
                varpi += 2.0 * np.pi
            raan = 0.0
            aop = varpi  # store as ω with Ω set to 0

            # ν from e_vec to r
            cos_ta = np.clip(np.dot(e_vec, r) / (e * rmag), -1.0, 1.0)
            ta = np.arccos(cos_ta)
            if np.dot(r, v) < 0:
                ta = 2.0 * np.pi - ta

        else:
            # circular equatorial: both Ω and ω undefined.
            # Use true longitude ℓ = angle from +X to r in equatorial plane.
            raan = 0.0
            aop = 0.0
            ta = np.arctan2(r[1], r[0])
            if ta < 0:
                ta += 2.0 * np.pi

        if degrees:
            inc_deg = np.degrees(inc)
            raan_deg = np.degrees(raan)
            raan_deg = (raan_deg + 180)%360 - 180
            aop_deg = np.degrees(aop)
            ta_deg = np.degrees(ta)
            return a, e, inc_deg, raan_deg, aop_deg, ta_deg
        else:
            return a, e, inc, raan, aop, ta