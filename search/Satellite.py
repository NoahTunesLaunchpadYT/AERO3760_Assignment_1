from helper_coordinate_transforms import *

class Satellite:
    def __init__(self, state):
        # state should be (r_km, v_km_s) as two length-3 arrays
        r = np.array(state[0], dtype=float)
        v = np.array(state[1], dtype=float)
        self.states = [ (r, v) ]

    @classmethod
    def from_state_vector(cls, r_km, v_km_s):
        return cls((r_km, v_km_s))

    @classmethod
    def from_keplerian(cls, a_km, e, inc_deg, raan_deg, aop_deg, tru_deg, mu=MU_EARTH):
        r, v = kepler_to_rv(a_km, e, inc_deg, raan_deg, aop_deg, tru_deg, mu)
        return cls((r, v))

    @classmethod
    def from_peri_apo(cls, tru_deg, rp_km, ra_km, raan_deg, inc_deg, aop_deg, mu=MU_EARTH):
        a = 0.5 * (rp_km + ra_km)
        e = (ra_km - rp_km) / (ra_km + rp_km)
        r, v = kepler_to_rv(a, e, inc_deg, raan_deg, aop_deg, tru_deg, mu)
        return cls((r, v))

    def add_state(self, r_km, v_km_s):
        r = np.array(r_km, dtype=float)
        v = np.array(v_km_s, dtype=float)
        self.states.append((r, v))

    def current_state(self):
        return self.states[-1]

    def __repr__(self):
        r, v = self.current_state()
        return f"Satellite(states={len(self.states)}, r={r}, v={v})"
