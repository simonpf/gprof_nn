from pathlib import Path

import numpy as np

class Equalizer():
    def __init__(self, sensor):

        data_path = Path(__file__).parent / "data"
        self.percs_obs = np.load(
                data_path / f"{sensor.name.lower()}_percs_obs.npz"
        )["percs"]
        self.percs_train = np.load(
            data_path / f"{sensor.name.lower()}_percs_train.npz"
        )["percs"]
        self.biases = self.percs_obs - self.percs_train

        self.n_freqs = sensor.n_freqs
        self.n_angles = sensor.n_angles

        angles = sensor.angles
        self.angle_bins = np.zeros(angles.size + 1)
        self.angle_bins[1:-1] = 0.5 * (angles[1:] + angles[:-1])
        self.angle_bins[0] = 2.0 * self.angle_bins[1] - self.angle_bins[2]
        self.angle_bins[-1] = 2.0 * self.angle_bins[-2] - self.angle_bins[-3]

    def __call__(self, tbs, eia, surface_type):

        tbs_c = tbs.copy()

        for i in range(18):
            for j in range(self.n_freqs):
                for k in range(self.n_angles):
                    ang_l = self.angle_bins[k + 1]
                    ang_r = self.angle_bins[k]
                    inds = (np.abs(eia) >= ang_l) * (np.abs(eia) < ang_r) * (surface_type == i + 1)
                    t_inds = np.digitize(tbs_c[inds, j], self.percs_train[i, j, k, 1:-1])
                    tbs_c[inds, j] += self.biases[i, j, k, t_inds]

        return tbs_c



