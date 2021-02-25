import pyemma as pm


class Dynamics(object):

    def __init__(self, trans_mat):
        self.model = pm.msm.MSM(trans_mat)

    def sample(self, starting_states, traj_lengths):
        return []


