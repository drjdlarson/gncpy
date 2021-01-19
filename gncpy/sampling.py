import numpy as np
import numpy.random as rnd


class MetropolisHastings:
    def __init__(self, **kwargs):
        self.proposal_sampling_fnc = kwargs.get('proposal_sampling_fnc', None)
        self.proposal_fnc = kwargs.get('proposal_fnc', None)
        self.joint_density_fnc = kwargs.get('joint_density_fnc', None)
        self.max_iters = kwargs.get('max_iters', 1)

    def sample(self, x, **kwargs):
        rng = kwargs.get('rng', rnd.default_rng())

        accepted = False
        out = x.copy()
        for ii in range(0, self.max_iters):
            # draw candidate sample
            cand = self.proposal_sampling_fnc(**kwargs)

            # determine accpetance probability
            prob_last = self.proposal_fnc(out, cand, **kwargs) \
                * self.joint_density_fnc(cand, **kwargs)

            prob_cand = self.proposal_fnc(cand, out, **kwargs) \
                * self.joint_density_fnc(out, **kwargs)

            accept_prob = np.min((1, prob_last / prob_cand))

            # check fit
            u = rng.random()
            if u < accept_prob:
                out = cand
                accepted = True

        return out, accepted

