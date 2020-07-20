import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class MultiLoopEnv(gym.Env):
    def __init__(self):
        n_center = 1
        n_loop_one = 2
        n_loop_two = 4
        n_loop_three = 6
        n_loop_four = 8
        self.center = center = 0  # `center` is @idx=0, followed by loops 1 to 4
        loop_one = center + n_loop_one  # center, loop-one
        loop_two = center + n_loop_one + n_loop_two  # center, loop-one, loop-two
        loop_three = center + n_loop_one + n_loop_two + n_loop_three  # center, loop-one, loop-two, loop-three
        loop_four = center + n_loop_one + n_loop_two + n_loop_three + n_loop_four  # center, ..., loop-four
        next_to_center_one = center + 1  # center, next_to_center_one, ...
        next_to_center_two = loop_one + 1  # ..., loop-one, next_to_center_two, ...
        next_to_center_three = loop_two + 1  # ..., loop-two, next_to_center_three, ...
        next_tocenter_four = loop_three + 1  # ..., loop-three, next_to_center_two, ...
        self.nS = nS = n_center + n_loop_one + n_loop_two + n_loop_three + n_loop_four
        self.observation_space = spaces.Discrete(nS)  # by convention, name `observation_`, instead of `state_`
        self.state = center  # initial state

        fwd = 0
        one = 1
        two = 2
        three = 3
        four = 4
        self.nA = nA = 5
        self.action_space = spaces.Discrete(nA)  # NOTE: not every action is available in every state
        validation_in_state = []
        for s in range(nS):
            validation_in_state.append([one, two, three, four] if s == center else [fwd])
        self.validation_in_state = validation_in_state

        zero_rew = 0
        OneRew = 3
        TwoRew = 6
        ThreeRew = 9
        FourRew = 12
        self.rewlist = rewlist = [zero_rew, OneRew, TwoRew, ThreeRew, FourRew]
        self.nR = nR = len(self.rewlist)

        P = np.zeros((nA, nS, nS * nR))  # the last dim is (nS*nR) to emphasize the joint probability
        s_dimth, a_dimth, srnext_dimth = 0, 1, 2
        for s in range(nS):
            for a in range(nA):
                if (a in validation_in_state[s]):
                    if a == one:  # only valid in center
                        snext = next_to_center_one
                        rnext = rewlist.index(zero_rew)
                    elif a == two:  # only valid in center
                        snext = next_to_center_two
                        rnext = rewlist.index(zero_rew)
                    elif a == three:  # only valid in center
                        snext = next_to_center_three
                        rnext = rewlist.index(zero_rew)
                    elif a == four:  # only valid in center
                        snext = next_tocenter_four
                        rnext = rewlist.index(zero_rew)
                    else:  # fwd, valid in allrooms except in center
                        if s == loop_one:
                            snext = center
                            rnext = rewlist.index(OneRew)
                        elif s == loop_two:
                            snext = center
                            rnext = rewlist.index(TwoRew)
                        elif s == loop_three:
                            snext = center
                            rnext = rewlist.index(ThreeRew)
                        elif s == loop_four:
                            snext = center
                            rnext = rewlist.index(FourRew)
                        else:  # states other than (printer, mail, and center)
                            snext = s + 1
                            rnext = rewlist.index(zero_rew)
                    srnext = np.ravel_multi_index((snext, rnext), (nS, nR), order='C')
                    P[a, s, srnext] = 1.  # deterministic transition dynamics
                else:
                    P[a, s, :] = [np.nan] * (nS * nR)
        self.nextstate_nextreward_probdistrib = self.P = P
        Pmasked = np.ma.masked_invalid(P)
        assert np.allclose(Pmasked.sum(axis=srnext_dimth, keepdims=True), np.ones_like(P))

        self.initialstate_probdistrib = np.zeros(nS)
        self.initialstate_probdistrib[center] = 1.

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.center
        return self.state

    def step(self, action):
        if action not in self.validation_in_state[self.state]:
            raise RuntimeError('Invalid action={} in state={}'.format(action, self.state))
        srnext = np.argmax(np.random.multinomial(n=1, pvals=self.P[action, self.state, :]))
        snext, rnext = np.unravel_index(srnext, (self.nS, self.nR), order='C')
        self.state = snext
        reward = self.rewlist[rnext]
        done = False;
        info = {}
        return self.state, reward, done, info
