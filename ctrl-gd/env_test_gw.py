import random
import argparse
import time
import gym
import numpy as np

seed = 1234
np.random.seed(seed)
random.seed(seed)
logdir = 'plot'

def main():
    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed)  # https://github.com/openai/gym/issues/681
    validation = env.validation_in_state
    print(arg.envid, 'nS', env.nS, 'nA', env.nA)
    print('Simulate...')
    start = time.time()
    nstep = 0  # total n step so far
    nep = 0  # the number of finished episodes, equivalently the episode index
    nstepmax = 1000  # but this is checked after an episode is finished
    nstepmax_per_episode = nstepmax # for realistic experiment, where the horizon is finite

    while nstep < nstepmax:
        s = env.reset() # begin a new episode
        for t in range(nstepmax_per_episode):

            a = random.choice(validation[s])

            snext, rnext, dnext, _ = env.step(a)
            nstep += 1

            msg = ['ep {}'.format(nep), 't {}'.format(t), 's {}'.format(s), 'a {}'.format(a),
                'snext {}'.format(snext), 'rnext {}'.format(rnext), 'dnext {}'.format(dnext)]
            print(' | '.join(msg))

            if dnext:
                nep += 1
                break
            else:
                s = snext
    print('END ' + ' | '.join(['nstep {}'.format(nstep), 'nep {}'.format(nep)]))
    print('took {:.5f} mins'.format((time.time() - start)/60.))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envid', help='env ID', type=str, required=True)
    parser.add_argument('--ncpu', help='n CPUs', type=int, default=None)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    main()
