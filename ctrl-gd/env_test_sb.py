#!/usr/bin/env python3
# ./env_test.py --envid gym_gridnav:GridNavSquare5f-v0 --ncpu 3
import random, argparse, time
import gym, numpy as np
import gym_util as gu
from gym_util.pnsr_util import get_Pnsr_dict, get_Pnsr_mat

seed = 12345
np.random.seed(seed); random.seed(seed)
logdir = 'plot'

def main():
    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed) # https://github.com/openai/gym/issues/681
    print(arg.envid, 'nS', env.nS, 'nA', env.nA)
    # print(env.__dict__.keys())

    # for s in range(env.nS):
    #     print('')
    #     for a in range(env.nA):
    #         print('s {} a {}: {:.3}'.format(s, env.action_list[a], env.get_reward(s, a)))

    # aidx = np.random.choice(np.arange(env.nA), size=(env.nS,), p=None)
    # pi = np.zeros((env.nS, env.nA))
    # pi[(np.arange(env.nS), aidx)] = 1 # a deterministic policy
    # # try:
    # #     env.plot_policy(pi, logdir, tag='__'.join(['uniform-random-policy', arg.envid]))
    # # except:
    # #     pass
    # env.plot_transition(pi, logdir, tag='__'.join(['uniform-random-policy', arg.envid]))
    # exit()

    # print('Building Pnsr...'); start = time.time()
    # Pnsr_dict = get_Pnsr_dict(env, arg.ncpu)
    # print('len(Pnsr_dict) {:,}'.format(len(Pnsr_dict)))
    # Pnsr = get_Pnsr_mat(env)
    # print('took {:.5f} mins'.format((time.time() - start)/60.))
    # print('Pnsr.shape', Pnsr.shape)
    # print('Pnsr.size() {:,}'.format(Pnsr.size))
    # assert len(Pnsr_dict)>0
    # exit('exit Building Pnsr...')

    print('Simulate...'); start = time.time()
    nstep = 0 # total n step so far
    nep = 0 # the number of finished episodes, equivalently the episode index
    nstepmax = 1000 # but this is checked after an episode is finished
    nstepmax_per_episode = nstepmax # for realistic experiment, where the horizon is finite

    while nstep <= nstepmax:
        s = env.reset() # begin a new episode
        for t in range(nstepmax_per_episode):
            if s != 0:
                a = 0
            else:
                a = random.randint(1, 2)

            snext, rnext, dnext, _ = env.step(a)
            nstep += 1

            msg = ['ep {}'.format(nep), 't {}'.format(t), 's {}'.format(s), 'a {}'.format(a),
                'snext {}'.format(snext), 'rnext {}'.format(rnext), 'dnext {}'.format(dnext)]
            print(' | '.join(msg))
            # env.plot_state(s, a, logdir, tag='_'.join([str(t).zfill(3), str(nep).zfill(3)]))
            # exit()

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
