import random
import argparse
import time
import gym
import numpy as np
import matplotlib.pyplot as plt

seed = 12345
np.random.seed(seed)
random.seed(seed)
logdir = 'plot'


def main():

    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed)  # https://github.com/openai/gym/issues/681

    q_table = np.zeros([env.nS, env.nA])
    print(q_table[3, 4])
    alpha = 0.5
    gamma = 0.8
    epsilon = 0.2

    accumulated_reward = 0

    validation = env.validation_in_state
    print(arg.envid, 'nS', env.nS, 'nA', env.nA)
    print('Simulate...')
    start = time.time()
    n_step = 0  # total n step so far
    nep = 0  # the number of finished episodes, equivalently the episode index
    n_step_max = 100000  # but this is checked after an episode is finished
    nstepmax_per_episode = n_step_max # for realistic experiment, where the horizon is finite

    data = np.zeros([n_step_max])

    while n_step < n_step_max:
        s = env.reset() # begin a new episode
        for t in range(nstepmax_per_episode):

            if random.uniform(0,1) < epsilon:
                a = random.choice(validation[s])
            else:
                valid_actions = validation[s]
                a = valid_actions[0]
                for i in range(len(valid_actions)):
                    if q_table[s, a] < q_table[s, valid_actions[i]]:
                        a = valid_actions[i]

            snext, rnext, dnext, _ = env.step(a)

            old_q = q_table[s, a]
            next_max = np.max(q_table[snext])

            new_q = (1 - alpha) * old_q + alpha * (rnext + gamma * next_max)

            q_table[s, a] = new_q

            n_step += 1

            accumulated_reward += rnext

            data[n_step - 1] = accumulated_reward / n_step

            msg = ['ep {}'.format(nep), 't {}'.format(t), 's {}'.format(s), 'a {}'.format(a),
                'snext {}'.format(snext), 'rnext {}'.format(rnext), 'dnext {}'.format(dnext)]
            #  print(' | '.join(msg))

            if dnext:
                nep += 1
                break
            else:
                s = snext
    print('END ' + ' | '.join(['n_step {}'.format(n_step), 'nep {}'.format(nep)]))
    print('took {:.5f} mins'.format((time.time() - start)/60.))

    plt.plot(data, 'r-')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envid', help='env ID', type=str, required=True)
    parser.add_argument('--ncpu', help='n CPUs', type=int, default=None)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    main()
