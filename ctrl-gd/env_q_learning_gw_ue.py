import random
import argparse
import time
import gym
import numpy as np
import matplotlib.pyplot as plt

seed = 123
np.random.seed(seed)
random.seed(seed)
logdir = 'plot'


def main():

    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed)  # https://github.com/openai/gym/issues/681

    q_table = np.zeros([env.nS, env.nA])
    #  print(q_table[3, 4])
    alpha = 0.5
    gamma = 0.99
    p = 1
    c = 50

    accumulated_reward = 0

    validation = env.validation_in_state
    print(arg.envid, 'nS', env.nS, 'nA', env.nA)
    print('Simulate...')
    start = time.time()
    n_step = 0  # total n step so far
    nep = 0  # the number of finished episodes, equivalently the episode index
    n_step_max = 100000  # but this is checked after an episode is finished
    nstepmax_per_episode = n_step_max  # for realistic experiment, where the horizon is finite

    data = np.zeros([n_step_max])
    nf = np.ones((env.nS, env.nA))

    while n_step < n_step_max:

        s = env.reset()  # begin a new episode

        for t in range(nstepmax_per_episode):

            if random.uniform(0, 1) > p:
                a = random.choice(validation[s])

            else:
                valid_actions = validation[s]
                valid_r = []

                for i in range(len(valid_actions)):
                    valid_r.append(q_table[s, valid_actions[i]])

                #  max_r = np.max(valid_r)
                ue = []

                #  print(max_r)
                #  print(valid_r)

                for i in range(len(valid_r)):
                    ue.append(valid_r[i] + c / nf[s, valid_actions[i]])

                #  print(n_step)
                a = valid_actions[np.argmax(ue)]
                nf[s, a] += 1

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

    actions = np.zeros((20, 20))
    square = 0
    for i in range(20):
        for j in range(20):
            actions[i, j] = np.argmax(q_table[square, :])
            square += 1

    print(actions)

    #  plt.plot(data, 'r-')
    #  plt.ylabel('Average Reward')
    #  plt.xlabel('Steps')
    #  plt.show()

    return data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envid', help='env ID', type=str, required=True)
    parser.add_argument('--ncpu', help='n CPUs', type=int, default=None)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    data_one = main()
    data_two = main()
    data_three = main()
    data_four = main()
    data_five = main()

    plt.plot(data_one, 'r-')
    plt.plot(data_two, 'g-')
    plt.plot(data_three, 'c-')
    plt.plot(data_four, 'm-')
    plt.plot(data_five, 'b-')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    #  plt.axis((0, 500000, 0.95, 1.35))
    plt.show()

