import random
import argparse
import time
import gym
import numpy as np
import matplotlib.pyplot as plt

seed = 999
np.random.seed(seed)
random.seed(seed)
logdir = 'plot'


def q_main(gamma):

    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed)

    q_table = np.zeros([env.nS, env.nA])

    alpha = 0.5
    epsilon = 0.1

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

    return data


def r_main():

    arg = parse_args()
    env = gym.make(arg.envid)
    env.seed(seed)
    env.action_space.np_random.seed(seed)  # https://github.com/openai/gym/issues/681

    r_table = np.zeros([env.nS, env.nA])
    alpha = 0.2
    beta = 0.2
    epsilon = 0.1
    rho = 0
    accumulated_reward = 0

    validation = env.validation_in_state
    print(arg.envid, 'nS', env.nS, 'nA', env.nA)
    print('Simulate...')
    start = time.time()
    n_step = 0  # total n step so far
    nep = 0  # the number of finished episodes, equivalently the episode index
    n_step_max = 100000  # but this is checked after an episode is finished
    n_step_max_per_episode = n_step_max  # for realistic experiment, where the horizon is finite

    data = np.zeros([n_step_max])

    while n_step < n_step_max:

        s = env.reset()  # begin a new episode

        for t in range(n_step_max_per_episode):

            explore = False

            if random.uniform(0, 1) < epsilon:
                a = random.choice(validation[s])
                explore = True
            else:

                valid_actions = validation[s]
                valid_r = []

                for i in range(len(valid_actions)):
                    valid_r.append(r_table[s, valid_actions[i]])

                max_r = np.max(valid_r)
                max_actions = []

                #  print(max_r)
                #  print(valid_r)

                for i in range(len(valid_r)):
                    if valid_r[i] == max_r:
                        max_actions.append(valid_actions[i])

                #  print(n_step)
                a = random.choice(max_actions)

            snext, rnext, dnext, _ = env.step(a)

            old_r = r_table[s, a]

            valid_r_next = []
            valid_actions_next = validation[snext]

            for i in range(len(valid_actions_next)):
                valid_r_next.append(r_table[snext, valid_actions_next[i]])

            next_max = np.max(valid_r_next)

            new_r = (1 - beta) * old_r + beta * (rnext - rho + next_max)
            #  print(new_r)

            if not explore:
                rho = ((1 - alpha) * rho) + (alpha * (rnext + next_max - r_table[s, a]))

            r_table[s, a] = new_r

            #  print(s, a)

            n_step += 1
            accumulated_reward += rnext

            data[n_step - 1] = accumulated_reward/n_step

            #  print(accumulated_reward/n_step)

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
    for i in range(1):             #env.nS):
        print(i)

    return data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--envid', help='env ID', type=str, required=True)
    parser.add_argument('--ncpu', help='n CPUs', type=int, default=None)
    arg = parser.parse_args()
    return arg


if __name__ == '__main__':
    data_q_one = q_main(0.8)
    data_q_two = q_main(0.9)
    data_q_three = q_main(0.95)
    data_q_four = q_main(0.98)
    data_r = r_main()

    plt.plot(data_q_one, 'r-', label='Q-Learning: gamma=0.8')
    plt.plot(data_q_two, 'g-', label='Q-Learning: gamma=0.9')
    plt.plot(data_q_three, 'c-', label='Q-Learning: gamma=0.95')
    plt.plot(data_q_four, 'm-', label='Q-Learning: gamma=0.98')
    plt.plot(data_r, 'b-', label='R-Learning')
    plt.ylabel('Average Reward')
    plt.xlabel('Steps')
    plt.axis((0, 100000, 0.95, 1.35))
    plt.title('Multi-Loop: R-Learning vs Q-Learning')
    plt.legend()
    plt.show()

    
