import gym
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from IPython.display import clear_output

env= gym.make("FrozenLake-v0")

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

Q_table = np.zeros((state_space_size, action_space_size))
# Q_table = np.array([[0.38476456,0.28895145, 0.2812079 ,  0.28734744],
#  [0.12983548 ,0.13169754 ,0.12074723, 0.25108776],
#  [0.27496168, 0.1127739,  0.10543775 ,0.10019571],
#  [0.05434957, 0.,         0.,         0.        ],
#  [0.41673245 ,0.19242203, 0.24120781, 0.24609419],
#  [0.,        0.,         0. ,        0.        ],
#  [0.04044708, 0.04064202, 0.28272311, 0.01272921],
#  [0.,         0.,         0.,         0.        ],
#  [0.18879314, 0.2886415,  0.31637049, 0.50328506],
#  [0.29933553, 0.61171204, 0.26461813, 0.3343198 ],
#  [0.55849537, 0.24675854, 0.25048084, 0.11779078],
#  [0.,         0.,         0.,         0.        ],
#  [0.,         0.,         0.,         0.        ],
#  [0.29198518, 0.25804139, 0.6865779,  0.33463239],
#  [0.55267149, 0.86207137, 0.69407157, 0.60331011],
#  [0.,         0.,         0.,         0.        ]])
print(Q_table)

num_episodes = 10000
max_steps_per_episode = 100

learning_rate = 0.01
discount_rate = 0.99

exploration_rate = .1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

all_rewards_episodes = []

for episode in range(num_episodes):
    state = env.reset()
    # print("******Episode",episode+1,"******")
    # time.sleep(1)
    done = False
    reward_in_this_episode = 0

    for step in range(max_steps_per_episode):
        # clear_output(wait=True)
        # env.render()
        # time.sleep(0.3)
        # Exploration or exploitation
        exploration_rate_threshold = random.uniform(0,1)

        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(Q_table[state, :])


        else:
            action = env.action_space.sample()

        new_state, reward , done, info = env.step(action)
        # reward = reward if not done else -0.1

        # Updating the Q Table

        Q_table[state, action] = Q_table[state, action]*(1 - learning_rate) + learning_rate * (reward + (discount_rate * np.max(Q_table[new_state, :])))

        state = new_state
        reward_in_this_episode += reward

        if done == True:
            # clear_output(wait=True)
            # env.render()
            # if reward == 1:
            #     print("*****Goal is reached*****")
            #     time.sleep(0.3)
            # else:
            #     print("*****Fell into Hole*****")
            #     time.sleep(0.3)
            #     clear_output(wait = True)
            break


    # Exploration Decay
    exploration_rate = min_exploration_rate + ( max_exploration_rate - min_exploration_rate) * np.exp(-exploration_rate* episode)

    all_rewards_episodes.append(reward_in_this_episode)


# env.close()
reward_per_thousand_episode = np.split(np.array(all_rewards_episodes),num_episodes/1000)
count = 1000
x=[]

print("\n_____Reward per thousand Episode_____\n")
for r in reward_per_thousand_episode :
    print(count," --> ",str(sum(r/1000)))
    x.append(sum(r/1000))


    count += 1000


print("\n****** Final Q Table *******\n")
print(Q_table)

plt.plot(x)
plt.show()





