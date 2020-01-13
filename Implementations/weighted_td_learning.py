import gym
import torch
from dqn import DQN, ReplayBuffer
from torch.optim import Adam
from torch.nn.functional import mse_loss

# In[]:

env = gym.make('CartPole-v1')

# In[]:


def run_current_policy(env, policy):
    cur_state = env.reset()
    cur_state = torch.Tensor(cur_state)
    total_step = 0
    total_reward = 0.0
    done = False
    while not done:
        action = policy.select_action(cur_state, 0)
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)
        total_reward += reward
        env.render(mode='rgb_array')
        total_step += 1
        cur_state = next_state
    print("Total timesteps = {}, total reward = {}".format(total_step, total_reward))

# In[]:

gamma = 0.95
learning_rate_unweighted = 0.001
learning_rate_weighted = 1e-3
weight_decay = 10

avg_history = {'episodes': [], 'timesteps_unweighted': [], 'timesteps_weighted': [],
               'unweighted_reward': [], 'weighted_reward':[],
               'loss_unweighted': [], 'loss_weighted': []}
agg_interval = 10

# initialize policy and replay buffer
policy = DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=24)

# TODO : Check if this can perform better in a smaller no. of hidden sizes
policy_weighted = DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=24)
optimizer_weighted = Adam(policy_weighted.parameters(), lr=learning_rate_weighted)

replay_buffer = ReplayBuffer()
replay_buffer_weighted = ReplayBuffer()

optimizer = Adam(policy.parameters(), lr=learning_rate_unweighted)
start_episode = 0

# Play with a random policy and see
# run_current_policy(env.env, policy)

train_episodes = 2000

# In[]:


# Update the policy by a TD(0)
def update_policy(cur_states, actions, next_states, rewards, dones):
    # target doesnt change when its terminal, thus multiply with (1-done)
    # target = R(st-1, at-1) + gamma * max(a') Q(st, a')
    targets = rewards + torch.mul(1 - dones, gamma * policy(next_states).max(dim=1).values)
    # expanded_targets are the Q values of all the actions for the current_states sampled
    # from the previous experience. These are the predictions
    expanded_targets = policy(cur_states)[range(actions.shape[0]), actions.squeeze(-1).tolist()].reshape(-1,)
    optimizer.zero_grad()
    loss = mse_loss(input=targets.detach(), target=expanded_targets)
    loss.backward()
    optimizer.step()
    return loss.item()


def update_weighted_policy(cur_state, next_state, reward, action, next_states, power):
    """
    performs a weighted td update by sampling states and weighing them as per the angle between
     the cur_state and the sampled states
    :param cur_state: the current state
    :param next_state: the next state
    :param reward: the reward for cur_State -> next_state transition
    :param cur_states: the sampled states from the replay_buffer_weighted
    :return: the loss computed
    """
    # append the next_state to the sampled next_states
    next_states = torch.cat([next_states, next_state.reshape((1, -1))])

    # get the weights of the sampled neighbors
    weights_sampled_neighbors = policy_weighted.layer3(policy_weighted.layer2(policy_weighted.layer1(next_states)))
    self_weight = policy_weighted.layer3(policy_weighted.layer2(policy_weighted.layer1(next_state)))

    # get the angle between next_state and the sampled next_states
    weights = torch.mm(weights_sampled_neighbors, self_weight.reshape((-1, 1)))
    # divide by the l2 norm of weights_sampled_neighbors and self_weight
    denominator = self_weight.norm(p=2) * weights_sampled_neighbors.norm(p=2, dim=1)
    weights = weights.squeeze(-1) / denominator

    # todo : sanity check this equation
    weights = torch.pow(weights, power)

    # todo : check the weights are being almost the same, is this the case for all iterations or only here?
    weights = weights.softmax(dim=0).detach()
    # TODO : Need to raise weights as per the power of no. of iterations

    # construct the target
    target = reward + gamma * torch.dot(weights, policy_weighted(next_states).max(dim=1).values)
    predicted = policy_weighted(cur_state)[action.item()]

    # the implementation is (input-target)^2
    optimizer_weighted.zero_grad()
    # todo : this is a stochastic update, try updatinf this in batches
    loss = mse_loss(input=target.detach(), target=predicted)
    loss.backward()
    optimizer_weighted.step()
    return loss.item()

# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):

    episode_timestep_unweighted = 0.1
    episode_timestep_weighted = 0
    episode_unweighted_reward = 0.0
    episode_weighted_reward = 0
    loss1_cumulative = 0
    loss2_cumulative = 0

    # # play with unweighted standard policy
    # done = False
    # cur_state = torch.Tensor(env.reset())
    # while not done:
    #     # select action
    #     action = policy.select_action(cur_state)
    #
    #     # take action in the environment
    #     next_state, reward, done, info = env.step(action.item())
    #     next_state = torch.Tensor(next_state)
    #
    #     # add the transition to replay buffer
    #     replay_buffer.add(cur_state, action, next_state, reward, done)
    #
    #     # sample minibatch of transitions from the replay buffer
    #     # the sampling is done every timestep and not every episode
    #     sample_transitions = replay_buffer.sample()
    #
    #     # update the policy using the sampled transitions
    #     loss1 = update_policy(**sample_transitions)
    #
    #     episode_unweighted_reward += reward
    #     episode_timestep_unweighted += 1
    #     loss1_cumulative += loss1
    #
    #     cur_state = next_state

    # Now play with weighted policy
    done = False
    cur_state = torch.Tensor(env.reset())
    while not done:
        # select action
        action = policy_weighted.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)

        # add the transition to replay buffer
        replay_buffer_weighted.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer_weighted.sample(100)

        # update the policy using the sampled transitions
        loss2 = update_weighted_policy(cur_state, next_state, reward, action, sample_transitions['next_states'], (episode_i//weight_decay)+1)

        episode_weighted_reward += reward
        episode_timestep_weighted += 1
        loss2_cumulative += loss2
        cur_state = next_state

    avg_history['episodes'].append(episode_i + 1)
    avg_history['timesteps_unweighted'].append(episode_timestep_unweighted)
    avg_history['unweighted_reward'].append(episode_unweighted_reward)
    avg_history['loss_unweighted'].append(loss1_cumulative/episode_timestep_unweighted)

    avg_history['timesteps_weighted'].append(episode_timestep_weighted)
    avg_history['weighted_reward'].append(episode_weighted_reward)
    avg_history['loss_weighted'].append(loss2_cumulative/episode_timestep_weighted)

    if (episode_i + 1) % agg_interval == 0:
        print('Episode : ', episode_i+1, 'Timesteps1 : ',
              avg_history['timesteps_unweighted'][-1], 'Timesteps2 : ', avg_history['timesteps_weighted'][-1],
              'Loss1 :', avg_history['loss_unweighted'][-1], 'Loss2 : ', avg_history['loss_weighted'][-1]
              )

# In[]:
import matplotlib
matplotlib.use('Qt5Agg')

# In[]:

import matplotlib.pyplot as plt
plt.plot(avg_history['episodes'], avg_history['loss_unweighted'], label='unweighted')
plt.ylabel('loss')
plt.plot(avg_history['episodes'], avg_history['loss_weighted'], label='weighted')
plt.legend()
plt.show()

# In[]:

fig, axes = plt.subplots(nrows=2, ncols=2)
fig.set_figheight(5)
fig.set_figwidth(10)
plt.subplots_adjust(wspace=0.5)
axes[0][0].plot(avg_history['episodes'], avg_history['timesteps_unweighted'])
axes[0][0].set_ylabel('Timesteps unweighted')
axes[0][1].plot(avg_history['episodes'], avg_history['unweighted_reward'])
axes[0][1].set_ylabel('Reward unweighted')
axes[1][0].plot(avg_history['episodes'], avg_history['timesteps_weighted'])
axes[1][0].set_ylabel('Timesteps weighted')
axes[1][1].plot(avg_history['episodes'], avg_history['weighted_reward'])
axes[1][1].set_ylabel('Reward weighted')


# In[]:

run_current_policy(env, policy)
env.close()
