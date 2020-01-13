import gym
import torch
from dqn import DQN, ReplayBuffer
from torch.optim import Adam
from torch.nn.functional import mse_loss
from plot_functions import plot_timesteps_and_rewards

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
learning_rate = 0.001

avg_history = {'episodes': [], 'timesteps': [], 'reward': []}
agg_interval = 10

# initialize policy and replay buffer
policy = DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=24)

# TODO : Check if this can perform better in a smaller no. of hidden sizes
policy_weighted = DQN(input_size=env.observation_space.shape[0], output_size=env.action_space.n, hidden_size=24)
optimizer_weighted = Adam(policy_weighted.parameters(), lr=learning_rate)

replay_buffer = ReplayBuffer()

optimizer = Adam(policy.parameters(), lr=learning_rate)
start_episode = 0

# Play with a random policy and see
# run_current_policy(env.env, policy)

train_episodes = 200

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
    loss = mse_loss(input=expanded_targets, target=targets.detach())
    loss.backward()
    optimizer.step()
    return loss.item()


def update_weighted_policy(cur_state, next_state, reward, cur_states):
    """
    performs a weighted td update by sampling states and weighing them as per the angle between
     the cur_state and the sampled states
    :param cur_state: the current state
    :param next_state: the next state
    :param reward: the reward for cur_State -> next_state transition
    :param cur_states: the sampled states from the replay_buffer
    :return: the loss computed
    """
    # get the weights of the sampled neighbors
    weights_sampled_neighbors = policy_weighted.layer3(policy_weighted.layer2(policy_weighted.layer1(cur_states)))
    self_weight = policy_weighted.layer3(policy_weighted.layer2(policy_weighted.layer1(cur_state)))
    # get the weights as per the angle between cur_state and the sampled states
    weights = torch.mm(weights_sampled_neighbors, self_weight.reshape((-1, 1)))

    weights = ( weights - weights.mean() ) / (weights.std() + 0.001)
    weights = weights.softmax(dim=0).detach()

    # construct the target
    target = reward + gamma*policy_weighted(next_state).max()

    predicted = torch.mul(policy_weighted(cur_states).max(dim=1).values, weights.squeeze(-1))
    # the implementation is (input-target)^2
    optimizer_weighted.zero_grad()
    loss = mse_loss(input=target, target=predicted)
    loss.backward()
    optimizer_weighted.step()
    return loss.item()

# In[]:

# Train the network to predict actions for each of the states
for episode_i in range(train_episodes):
    episode_timestep = 0
    episode_reward = 0.0

    done = False

    cur_state = torch.Tensor(env.reset())

    while not done:
        # select action
        action = policy.select_action(cur_state)

        # take action in the environment
        next_state, reward, done, info = env.step(action.item())
        next_state = torch.Tensor(next_state)

        # add the transition to replay buffer
        replay_buffer.add(cur_state, action, next_state, reward, done)

        # sample minibatch of transitions from the replay buffer
        # the sampling is done every timestep and not every episode
        sample_transitions = replay_buffer.sample()

        # update the policy using the sampled transitions
        update_policy(**sample_transitions)
        update_weighted_policy(cur_state, next_state, reward, sample_transitions['cur_states'])

        episode_reward += reward
        episode_timestep += 1

        cur_state = next_state

    avg_history['episodes'].append(episode_i + 1)
    avg_history['timesteps'].append(episode_timestep)
    avg_history['reward'].append(episode_reward)

    if (episode_i + 1) % agg_interval == 0:
        print('Episode : ', episode_i+1, 'Avg Timestep : ', avg_history['timesteps'][-1])

start_episode = start_episode + train_episodes
plot_timesteps_and_rewards(avg_history)
run_current_policy(env, policy)
env.close()
