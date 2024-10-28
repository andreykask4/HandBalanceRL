import numpy as np
import gym
import matplotlib.pyplot as plt
import os
import math
import pickle

# Custom environment for balancing a hand on a platform
class HandBalanceEnv(gym.Env):
    def __init__(self):
        super(HandBalanceEnv, self).__init__()
        self.action_space = gym.spaces.Discrete(3)  # Actions: left, stay, right
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)  # State: position, velocity
        self.state = None

    def reset(self):
        # Reset the state to a random position and zero velocity
        self.state = np.random.uniform(low=-0.1, high=0.1, size=(2,))
        return self.state

    def step(self, action):
        position, velocity = self.state
        
        # Update position based on action
        if action == 0:  # Move left
            position -= 0.1
        elif action == 2:  # Move right
            position += 0.1

        # Simulate instability by adding random noise to velocity
        velocity += np.random.uniform(-0.05, 0.05)
        position += velocity

        # Check if the hand is still on the platform
        done = abs(position) > 1.0
        reward = 1.0 if not done else -5.0  # Reward for staying on the platform
        if abs(position) < 0.1:
            reward += 2
        elif abs(position) < 0.3:
            reward += 1
        elif abs(position) < 0.5:
            pass
        elif abs(position) < 0.6:
            reward -= 1
        elif abs(position) < 0.7:
            reward -= 2
        else:
            reward -= 3


        # Clip the state to keep it within bounds
        self.state = np.clip([position, velocity], -1, 1)
        return self.state, reward, done, {}

    def render(self, ax, falling=False, last_position=None):
        ax.clear()  # Clear the previous frame
        ax.set_xlim(-1, 1)  # Set x-axis limits
        ax.set_ylim(-0.5, 1)  # Set y-axis limits

        # Draw the stick
        position, velocity = self.state  # Assuming state contains position and angle
        stick_length = 0.5  # Length of the stick
        stick_base_x = position  # x position of the stick's base
        stick_base_y = stick_length  # y position based on angle

        # Получаем новые координаты
        new_x, new_y = rotate_point_around_fixed(stick_base_x, 0, stick_base_x, stick_base_y, velocity)
        stick_x = [stick_base_x, new_x]
        stick_y = [0, new_y]

        # Draw the stick as a red line
        ax.plot(stick_x, stick_y, 'r-', lw=5)  
        ax.plot(stick_base_x, 0, 'bo')  # Draw the base of the stick as a blue dot

        if falling:
            ax.text(0, 0.5, "Falling!", fontsize=15, color='red', ha='center')
        plt.draw()  # Update the plot

# Q-learning agent
class QLearningAgent:
    def __init__(self, action_space):
        # Initialize Q-table with zeros
        self.q_table = np.zeros((10, 10, action_space.n))  # Discretized state space
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.epsilon = 0.1  # Exploration rate

    def discretize_state(self, state):
        # Discretize the continuous state into a finite state space
        position, velocity = state
        pos_index = int(np.digitize(position, bins=np.linspace(-1, 1, 10)) - 1)
        vel_index = int(np.digitize(velocity, bins=np.linspace(-1, 1, 10)) - 1)
        return pos_index, vel_index

    def choose_action(self, state):
        # Choose action based on epsilon-greedy policy
        if np.random.rand() < self.epsilon:
            return np.random.choice(range(self.q_table.shape[2]))  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update(self, state, action, reward, next_state):
        # Update the Q-table using the Q-learning formula
        best_next_action = np.argmax(self.q_table [next_state])  # Calculate the best next action
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_delta = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_delta


def rotate_point_around_fixed(x0, y0, x1, y1, speed):
    # Преобразуем угол в радианы
    angle_radians = speed * math.pi / 2
    
    # Вектор от фиксированной точки до конечной точки
    dx = x1 - x0
    dy = y1 - y0
    
    # Новые координаты конечной точки после поворота
    new_x1 = x0 + dx * math.cos(angle_radians) - dy * math.sin(angle_radians)
    new_y1 = y0 + dx * math.sin(angle_radians) + dy * math.cos(angle_radians)
    
    return new_x1, new_y1

# Training loop
def train_agent(episodes=1000):
    env = HandBalanceEnv()
    agent = QLearningAgent(env.action_space)
    rewards_per_episode = []  # To monitor rewards over episodes

    # Load previous results if they exist
    if os.path.exists('q_table.pkl') and os.path.exists('rewards.pkl'):
        with open('q_table.pkl', 'rb') as f:
            agent.q_table = pickle.load(f)
        with open('rewards.pkl', 'rb') as f:
            rewards_per_episode = pickle.load(f)

    try:
        for episode in range(episodes):
            state = env.reset()
            state = agent.discretize_state(state)  # Discretize the initial state
            total_reward = 0
            done = False
            
            fig, ax = plt.subplots(figsize=(12, 8))  # Create a larger figure for animation
            while not done:
                action = agent.choose_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = agent.discretize_state(next_state)  # Discretize the next state
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
                
                # Render the current state
                env.render(ax, falling=done)  # Render the environment
                plt.pause(0.0001)  # Pause for 1ms to update the animation
            print(total_reward)
            rewards_per_episode.append(total_reward)  # Store total reward for monitoring

            plt.close()  # Close the animation figure

    except KeyboardInterrupt:
        print("Training interrupted. Saving results...")
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(agent.q_table, f)
        with open('rewards.pkl', 'wb') as f:
            pickle.dump(rewards_per_episode, f)
        print("Results saved.")

    return rewards_per_episode

# Run the training and monitor performance
if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU
    rewards = train_agent(episodes=1000)

    # Plotting the rewards over episodes
    plt.plot(rewards)
    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()