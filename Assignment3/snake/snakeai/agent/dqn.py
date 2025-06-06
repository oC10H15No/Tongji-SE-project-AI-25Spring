import collections
import numpy as np

from snakeai.agent import AgentBase
from snakeai.utils.memory import ExperienceReplay


class DeepQNetworkAgent(AgentBase):
    """ Represents a Snake agent powered by DQN with experience replay. """

    def __init__(self, model, num_last_frames=4, memory_size=1000):
        """
        Create a new DQN-based agent.
        
        Args:
            model: a compiled DQN model.
            num_last_frames (int): the number of last frames the agent will consider.
            memory_size (int): memory size limit for experience replay (-1 for unlimited). 
        """
        assert model.input_shape[1] == num_last_frames, 'Model input shape should be (num_frames, grid_size, grid_size)'
        assert len(model.output_shape) == 2, 'Model output shape should be (num_samples, num_actions)'

        self.model = model
        self.num_last_frames = num_last_frames
        self.memory = ExperienceReplay((num_last_frames,) + model.input_shape[-2:], model.output_shape[-1], memory_size)
        self.frames = None

    def begin_episode(self):
        """ Reset the agent for a new episode. """
        self.frames = None

    def get_last_frames(self, observation):
        """
        Get the pixels of the last `num_last_frames` observations, the current frame being the last.
        
        Args:
            observation: observation at the current timestep. 

        Returns:
            Observations for the last `num_last_frames` frames.
        """
        frame = observation
        if self.frames is None:
            self.frames = collections.deque([frame] * self.num_last_frames)
        else:
            self.frames.append(frame)
            self.frames.popleft()
        return np.expand_dims(self.frames, 0)

    def train(self, env, num_episodes=1000, batch_size=50, discount_factor=0.9, checkpoint_freq=None,
              exploration_range=(1.0, 0.1), exploration_phase_size=0.5, start_episode=0):
        """
        Train the agent to perform well in the given Snake environment.
        
        Args:
            env:
                an instance of Snake environment.
            num_episodes (int):
                the number of episodes to run during the training.
            batch_size (int):
                the size of the learning sample for experience replay.
            discount_factor (float):
                discount factor (gamma) for computing the value function.
            checkpoint_freq (int):
                the number of episodes after which a new model checkpoint will be created.
            exploration_range (tuple):
                a (max, min) range specifying how the exploration rate should decay over time. 
            exploration_phase_size (float):
                the percentage of the training process at which
                the exploration rate should reach its minimum.
        """

        # Calculate the constant exploration decay speed for each episode.
        max_exploration_rate, min_exploration_rate = exploration_range
        
        adjusted_num_episodes_for_decay = num_episodes * exploration_phase_size
        
        if adjusted_num_episodes_for_decay <= 0:
            # If no decay phase in this run (e.g., num_episodes or exploration_phase_size is too small/zero)
            # Set exploration rate to min if resuming, or max if starting fresh.
            exploration_rate = min_exploration_rate if start_episode > 0 else max_exploration_rate
            exploration_decay = 0.0 # No decay if there's no phase for it
        else:
            # Calculate decay per episode based on the current run's parameters
            # This is how much epsilon would reduce per episode if starting from max_exploration_rate
            # and decaying over 'adjusted_num_episodes_for_decay' episodes.
            decay_per_episode_this_run = (max_exploration_rate - min_exploration_rate) / adjusted_num_episodes_for_decay
            
            if start_episode > 0:
                # Calculate what the exploration rate should be, as if it has been decaying
                # at `decay_per_episode_this_run` for `start_episode` number of episodes.
                exploration_rate = max_exploration_rate - (decay_per_episode_this_run * start_episode)
                exploration_rate = max(min_exploration_rate, exploration_rate) # Ensure it's not below min
            else:
                # Start from max exploration rate if it's a fresh training (start_episode is 0)
                exploration_rate = max_exploration_rate
            
            exploration_decay = decay_per_episode_this_run # This is the amount to subtract in each step of the current run
        
        for episode_offset in range(num_episodes):
            current_episode = episode_offset + start_episode
            
            # Reset the environment for the new episode.
            timestep = env.new_episode()
            self.begin_episode()
            game_over = False
            loss = 0.0

            # Observe the initial state.
            state = self.get_last_frames(timestep.observation)

            while not game_over:
                if np.random.random() < exploration_rate:
                    # Explore: take a random action.
                    action = np.random.randint(env.num_actions)
                else:
                    # Exploit: take the best known action for this state.
                    q = self.model.predict(state)
                    action = np.argmax(q[0])

                # Act on the environment.
                env.choose_action(action)
                timestep = env.timestep()

                # Remember a new piece of experience.
                reward = timestep.reward
                state_next = self.get_last_frames(timestep.observation)
                game_over = timestep.is_episode_end
                experience_item = [state, action, reward, state_next, game_over]
                self.memory.remember(*experience_item)
                state = state_next

                # Sample a random batch from experience.
                batch = self.memory.get_batch(
                    model=self.model,
                    batch_size=batch_size,
                    discount_factor=discount_factor
                )
                # Learn on the batch.
                if batch:
                    inputs, targets = batch
                    loss += float(self.model.train_on_batch(inputs, targets))

            if checkpoint_freq and (current_episode % checkpoint_freq) == 0 and current_episode > 0:
                self.model.save(f'dqn-{current_episode:08d}.h5')

            if exploration_rate > min_exploration_rate:
                exploration_rate -= exploration_decay
                exploration_rate = max(min_exploration_rate, exploration_rate)  # Ensure it does not go below the minimum

            # Calculate target absolute episode for logging
            target_absolute_episode = start_episode + num_episodes

            summary = 'Episode {:5d}/{:5d} | Loss {:8.4f} | Exploration {:.2f} | ' + \
                      'Fruits {:2d} | Timesteps {:4d} | Total Reward {:4d}'
            print(summary.format(
                current_episode + 1,  # Current absolute episode number (1-indexed)
                target_absolute_episode,  # Target absolute episode number for this run
                loss, 
                exploration_rate,
                env.stats.fruits_eaten, 
                env.stats.timesteps_survived, 
                env.stats.sum_episode_rewards,
            ))

        self.model.save(f'dqn-final.h5')

    def act(self, observation, reward):
        """
        Choose the next action to take.
        
        Args:
            observation: observable state for the current timestep. 
            reward: reward received at the beginning of the current timestep.

        Returns:
            The index of the action to take next.
        """
        state = self.get_last_frames(observation)
        q = self.model.predict(state)[0]
        return np.argmax(q)
