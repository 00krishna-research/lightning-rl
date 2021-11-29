"""
Deep Reinforcement Learning: Deep Q-network (DQN)
This example is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.
To run the template just run:
python reinforce_learn_Qnet.py
After ~1500 steps, you will see the total_reward hitting the max score of 200. Open up TensorBoard to
see the metrics:
tensorboard --logdir default
"""

from typing import Tuple, List, Dict
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from src.common import wrappers
from src.common.agents import ValueAgent
from src.common.experience import ExperienceSource, RLDataset
from src.common.memory import ReplayBuffer
from src.common.networks import CNN


class DQNLightning(pl.LightningModule):
    """ Basic DQN Model """

    def __init__(self, 
                 env, 
                 replay_size,
                 warm_start_steps,
                 gamma,
                 eps_start,
                 eps_end,
                 eps_last_frame,
                 sync_rate,
                 lr,
                 episode_length,
                 batch_size,
                 seed: int =123) -> None:
        super().__init__()


        self.env = wrappers.make_env(env)
        self.env.seed(seed)
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_last_frame = eps_last_frame
        self.gamma = gamma
        self.replay_size = replay_size
        self.warm_start_steps = warm_start_steps
        self.sync_rate = sync_rate
        self.lr = lr 
        self.episode_length = episode_length
        self.batch_size = batch_size

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.target_net = None
        self.buffer = None
        self.build_networks()

        self.agent = ValueAgent(self.net, self.n_actions, eps_start=self.eps_start,
                                eps_end=self.eps_end, eps_frames=self.eps_last_frame)
        self.source = ExperienceSource(self.env, self.agent, self.device)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(-21)
        self.avg_reward = -21

    def populate(self, warm_start: int) -> None:
        """Populates the buffer with initial experience"""
        if warm_start > 0:
            for _ in range(warm_start):
                self.source.agent.epsilon = 1.0
                exp, _, _ = self.source.step()
                self.buffer.append(exp)

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = CNN(self.obs_shape, self.n_actions)
        self.target_net = CNN(self.obs_shape, self.n_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes in a state x through the network and gets the q_values of each action as an output

        Args:
            x: environment state

        Returns:
            q values
        """
        output = self.net(x)
        return output

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a mini batch from the replay buffer

        Args:
            batch: current mini batch of replay data

        Returns:
            loss
        """
        states, actions, rewards, dones, next_states = batch

        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_state_action_values = next_state_values * self.gamma + rewards

        return nn.MSELoss()(state_action_values, expected_state_action_values)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved

        Args:
            batch: current mini batch of replay data
            _: batch number, not used

        Returns:
            Training loss and log metrics
        """
        self.agent.update_epsilon(self.global_step)

        # step through environment with agent and add to buffer
        exp, reward, done = self.source.step()
        self.buffer.append(exp)

        self.episode_reward += reward
        self.episode_steps += 1

        # calculates training loss
        loss = self.loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.reward_list.append(self.total_reward)
            self.avg_reward = sum(self.reward_list[-100:]) / 100
            self.episode_count += 1
            self.episode_reward = 0
            self.total_episode_steps = self.episode_steps
            self.episode_steps = 0

        # Soft update of target network
        if self.global_step % self.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        logdict = {'total_reward': self.total_reward,
               'avg_reward': self.avg_reward,
               'train_loss': loss,
               'episode_steps': self.total_episode_steps,
               'epsilon': self.agent.epsilon
               }
        status = {'steps': torch.tensor(self.global_step),
                  'avg_reward': torch.tensor(self.avg_reward),
                  'total_reward': torch.tensor(self.total_reward),
                  'episodes': self.episode_count,
                  'episode_steps': self.episode_steps,
                  'epsilon': self.agent.epsilon
                  }

        self.log("log", logdict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_reward", self.avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        return OrderedDict({'loss': loss, 'avg_reward': torch.tensor(self.avg_reward),
                            'log': logdict, 'progress_bar': status})

    def test_step(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        """Evaluate the agent for 10 episodes"""
        self.agent.epsilon = 0.0
        test_reward = self.source.run_episode()

        return {'test_reward': test_reward}

    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        rewards = [x['test_reward'] for x in outputs]
        avg_reward = sum(rewards) / len(rewards)
        tensorboard_logs = {'avg_test_reward': avg_reward}
        return {'avg_test_reward': avg_reward, 'test_log': tensorboard_logs}

    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        self.buffer = ReplayBuffer(self.replay_size)
        self.populate(self.warm_start_steps)

        dataset = RLDataset(self.buffer, self.episode_length)
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

    def test_dataloader(self) -> DataLoader:
        """Get test loader"""
        return self._dataloader()
