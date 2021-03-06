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
from copy import deepcopy
from itertools import chain
from typing import Tuple, List
import argparse
from collections import OrderedDict

import torch
from torch import Tensor
import torch.optim as optim
from torch.nn.functional import log_softmax, softmax
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import gym
from src.common.agents import PolicyAgent
from src.common.experience import EpisodicExperienceStream
from src.common.memory import Experience
from src.common.networks import MLP
from src.common.wrappers import ToTensor


class VPGLightning(pl.LightningModule):
    """ VPG Model """

    def __init__(self, 
                 env,
                 entropy_beta,
                 gamma,
                 batch_episodes,
                 lr,
                 seed   ) -> None:
        super().__init__()


        # self.env = wrappers.make_env(env)    # use for Atari
        self.env = ToTensor(gym.make(env))     # use for Box2D/Control
        self.env.seed(seed)
        self.env_name = env
        self.gamma = gamma
        self.batch_episodes = batch_episodes
        self.lr = lr

        self.obs_shape = self.env.observation_space.shape
        self.n_actions = self.env.action_space.n

        self.net = None
        self.build_networks()

        self.agent = PolicyAgent(self.net)

        self.total_reward = 0
        self.episode_reward = 0
        self.episode_count = 0
        self.episode_steps = 0
        self.total_episode_steps = 0
        self.entropy_beta = entropy_beta

        self.reward_list = []
        for _ in range(100):
            self.reward_list.append(0)
        self.avg_reward = 0

    def build_networks(self) -> None:
        """Initializes the DQN train and target networks"""
        self.net = MLP(self.obs_shape, self.n_actions)

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

    def calc_qvals(self, rewards: List[Tensor]) -> List[Tensor]:
        """
        Takes in the rewards for each batched episode and returns list of qvals for each batched episode

        Args:
            rewards: list of rewards for each episodes in the batch

        Returns:
            List of qvals for each episodes
        """
        res = []
        sum_r = 0.0
        for reward in reversed(rewards):
            sum_r *= self.gamma
            sum_r += reward
            res.append(deepcopy(sum_r))
        res = list(reversed(res))
        # Subtract the mean (baseline) from the q_vals to reduce the high variance
        sum_q = 0
        for rew in res:
            sum_q += rew
        mean_q = sum_q / len(res)
        return [q - mean_q for q in res]

    def process_batch(self, batch: List[List[Experience]]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """
        Takes in a batch of episodes and retrieves the q vals, the states and the actions for the batch

        Args:
            batch: list of episodes, each containing a list of Experiences

        Returns:
            q_vals, states and actions used for calculating the loss
        """
        # get outputs for each episode
        batch_rewards, batch_states, batch_actions = [], [], []
        for episode in batch:
            ep_rewards, ep_states, ep_actions = [], [], []

            # log the outputs for each step
            for step in episode:
                ep_rewards.append(step[2].float())
                ep_states.append(step[0])
                ep_actions.append(step[1])

            # add episode outputs to the batch
            batch_rewards.append(ep_rewards)
            batch_states.append(ep_states)
            batch_actions.append(ep_actions)

        # get qvals
        batch_qvals = []
        for reward in batch_rewards:
            batch_qvals.append(self.calc_qvals(reward))

        # flatten the batched outputs
        batch_actions, batch_qvals, batch_rewards, batch_states = self.flatten_batch(batch_actions, batch_qvals,
                                                                                     batch_rewards, batch_states)

        return batch_qvals, batch_states, batch_actions, batch_rewards

    @staticmethod
    def flatten_batch(batch_actions: List[List[Tensor]], batch_qvals: List[List[Tensor]],
                      batch_rewards: List[List[Tensor]], batch_states: List[List[Tuple[Tensor, Tensor]]]) \
            -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Takes in the outputs of the processed batch and flattens the several episodes into a single tensor for each
        batched output

        Args:
            batch_actions: actions taken in each batch episodes
            batch_qvals: Q vals for each batch episode
            batch_rewards: reward for each batch episode
            batch_states: states for each batch episodes

        Returns:
            The input batched results flattend into a single tensor
        """
        # flatten all episode steps into a single list
        batch_qvals = list(chain.from_iterable(batch_qvals))
        batch_states = list(chain.from_iterable(batch_states))
        batch_actions = list(chain.from_iterable(batch_actions))
        batch_rewards = list(chain.from_iterable(batch_rewards))

        # stack steps into single tensor and remove extra dimension
        batch_qvals = torch.stack(batch_qvals).squeeze()
        batch_states = torch.stack(batch_states).squeeze()
        batch_actions = torch.stack(batch_actions).squeeze()
        batch_rewards = torch.stack(batch_rewards).squeeze()

        return batch_actions, batch_qvals, batch_rewards, batch_states

    def loss(self, batch_qvals: List[Tensor], batch_states: List[Tensor], batch_actions: List[Tensor]) -> torch.Tensor:
        """
        Calculates the mse loss using a batch of states, actions and Q values from several episodes. These have all
        been flattend into a single tensor.

        Args:
            batch_qvals: current mini batch of q values
            batch_actions: current batch of actions
            batch_states: current batch of states

        Returns:
            loss
        """
        logits = self.net(batch_states)

        log_prob, policy_loss = self.calc_policy_loss(batch_actions, batch_qvals, batch_states, logits)

        entropy_loss_v = self.calc_entropy_loss(log_prob, logits)

        loss = policy_loss + entropy_loss_v

        return loss

    def calc_entropy_loss(self, log_prob: Tensor, logits: Tensor) -> Tensor:
        """
        Calculates the entropy to be added to the loss function
        Args:
            log_prob: log probabilities for each action
            logits: the raw outputs of the network

        Returns:
            entropy penalty for each state
        """
        prob_v = softmax(logits, dim=1)
        entropy_v = -(prob_v * log_prob).sum(dim=1).mean()
        entropy_loss_v = -self.entropy_beta * entropy_v
        return entropy_loss_v

    @staticmethod
    def calc_policy_loss(batch_actions: Tensor, batch_qvals: Tensor,
                         batch_states: Tensor, logits: Tensor) -> Tuple[List, Tensor]:
        """
        Calculate the policy loss give the batch outputs and logits
        Args:
            batch_actions: actions from batched episodes
            batch_qvals: Q values from batched episodes
            batch_states: states from batched episodes
            logits: raw output of the network given the batch_states

        Returns:
            policy loss
        """
        log_prob = log_softmax(logits, dim=1)
        log_prob_actions = batch_qvals * log_prob[range(len(batch_states)), batch_actions]
        policy_loss = -log_prob_actions.mean()
        return log_prob, policy_loss

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

        batch_qvals, batch_states, batch_actions, batch_rewards = self.process_batch(batch)

        # get avg reward over the batched episodes
        self.episode_reward = sum(batch_rewards) / len(batch)
        self.reward_list.append(self.episode_reward)
        self.avg_reward = sum(self.reward_list) / len(self.reward_list)

        # calculates training loss
        loss = self.loss(batch_qvals, batch_states, batch_actions)

        self.episode_count += self.batch_episodes

        logdict = {'episode_reward': torch.tensor(self.episode_reward),
               'train_loss': loss,
               'avg_reward': self.avg_reward
               }
        status = {'steps': torch.tensor(self.global_step),
                  'episode_reward': torch.tensor(self.episode_reward),
                  'episodes': torch.tensor(self.episode_count),
                  'avg_reward': self.avg_reward
                  }
        self.log("log", logdict, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("avg_reward", self.avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=False)
        self.episode_reward = 0

        return OrderedDict({'loss': loss, 'reward': self.avg_reward, 'log': logdict, 'progress_bar': status})


    def configure_optimizers(self) -> List[Optimizer]:
        """ Initialize Adam optimizer"""
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        return [optimizer]

    def _dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""
        dataset = EpisodicExperienceStream(self.env, self.agent, self.device, episodes=self.batch_episodes)
        dataloader = DataLoader(dataset=dataset)
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader"""
        return self._dataloader()

