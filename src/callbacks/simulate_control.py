
from typing import List

import torch
import torch.nn as nn
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from gym.wrappers.monitoring import video_recorder
import glob
import base64, io
import gym
from IPython.display import HTML
from IPython import display 
from torch.distributions import Categorical
from torch.nn.functional import log_softmax, softmax



class SimulateControl(Callback):
    """Make wandb watch model at the beginning of the run."""

    def __init__(self, simulation_steps: int):
        self.simulation_steps = simulation_steps

    def on_train_end(self, trainer, pl_module):

        self.env = trainer.model.env
        self.env_name = trainer.model.env_name
        self.model = trainer.model.forward
        state = self.env.reset()
        vid = video_recorder.VideoRecorder(self.env, 
                                           path="{}.mp4".format(self.env_name))
        done = False
        for t in range(self.simulation_steps):
            vid.capture_frame()
            probs = self.model(state)
            probs = softmax(probs.unsqueeze(0), dim=1)
            model = Categorical(probs)
            action_sample = model.sample()
            action = action_sample.item()
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            if done:
                state = self.env.reset()
        vid.close()
        self.env.close()
