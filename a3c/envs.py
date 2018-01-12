import gym
import numpy as np
from gym.spaces.box import Box
from dragonfly.DragonflyGym import DragonflyEnv
from dragonfly.segmentation import Segmentation

env_count = 0
segmentation = None

def create_unreal_env():
    global env_count
    env = SegmentedEnv(ip='140.112.170.182', port=16660 + env_count)
    env_count += 1
    return env

class SegmentedEnv(DragonflyEnv):
    def __init__(self, **kwargs):
        super(SegmentedEnv, self).__init__(**kwargs)
        self.num_classes = len(utils.labels)
        self.observation_space = gym.spaces.Box(
            0, 255, shape=(self.screen_height, self.screen_width, self.num_classes));
        global segmentation
        if segmentation is None:
            segmentation = Segmentation()

    def observe(self):
        observation = super(SegmentedEnv, self).observe()
        class_image = segmentation.segment(observation)
        one_hot_image = np.eye(self.num_classes)[class_image.reshape(-1)]\
            .reshape(self.screen_height, self.screen_width, self.num_classes)
        return one_hot_image.transpose(2, 0, 1)
