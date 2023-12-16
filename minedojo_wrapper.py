import math
from abc import ABC

import gym
import minedojo.sim.wrappers
import numpy as np
from gym.spaces import Discrete, Box
from skimage.transform import downscale_local_mean

class MinedojoActionWrapper(gym.ActionWrapper, ABC):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(6)

    def action(self, action):
        forward = action == 0
        backward = action == 1
        # left = action == 2
        # right = action == 3
        cam_pitch_l = action == 2
        cam_pitch_r = action == 3
        cam_yaw_up = action == 4
        cam_yaw_down = action == 5

        return [
            1 if forward else 2 if backward else 0,  # 0
            0,
            0,
            11 if cam_yaw_up else 13 if cam_yaw_down else 12,  # 3
            11 if cam_pitch_l else 13 if cam_pitch_r else 12,  # 4
            0,
            0,
            0,
        ]


class MinedojoObservationWrapper(gym.ObservationWrapper, ABC):

    def __init__(self, env, target_size):
        super().__init__(env)
        self.observation_space = Box(shape=(*target_size, 3), low=0, high=255, dtype=np.uint8)

    def observation(self, obs):
        obs_img = obs['rgb']
        obs_img = np.moveaxis(obs_img, 0, 2)
        return self.scale_image(obs_img)

    def scale_image(self, image, method='downsample_local_mean'):
        source_shape = image.shape
        target_shape = self.observation_space.shape

        assert target_shape[0] == target_shape[1], "Target shape must be quadratic"

        # Image should always be quadratic, remove bottom of image by slicing
        slice_width = source_shape[1]
        image = image[:slice_width, :slice_width]

        if method == 'downsample_local_mean':
            downscale_factor = math.floor(slice_width / target_shape[0])
            return downscale_local_mean(image, (downscale_factor, downscale_factor))
