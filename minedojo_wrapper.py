import math
import random
from abc import ABC

import gym
import minedojo
import numpy as np
from gym.spaces import Discrete, Box
from skimage.transform import downscale_local_mean

from Environments import SkyBlockDrawer


class MinedojoSkyBlockEnv(gym.Wrapper, ABC):

    def __init__(self,
                 n_islands=4200,
                 island_margin=50,
                 image_size=(560, 512),
                 obs_size=(64, 64),
                 break_speed_multiplier=10):
        self.draw_string, respawn_locations = SkyBlockDrawer.draw_skyblock_grid(n_islands, marg=island_margin)

        env = minedojo.make(
            task_id='open-ended',
            image_size=image_size,
            generate_world_type='flat',
            flat_world_seed_string="0",
            start_position=dict(x=0, y=2, z=0, yaw=0, pitch=0),
            start_time=6000,
            allow_time_passage=False,
            drawing_str=self.draw_string,
            use_lidar=True,
            allow_mob_spawn=False,
            break_speed_multiplier=break_speed_multiplier,
            lidar_rays=[(0, 0, 999)]
        )

        env = MinedojoActionWrapper(env)
        env = MinedojoSkyBlockRewardWrapper(env)
        env = MinedojoObservationWrapper(env, target_size=obs_size)
        env = MinedojoSkyBlockResetWrapper(env, respawn_locations)

        super().__init__(env)


class MinedojoActionWrapper(gym.ActionWrapper, ABC):

    craft_id = {
        "wooden_pickaxe": 219,
        "crafting_table": 7,
        "stick": 41,
        "planks": 29
    }

    actions = [
        "forward", "backward", "jump",
        "cam_pitch_l", "cam_pitch_r",
        "cam_yaw_up", "cam_yaw_down",
        "attack", "place",
        "craft_pickaxe", "craft_table", "craft_sticks", "craft_planks"
    ]

    FORWARD = actions.index("forward")
    BACKWARD = actions.index("backward")
    CAM_PITCH_L = actions.index("cam_pitch_l")
    CAM_PITCH_R = actions.index("cam_pitch_r")
    CAM_YAW_UP = actions.index("cam_yaw_up")
    CAM_YAW_DOWN = actions.index("cam_yaw_down")
    ATTACK = actions.index("attack")
    JUMP = actions.index("jump")
    PLACE = actions.index("place")
    CRAFT_PICKAXE = actions.index("craft_pickaxe")
    CRAFT_TABLE = actions.index("craft_table")
    CRAFT_STICKS = actions.index("craft_sticks")
    CRAFT_PLANKS = actions.index("craft_planks")

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Discrete(len(self.actions))

    def action(self, action):
        forward = action == self.FORWARD
        backward = action == self.BACKWARD
        cam_pitch_l = action == self.CAM_PITCH_L
        cam_pitch_r = action == self.CAM_PITCH_R
        cam_yaw_up = action == self.CAM_YAW_UP
        cam_yaw_down = action == self.CAM_YAW_DOWN
        attack = action == self.ATTACK
        jump = action == self.JUMP
        place = action == self.PLACE

        craft_id = self.craft_id["wooden_pickaxe"] if action == self.CRAFT_PICKAXE \
            else self.craft_id["crafting_table"] if action == self.CRAFT_TABLE \
            else self.craft_id["stick"] if action == self.CRAFT_STICKS \
            else self.craft_id["planks"] if action == self.CRAFT_PLANKS \
            else 0
        craft = craft_id != 0

        return [
            1 if forward else 2 if backward else 0,  # 0
            0,
            1 if jump else 0,
            11 if cam_yaw_up else 13 if cam_yaw_down else 12,  # 3
            11 if cam_pitch_l else 13 if cam_pitch_r else 12,  # 4
            3 if attack else 4 if craft else 6 if place else 0,
            craft_id,
            0,
        ]


class MinedojoObservationWrapper(gym.ObservationWrapper, ABC):

    def __init__(self, env, target_size):
        super().__init__(env)
        self.observation_space = Box(shape=(*target_size, 3), low=0, high=255, dtype=np.uint8)

    def observation(self, obs):
        obs_img = obs['rgb']
        obs_img = np.moveaxis(obs_img, 0, 2)
        obs_img = self.scale_image(obs_img)
        return obs_img

    def scale_image(self, image, method='downsample_local_mean'):
        source_shape = image.shape
        target_shape = self.observation_space.shape

        # No scaling required, return original image
        if source_shape == target_shape:
            return image

        assert target_shape[0] == target_shape[1], "Target shape must be quadratic"

        # Image should always be quadratic, remove bottom of image by slicing
        slice_width = source_shape[1]
        image = image[:slice_width, :slice_width]

        if method == 'downsample_local_mean':
            sc_fc = math.floor(slice_width / target_shape[0])
            return downscale_local_mean(image, (sc_fc, sc_fc, 1)).astype(np.uint8)


class MinedojoSkyBlockResetWrapper(gym.Wrapper, ABC):

    def __init__(self, env, respawn_points):
        super().__init__(env)
        self.respawn_points = respawn_points
        self.respawn_point_index = 0
        self.env_initiated = False

    def reset(self, **kwargs):
        if not self.env_initiated:
            self.env.reset()
            self.env_initiated = True

        x, y, z, yaw, pitch = self.get_respawn_location()
        self.respawn_point_index += 1

        self.env.teleport_agent(x=x, y=y, z=z, yaw=yaw, pitch=pitch)

        for i in range(2):
            # Perform 2 no-ops to let the agent land on the ground
            obs, reward, done, info = self.step(100)

        return obs

    def get_respawn_location(self):
        possible_cords = self.respawn_points[self.respawn_point_index]

        cord = random.choice(possible_cords)
        yaw = random.choice([-90, 0, 90, 180])

        return cord[0], 2, cord[1], yaw, 0


class MinedojoSkyBlockRewardWrapper(gym.Wrapper, ABC):

    def __init__(self, env,
                 terminate_at_y_pos=-5,
                 exploration_reward=1,
                 death_penalty=-100,
                 pickup_block_reward_dict=None):
        super().__init__(env)

        if pickup_block_reward_dict is None:
            pickup_block_reward_dict = {
                'wood': 15,
                'dirt': 5,
                'sticks': 10,
                'planks': 15,
                'crafting_table': 30,
                'wooden_pickaxe': 100,
                'cobblestone': 1000
            }

        self.pickup_block_reward_dict = pickup_block_reward_dict
        self.terminate_at_y_pos = terminate_at_y_pos
        self.exploration_reward = exploration_reward
        self.death_penalty = death_penalty

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        is_dead = info.get('ypos') < self.terminate_at_y_pos or info.get('is_dead') or info.get(
            'living_death_event_fired')

        if is_dead:
            done = True
            reward += self.death_penalty

        if action == MinedojoActionWrapper.FORWARD:
            reward += self.exploration_reward

        reward += self.get_delta_inv_reward(obs["delta_inv"])

        return obs, reward, done, info

    def get_delta_inv_reward(self, delta_inv):
        """
        Delta inventory observation consists of inventory changes caused by crafting and by other actions (Such as
        picking up items). This method will look at these changes and calculate the reward based on the item-pickup.
        The function does not differ on inventory change source reward

        :param delta_inv: The delta inventory observation space. Found by calling obs["delta_inv"]
        :return: Reward earned by the recorded inventory changes
        """
        reward = 0

        delta_inv_name_other = delta_inv["inc_name_by_other"][0]
        delta_inv_quantity_other = delta_inv["inc_quantity_by_other"][0]
        delta_inv_name_craft = delta_inv["inc_name_by_craft"][0]
        delta_inv_quantity_craft = delta_inv["inc_quantity_by_craft"][0]

        if delta_inv_name_other in self.pickup_block_reward_dict.keys():
            reward += self.pickup_block_reward_dict[delta_inv_name_other] * delta_inv_quantity_other

        if delta_inv_name_craft in self.pickup_block_reward_dict.keys():
            reward += self.pickup_block_reward_dict[delta_inv_name_craft] * delta_inv_quantity_craft

        return reward


