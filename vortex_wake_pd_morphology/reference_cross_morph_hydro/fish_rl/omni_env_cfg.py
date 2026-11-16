"""N=5 FishGoalOmni environment: only task changes from the validated V4 task."""
from __future__ import annotations

import math

from mjlab.envs.mdp.events import reset_root_state_uniform
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.registry import register_mjlab_task

from fish_rl import mdp
from fish_rl.env_cfg import make_env_cfg
from fish_rl.omni_mdp import FishGoalOmniCfg
from fish_rl.rl_cfg import make_rl_cfg

TASK_ID = "Mjlab-Fish-Goal-Omni-N5"


def make_omni_env_cfg(*, num_envs: int = 4096, play: bool = False):
    cfg = make_env_cfg(num_envs=num_envs, play=play)
    cfg.episode_length_s = 30.0
    cfg.events["randomize_initial_yaw"] = EventTermCfg(
        func=reset_root_state_uniform, mode="reset",
        params={"pose_range": {"yaw": (-math.pi, math.pi)},
                "asset_cfg": SceneEntityCfg("fish")})
    cfg.commands["goal"] = FishGoalOmniCfg(
        distance_range=(1.5, 6.0), angle_range=(-math.pi, math.pi),
        marker_name="pool_goal_marker" if play else "goal_marker")
    cfg.terminations["out_of_bounds"] = TerminationTermCfg(
        func=mdp.out_of_bounds, params={"radius": 12.0})
    return cfg


def register_omni_task(*, num_envs: int = 4096) -> str:
    agent = make_rl_cfg()
    agent.experiment_name = "fish_N5_goal_omni"
    agent.max_iterations = 1500
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=make_omni_env_cfg(num_envs=num_envs),
        play_env_cfg=make_omni_env_cfg(num_envs=1, play=True),
        rl_cfg=agent,
    )
    return TASK_ID
