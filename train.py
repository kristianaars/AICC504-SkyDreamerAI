from minedojo_wrapper import MinedojoSkyBlockEnv

import warnings
import dreamerv3
from dreamerv3 import embodied
from dreamerv3.embodied.envs import from_gym


def main():
    warnings.filterwarnings('ignore', '.*truncated to dtype int32.*')

    # View dreamer3 documentation for config-explaination
    config = embodied.Config(dreamerv3.configs['defaults'])
    config = config.update(dreamerv3.configs['medium'])
    config = config.update({
        'logdir': '~/logdir/run4',
        'run.train_ratio': 64,
        'run.log_every': 10,  # Seconds
        'batch_size': 8,
        'jax.prealloc': False,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
        'jax.platform': 'cpu',  # Can be changed to gpu if CUDA-GPU is available. Requires JAX with CUDA
    })

    config = embodied.Flags(config).parse()

    logdir = embodied.Path(config.logdir)
    step = embodied.Counter()
    logger = embodied.Logger(step, [
        embodied.logger.TerminalOutput(),
        embodied.logger.JSONLOutput(logdir, 'metrics.jsonl'),
        embodied.logger.TensorBoardOutput(logdir),
    ])

    env = MinedojoSkyBlockEnv(n_islands=512)

    # Wrap gym environment in dreamerv3 compatible format
    env = from_gym.FromGym(env, obs_key='image')
    env = dreamerv3.wrap_env(env, config)
    env = embodied.BatchEnv([env], parallel=False)

    agent = dreamerv3.Agent(env.obs_space, env.act_space, step, config)
    replay = embodied.replay.Uniform(config.batch_length, config.replay_size, logdir / 'replay')

    args = embodied.Config(**config.run, logdir=config.logdir, batch_steps=config.batch_size * config.batch_length)
    embodied.run.train(agent, env, replay, logger, args)
    # embodied.run.eval_only(agent, env, logger, args)


if __name__ == '__main__':
    main()
