from gym.envs.registration import register

# ModeEstimation
register(
    id='ModeEstimation0-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.0, 'observation_type': None}
)

register(
    id='ModeEstimation2-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.2, 'observation_type': None}
)

register(
    id='ModeEstimation4-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.4, 'observation_type': None}
)

register(
    id='ModeEstimation6-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.6, 'observation_type': None}
)

register(
    id='ModeEstimation8-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.8, 'observation_type': None}
)

register(
    id='ModeEstimation10-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 1.0, 'observation_type': None}
)

# ModeEstimation Onehot
register(
    id='ModeEstimationOnehot0-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.0, 'observation_type': 'onehot'}
)

register(
    id='ModeEstimationOnehot2-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.2, 'observation_type': 'onehot'}
)

register(
    id='ModeEstimationOnehot4-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.4, 'observation_type': 'onehot'}
)

register(
    id='ModeEstimationOnehot6-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.6, 'observation_type': 'onehot'}
)

register(
    id='ModeEstimationOnehot8-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.8, 'observation_type': 'onehot'}
)

register(
    id='ModeEstimationOnehot10-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 1.0, 'observation_type': 'onehot'}
)

# ModeEstimation Binary
register(
    id='ModeEstimationBinary0-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.0, 'observation_type': 'binary'}
)

register(
    id='ModeEstimationBinary2-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.2, 'observation_type': 'binary'}
)

register(
    id='ModeEstimationBinary4-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.4, 'observation_type': 'binary'}
)

register(
    id='ModeEstimationBinary6-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.6, 'observation_type': 'binary'}
)

register(
    id='ModeEstimationBinary8-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.8, 'observation_type': 'binary'}
)

register(
    id='ModeEstimationBinary10-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 1.0, 'observation_type': 'binary'}
)
