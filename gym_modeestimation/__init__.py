from gym.envs.registration import register

# ModeEstimation
register(
    id='ModeEstimationEPS00-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.0}
)

register(
    id='ModeEstimationEPS02-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.2}
)

register(
    id='ModeEstimationEPS04-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.4}
)

register(
    id='ModeEstimationEPS06-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.6}
)

register(
    id='ModeEstimationEPS08-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 0.8}
)

register(
    id='ModeEstimationEPS10-v0',
    entry_point='gym_modeestimation.modeestimation:ModeEstimationEnv',
    kwargs={'eps': 1.0}
)
