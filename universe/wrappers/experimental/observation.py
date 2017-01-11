import logging

from universe import vectorized, runtime_spec

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def CropObservations(env):
    """"
    Crops the visual observations of an environment so that they only contain the game screen.
    Removes anything outside the game that usually belongs to universe (browser borders and so on).
    """
    if env.spec.tags.get('flashgames', False):
        spec = runtime_spec('flashgames').server_registry[env.spec.id]
        return _CropObservations(env, x=18, y=84, height=spec["height"], width=spec["width"])
    elif (env.spec.tags.get('atari', False) and env.spec.tags.get('vnc', False)):
        return _CropObservations(env, height=194, width=160)
    else:
        # if unknown environment (or local atari), do nothing
        return env

class _CropObservations(vectorized.ObservationWrapper):
    def __init__(self, env, height, width, x=0, y=0):
        super(_CropObservations, self).__init__(env)
        self.x = x
        self.y = y
        self.height = height
        self.width = width

        # modify observation_space? (if so, how to know depth and channels before we have seen the first frame?)
        # self.observation_space = Box(0, 255, shape=(height, width, 3))

    def _observation(self, observation_n):
        return [self._crop_frame(observation) for observation in observation_n]

    def _crop_frame(self, frame):
        if frame is not None:
            if isinstance(frame, dict):
                frame['vision'] = frame['vision'][self.y:self.y + self.height, self.x:self.x + self.width]
            else:
                frame = frame[self.y:self.y + self.height, self.x:self.x + self.width]
        return frame
