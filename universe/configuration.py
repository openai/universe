import logging
from gym import configuration

universe_logger = logging.getLogger('universe')
universe_logger.setLevel(logging.INFO)

extra_logger = logging.getLogger('universe.extra')
extra_logger.setLevel(logging.INFO)

if hasattr(configuration, '_extra_loggers'):
    configuration._extra_loggers.append(universe_logger)
    configuration._extra_loggers.append(extra_logger)
