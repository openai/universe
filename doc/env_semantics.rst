Environment semantics
*********************
  
Real-time environments
======================

Universe environments differ from other Gym environments in that the
environment keeps running in real-time, even when the agent does not
call ``step``. This has a few important implications:

* Actions and observations can no longer be considered to
  occur on a "clock tick".
* An explicit call to ``reset`` is asynchronous and returns
  immediately, even though the environment has not yet finished
  resetting. (If you would prefer the ``reset`` call to block
  until the reset has finished, you can wrap
  the client-side environment with a `BlockingReset <https://github.com/openai/universe/blob/master/universe/wrappers/blocking_reset.py>`__ wrapper)
* Since the environment will not have waited to finish
  connecting to the VNC server before returning, the initial return
  values from ``reset`` will be ``None`` to indicate that there is
  not yet a valid observation.
* An agent that successfully learns from a Universe environment cannot
  take "thinking breaks": it must keep sending actions to the
  environment at all times.
* Lag and latency play a major role in your agent's ability to
  successfully learn in a given environment. The latency and profiling
  numbers returned in the ``info`` dictionary can provide important
  information for training.

Vectorized API
==============

The vectorized Gym API allows a single client-side environment to
control a vector of remotes. The main difference with the
non-vectorized Gym API is that individual environments will
automatically reset upon reaching the end of an episode. (An episode
is defined as ending when an agent has concretely succeeded or failed
at the task, such as after clearing a level of a game, or losing the
game. Some environments without clearly delineated success and
failure conditions may not have episodes.)

There are two API methods, ``reset`` and ``step``. The semantics are:

- ``reset`` takes no arguments and returns a vector of observations:

.. code:: python

  observation_n = env.reset()

- ``step`` consumes a vector of actions, and returns a vector of
  observations, vector of rewards, vector of done booleans, and an
  info dictionary. The info dictionary has an ``n`` key, which
  contains a vector of infos specific to each env:

.. code:: python

  observation_n, reward_n, done_n, info = env.step(action_n)
  # len(info['n']) == len(observation_n)

Some important notes:

- At any given moment, some of the environments may be
  resetting. Resetting environments will have a ``None`` value for
  their observation. For example, an ``observation_n`` of ``[None,
  {'vision': ...}, {'vision': ...}]`` indicates that the environment
  at index 0 is resetting.
- When an index returns ``done=True``, the corresponding environment
  will automatically start resetting.
- The user must call ``reset`` once before calling ``step``; undefined
  behavior will result if ``reset`` is not called. Further ``reset``
  calls are allowed, but generally are used only if the environment has
  been idle for a while (such as with periodic evaluation), or when it
  is important to start at the beginning 

Versioning
==========

The remote is versioned and has fixed semantics, assuming sufficient
compute resources are applied (i.e. if you don't have enough CPU, your
flash environments will likely behave differently). The client's exact
semantics will depend on the version of universe you have installed,
and you should track the version of that together with the rest of
your agent code.

