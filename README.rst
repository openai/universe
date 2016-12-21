universe
***************

.. image:: https://travis-ci.org/openai/universe.svg?branch=master
    :target: https://travis-ci.org/openai/universe

`Universe <https://openai.com/blog/universe/>`_ is a software
platform for measuring and training an AI's general intelligence
across the world's supply of games, websites and other
applications. This is the ``universe`` open-source library, which
provides a simple `Gym <https://github.com/openai/gym>`__
interface to each Universe environment.

Universe allows anyone to train and evaluate AI agents on an extremely
wide range of real-time, complex environments.

Universe makes it possible for any existing program to become an
OpenAI Gym environment, without needing special access to the
program's internals, source code, or APIs. It does this by packaging
the program into a Docker container, and presenting the AI with the
same interface a human uses: sending keyboard and mouse events, and
receiving screen pixels. Our initial release contains over 1,000
environments in which an AI agent can take actions and gather
observations.

Additionally, some environments include a reward signal sent to the
agent, to guide reinforcement learning. We've included a few hundred
environments with reward signals. These environments also include
automated start menu clickthroughs, allowing your agent to skip to the
interesting part of the environment.

We'd like the community's `help <https://openai.com/blog/universe/#help>`_
to grow the number of available environments, including integrating
increasingly large and complex games.

The following classes of tasks are packaged inside of
publicly-available Docker containers, and can be run today with no
work on your part:

- Atari and CartPole environments over VNC: ``gym-core.Pong-v3``, ``gym-core.CartPole-v0``, etc.
- Flashgames over VNC: ``flashgames.DuskDrive-v0``, etc.
- Browser tasks ("World of Bits") over VNC: ``wob.mini.TicTacToe-v0``, etc.

We've scoped out integrations for many other games, including
completing a high-quality GTA V integration (thanks to `Craig Quiter <http://deepdrive.io/>`_ and NVIDIA), but these aren't included in today's release.

.. contents:: **Contents of this document**
   :depth: 2


Getting started
===============

Installation
------------

Supported systems
~~~~~~~~~~~~~~~~~

We currently support Linux and OSX running Python 2.7 or 3.5.

We recommend setting up a `conda environment <http://conda.pydata.org/docs/using/envs.html>`__
before getting started, to keep all your Universe-related packages in the same place.

Install Universe
~~~~~~~~~~~~~~~~
To get started, first install ``universe``:

.. code:: shell

    git clone https://github.com/openai/universe.git
    cd universe
    pip install -e .

If this errors out, you may be missing some required packages. Here's
the list of required packages we know about so far (please let us know
if you had to install any others).

On Ubuntu 16.04:

.. code:: shell

    pip install numpy
    sudo apt-get install golang libjpeg-turbo8-dev make

On Ubuntu 14.04:

.. code:: shell

    sudo add-apt-repository ppa:ubuntu-lxc/lxd-stable  # for newer golang
    sudo apt-get update
    sudo apt-get install golang libjpeg-turbo8-dev make

On OSX:

You might need to install Command Line Tools by running:

.. code:: shell

    xcode-select --install

Or ``numpy``, ``libjpeg-turbo`` and ``incremental`` packages:

.. code:: shell

    pip install numpy incremental
    brew install golang libjpeg-turbo

Install Docker
~~~~~~~~~~~~~~

The majority of the environments in Universe run inside Docker
containers, so you will need to `install Docker
<https://docs.docker.com/engine/installation/>`__ (on OSX, we
recommend `Docker for Mac
<https://docs.docker.com/docker-for-mac/>`__). You should be able to
run ``docker ps`` and get something like this:

.. code:: shell

     $ docker ps
     CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES

Notes on installation
~~~~~~~~~~~~~~~~~~~~~

* When installing ``universe``, you may see ``warning`` messages.  These lines occur when installing numpy and are normal.
* You'll need a ``go version`` of at least 1.5. Ubuntu 14.04 has an older Go, so you'll need to `upgrade <https://golang.org/doc/install>`_ your Go installation.
* We run Python 3.5 internally, so the Python 3.5 variants will be much more thoroughly performance tested. Please let us know if you see any issues on 2.7.
* While we don't officially support Windows, we expect our code to be very close to working there. We'd be happy to take pull requests that take our Windows compatibility to 100%.

System overview
---------------

A Universe **environment** is similar to any other Gym environment:
the agent submits actions and receives observations using the ``step()``
method.

Internally, a Universe environment consists of two pieces: a **client** and a **remote**:

* The **client** is a `VNCEnv
  <https://github.com/openai/universe/blob/master/universe/envs/vnc_env.py>`_
  instance which lives in the same process as the agent. It performs
  functions like receiving the agent's actions, proxying them to the
  **remote**, queuing up rewards for the agent, and maintaining a
  local view of the current episode state.
* The **remote** is the running environment dynamics, usually a
  program running inside of a Docker container. It can run anywhere --
  locally, on a remote server, or in the cloud. (We have a separate
  page describing how to manage `remotes <doc/remotes.rst>`__.)
* The client and the remote communicate with one another using the
  `VNC <https://en.wikipedia.org/wiki/Virtual_Network_Computing>`__
  remote desktop system, as well as over an auxiliary WebSocket
  channel for reward, diagnostic, and control messages. (For more
  information on client-remote communication, see the separate page on
  the `Universe internal communication protocols
  <doc/protocols.rst>`__.)

The code in this repository corresponds to the **client** side of the
Universe environments. Additionally, you can freely access the Docker
images for the **remotes**. We'll release the source repositories for
the remotes in the future, along with tools to enable users to
integrate new environments. Please sign up for our `beta
<https://docs.google.com/forms/d/e/1FAIpQLScAiW-kIS0mz6hdzzFZJJFlXlicDvQs1TX9XMEkipNwjV5VlA/viewform>`_
if you'd like early access.

Run your first agent
--------------------

Now that you've installed the ``universe`` library, you should make
sure it actually works. You can paste the example below into your
``python`` REPL. (You may need to press enter an extra time to make
sure the ``while`` loop is executing.)

.. code:: python

  import gym
  import universe  # register the universe environments

  env = gym.make('flashgames.DuskDrive-v0')
  env.configure(remotes=1)  # automatically creates a local docker container
  observation_n = env.reset()

  while True:
    action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]  # your agent here
    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()

The example will instantiate a client in your Python process,
automatically pull the ``quay.io/openai/universe.flashgames`` image,
and will start that image as the remote. (In our `remotes
<doc/remotes.rst>`__ documentation page, we explain other ways you can run
remotes.)

It will take a few minutes for the image to pull the first time. After that,
if all goes well, a window like the one below will soon pop up. Your
agent, which is just pressing the up arrow repeatedly, is now
playing a Flash racing game called `Dusk Drive
<http://www.kongregate.com/games/longanimals/dusk-drive>`__. Your agent
is programmatically controlling a VNC client, connected to a VNC
server running inside of a Docker container in the cloud, rendering a
headless Chrome with Flash enabled:

.. image:: https://github.com/openai/universe/blob/master/doc/dusk-drive.png?raw=true
     :width: 600px

You can even connect your own VNC client to the environment, either
just to observe or to interfere with your agent. Our ``flashgames``
and ``gym-core`` images conveniently bundle a browser-based VNC
client, which can be accessed at
``http://localhost:15900/viewer/?password=openai``. If you're on Mac,
connecting to a VNC server is as easy as running: ``open
vnc://localhost:5900``.

(If using docker-machine, you'll need to replace "localhost" with the
IP address of your Docker daemon, and use ``openai`` as the password.)

Breaking down the example
~~~~~~~~~~~~~~~~~~~~~~~~~

So we managed to run an agent, what did all the code actually
mean? We'll go line-by-line through the example.

* First, we import the `gym <https://github.com/openai/gym>`__ library,
  which is the base on which Universe is built. We also import
  ``universe``, which `registers
  <https://github.com/openai/universe/blob/master/universe/__init__.py>`__
  all the Universe environments.

.. code:: python

  import gym
  import universe # register the universe environments

* Next, we create the environment instance. Behind the scenes, ``gym``
  looks up the `registration
  <https://github.com/openai/universe/blob/master/universe/__init__.py>`__
  for ``flashgames.DuskDrive-v0``, and instantiates a `VNCEnv
  <https://github.com/openai/universe/blob/master/universe/envs/vnc_env.py#L88>`__
  object which has been `wrapped
  <https://github.com/openai/universe/blob/master/universe/wrappers/__init__.py#L42>`__
  to add a few useful diagnostics and utilities. The ``VNCEnv`` object
  is the *client* part of the environment, and it is not yet connected
  to a *remote*.

.. code:: python

  env = gym.make('flashgames.DuskDrive-v0')

* The call to ``configure()`` connects the client to a remote
  environment server. When called with ``configure(remotes=1)``,
  Universe will automatically create a Docker image running locally on
  your computer. The local client connects to the remote using VNC.
  (More information on client-remote communication can be found in the
  page on `universe internal communication protocols
  <doc/protocols.rst>`__. More on configuring remotes is at `remotes <doc/remotes.rst>`__.)

.. code:: python

  env.configure(remotes=1)

* When starting a new environment, you call ``env.reset()``. Universe
  environments run in real-time, rather than stepping synchronously
  with the agent's actions, so ``reset`` is asynchronous and returns
  immediately. Since the environment will not have waited to finish
  connecting to the VNC server before returning, the initial observations
  from ``reset`` will be ``None`` to indicate that there is
  not yet a valid observation.

  Similarly, the environment keeps running in the background even
  if the agent does not call ``env.step()``.  This means that an agent
  that successfully learns from a Universe environment cannot take
  "thinking breaks":  it must keep sending actions to the environment at all times.

  Additionally, Universe introduces the *vectorized* Gym
  API. Rather than controlling a single environment at a time, the agent
  can control a fixed-size vector of ``n`` environments, each with its
  own remote. The return value from ``reset`` is therefore a *vector*
  of observations. For more information, see the separate page on
  `environment semantics <doc/env_semantics.rst>`__)

.. code:: python

  observation_n = env.reset()

* At each ``step()`` call, the agent submits a vector of actions; one for
  each environment instance it is controlling. Each VNC action is a
  list of events; above, each action is the single event "press the
  ``ArrowUp`` key". The agent could press and release the key in one
  action by instead submitting ``[('KeyEvent', 'ArrowUp', True),
  ('KeyEvent', 'ArrowUp', False)]`` for each observation.

  In fact, the agent could largely have the same effect by just
  submitting ``('KeyEvent', 'ArrowUp', True)`` once and then calling
  ``env.step([[] for ob in observation_n])`` thereafter, without ever
  releasing the key using ``('KeyEvent', 'ArrowUp', False)``. The
  browser running inside the remote would continue to statefully
  represent the arrow key as being pressed. Sending other unrelated
  keypresses would not disrupt the up arrow keypress; only explicitly
  releasing the key would cancel it.  There's one slight subtlety:
  when the episode resets, the browser will reset, and will forget
  about the keypress; you'd need to submit a new ``ArrowUp`` at the
  start of each episode.

.. code:: python

  action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n]

* After we submit the action to the environment and render one frame,
  ``step()`` returns a list of *observations*, a list of *rewards*, a
  list of *"done" booleans* indicating whether the episode has ended,
  and then finally an *info dictionary* of the form ``{'n': [{},
  ...]}``, in which you can access the info for environment ``i`` as
  ``info['n'][i]``.

  Each environment's ``info`` message contains useful diagnostic
  information, including latency data, client and remote timings,
  VNC update counts, and reward message counts.

.. code:: python

    observation_n, reward_n, done_n, info = env.step(action_n)
    env.render()

* We call ``step`` in what looks like a busy loop. In reality, there
  is a `Throttle
  <https://github.com/openai/universe/blob/master/universe/wrappers/__init__.py#L18>`__
  wrapper on the client which defaults to a target frame rate of 60fps, or one
  frame every 16.7ms. If you call it more frequently than that,
  ``step`` will `sleep
  <https://github.com/openai/universe/blob/master/universe/wrappers/throttle.py>`__
  with any leftover time.


Testing
=======

We are using `pytest <http://doc.pytest.org/en/latest/>`__ for tests. You can run them via:

.. code:: shell

    pytest

Run ``pytest --help`` for useful options, such as ``pytest -s`` (disables output capture) or ``pytest -k <expression>`` (runs only specific tests).

Additional documentation
========================

More documentation not covered in this README can be found in the
`doc folder <doc>`__ of this repository.

What's next?
============

* Get started training RL algorithms! You can try out the `Universe Starter Agent <https://github.com/openai/universe-starter-agent>`_, an implementation of the `A3C algorithm <https://arxiv.org/abs/1602.01783>`_ that can solve several VNC environments.

* For more information on how to manage remotes, see the separate documentation page on `remotes <doc/remotes.rst>`__.

* Sign up for a `beta <https://docs.google.com/forms/d/e/1FAIpQLScAiW-kIS0mz6hdzzFZJJFlXlicDvQs1TX9XMEkipNwjV5VlA/viewform>`_ to get early access to upcoming Universe releases, such as tools to integrate new Universe environments or a dataset of recorded human demonstrations.
