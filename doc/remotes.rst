Remotes
*******

Since the remote part of the environment runs in its own server
process, managing remotes is an important task. The remote can run
anywhere - locally, or in the cloud. This section will explain
three ways to set up remotes.

.. contents:: **Contents of this document**
   :depth: 2

Docker installation
===================

The majority of the remotes for Universe environments run inside
Docker containers, so the first step to running your own remotes is
to `install Docker <https://docs.docker.com/engine/installation/>`__ (on
OSX, we recommend `Docker for Mac
<https://docs.docker.com/docker-for-mac/>`__). You should be able to
run ``docker ps`` and get something like this:

.. code:: shell

     $ docker ps
     CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES

	 
How to start a remote
=====================
	 
There are currently three ways to start a remote:

- Create an **automatic local remote** using ``env.configure(remotes=1)``.
  In this case, ``universe`` automatically creates a remote locally by spinning
  up a docker container for you.
  
- Create a **manual remote** by spinning up your own Docker container,
  locally or on a server you control.
  
- Create a **starter cluster** in AWS, which will automatically provide you
  with cloud-hosted remotes.


Automatic local remotes
-----------------------

To create an **automatic local remote**, call
``env.configure(remotes=1)`` (or ``4`` if you'd like 4 remotes).
This will download the ``quay.io/openai/universe.flashgames`` docker
container and start 1 copy of it locally.

.. code:: python

    import gym
    import universe # register the universe environments

    env = gym.make('flashgames.DuskDrive-v0')
    env.configure(remotes=1) # downloads and starts a flashgames runtime
    observation_n = env.reset()

    while True:
            action_n = [[('KeyEvent', 'ArrowUp', True)] for ob in observation_n] # your agent here
            observation_n, reward_n, done_n, info = env.step(action_n)
            env.render()


Manual remotes
--------------

To create a **manual remote**, start the remote Docker container
manually on the command line. Remotes can run locally on the same machine as
the client, or you can start them on servers you control.

To find the appropriate Docker command-line invocation for each
environment, you can look at where we `register
<https://github.com/openai/universe/blob/master/universe/runtimes/__init__.py>`__
the runtime for each environment. The command is also printed out
conveniently when you run with ``remotes=1``:

.. code:: shell

    [2016-11-25 23:51:04,223] [0] Creating container:
	image=quay.io/openai/universe.flashgames:0.19.19. Run the same thing by hand as:
	docker run -p 10000:5900 -p 10001:15900 --cap-add NET_ADMIN --cap-add SYS_ADMIN
	--ipc host quay.io/openai/universe.flashgames:0.19.19

Once you have started the docker container, configure your agent to
  connect to the VNC server (port 5900 by default) and the reward/info channel
  (port 15900 by default):

.. code:: python

    env.configure(remotes='vnc://localhost:5900+15900')

To connect manually to multiple remotes, separate them by commas:

.. code:: python

    env.configure(remotes='vnc://localhost:5900+15900,vnc://localhost:5901+15901')

If your docker container is running on a server rather than on localhost,
just plug in the appropriate URL or IP address:

.. code:: python

    env.configure(remotes='vnc://your.host.here:5900+15900')

Automatic cloud-hosted remotes: starter cluster
-----------------------------------------------

If you have an AWS account, you can spin up a **starter Docker cluster** to host your own remotes. First click the "Launch Stack" button and follow the steps on the AWS console to deploy your cluster.

  .. image:: https://s3.amazonaws.com/cloudformation-examples/cloudformation-launch-stack.png
     :target: https://console.aws.amazon.com/cloudformation/home#/stacks/new?stackName=OpenAI-Universe&templateURL=thttps://s3-us-west-2.amazonaws.com/openai-public/universe/starter-cluster-cf-0.1.0.json

Once your stack on AWS is ready, run `starter-cluster` to start your environments

  .. code:: shell

    $ example/starter-cluster/starter-cluster start -s [stack-name] -i [path-to-ssh-key] \
        --runtime [universe-runtime] -n [number-of-envs]

or example, the follow will start two flashgames remotes:

  .. code:: shell
			
    $ pip install -r bin/starter-cluster-requirements.txt
    $ bin/starter-cluster -v start -s OpenAI-Universe -i my-ec2-key.pem -r flashgames -n 2
    Creating network "flashgames_default" with the default driver
    Pulling flashgames-0 (quay.io/openai/universe.flashgames:0.19.36)...
    ip-172-33-1-4: Pulling quay.io/openai/universe.flashgames:0.19.36... : downloaded
    ip-172-33-28-242: Pulling quay.io/openai/universe.flashgames:0.19.36... : downloaded
    Creating flashgames_flashgames-0_1
    Creating flashgames_flashgames-1_1
    Environments started.
    Remotes:
      vnc://54.245.154.123:5013+5015
      vnc://54.245.154.123:5006+5008

Now you can pass the IP address and ports for your remotes to your agent,
as was described in the previous section on manual remotes. For example:

  .. code:: shell
			
    $ python bin/random_agent.py -e flashgames.DuskDrive-v0 -r vnc://54.245.154.123:5013+5015,54.245.154.123:5006+5008

Running ``bin/starter-cluster start`` again will restart your remotes. To stop them, run:

  .. code:: shell
			
    $ bin/starter-cluster stop -s OpenAI-Universe -i my-ec2-key.pem -r flashgames
    Stopping flashgames_flashgames-1_1 ... done
    Stopping flashgames_flashgames-0_1 ... done
    Removing flashgames_flashgames-1_1 ... done
    Removing flashgames_flashgames-0_1 ... done
    Removing network flashgames_default
    Environments stopped.

Region
~~~~~~

By default, starter cluster remotes are spawned in AWS's ``us-west-2``
region. In our experience, the latencies of training over the public
internet are acceptable, but if you have trouble, it may make sense to
try running your agent code on an AWS server in the same region as the
remote.

Scaling Up
~~~~~~~~~~

If you encounter the following

.. code:: shell
   
  $ bin/starter-cluster -v start -s OpenAI-Universe -i my-ec2-key.pem -r flashgames   -n 2
    Creating network "flashgames_default" with the default driver
    Pulling flashgames-0 (quay.io/openai/universe.flashgames:0.19.36)...
    ip-172-33-1-4: Pulling quay.io/openai/universe.flashgames:0.19.36... : downloaded
    ip-172-33-28-242: Pulling quay.io/openai/universe.flashgames:0.19.36... :   downloaded
    ip-172-33-9-51: Pulling quay.io/openai/universe.flashgames:0.19.36... :   downloaded
    ip-172-33-27-141: Pulling quay.io/openai/universe.flashgames:0.19.36... :   downloaded
    Creating flashgames_flashgames-2_1
    Creating flashgames_flashgames-3_1
    Creating flashgames_flashgames-0_1
    Creating flashgames_flashgames-1_1
    Creating flashgames_flashgames-4_1

    ERROR: for flashgames-0  no resources available to schedule container

then it means you've run out of computing resources on your cluster, and
have to add more worker nodes. You can do so by going to the AWS
Cloudformation console:

1. Select your stack
2. Click "Update Stack" in the "Actions" dropdown
3. Hit "Next" on the "Select Template" page
4. Input the new swarm size and hit "Next"
5. Hit "Next" on the "Options" page
6. Hit "Update" on the "Review" page


Reusing remotes
===============

If a consistent ``client_id`` is supplied to ``configure()``, then the
client will attempt to reuse the same remote for the new environment
rather than spinning up a new one each time.

Switching between environments in the same *runtime*
(i.e. environments that run on the same underlying docker container)
is possible without creating a new remote; however, if you want to
switch to an environment in a different runtime, you will need to create
a new remote. For example, you can switch between
``flashgames.DuskDrive-v0`` and ``flashgames.NeonRace-v0`` without
starting a new remote, because they both run in the ``flashgames``
runtime, but if you want to switch to ``wob.mini.UseColorwheel2-v0``
you cannot re-use the same remote.

The configuration for the runtimes is defined in
`universe/runtimes/__init__.py <https://github.com/openai/universe/blob/master/universe/runtimes/__init__.py>`__,
and the specific version number tags for the corresponding Docker
images are specified in
`runtimes.yml <https://github.com/openai/universe/blob/master/universe/runtimes.yml>`__.


