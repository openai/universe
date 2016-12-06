Universe internal communication protocols
*****************************************

Network architecture
====================

A Universe environment consists of two components that run in
separate processes and communicate over the network.  The agent's
machine runs the environment **client** code, which connects
to the **remote** environment server.

Many related environments can be served from the same **runtime**,
usually packaged as a Docker container. For example, all the flash
games in Universe are served from the ``flashgames`` runtime, which
consists of the ``quay.io/openai/universe.flashgames`` Docker image
and runs with its corresponding set of configuration flags.

Each remote exposes two ports:

- A VNC port (5900 by default). The remote runs an off-the-shelf VNC
  server (usually TigerVNC), so that users can connect their own
  VNC viewers to the environments for interactive use. VNC delivers
  pixel observations to the agent, and the agent submits keyboard and
  mouse inputs over VNC as well.

- A rewarder port (15900 by default). The rewarder protocol is a
  bi-directional JSON protocol runs over WebSockets. The rewarder
  channel provides more than just a reward signal; in addition, it allows the
  agent to submit control commands (such as to indicating which of
  the available environments should be active for a given runtime) and
  to receive structured information from the environment (such as latencies
  and performance timings).

VNC system and Remote Frame Buffer protocol
===========================================
 
Keyboard and mouse actions and pixel observations are sent between the
client and the remote using the `VNC
<https://en.wikipedia.org/wiki/Virtual_Network_Computing>`__
system. VNC is a pervasive standard for remote desktop operation. Many
implementations of VNC are available online, including VNC viewers
that make it easy to observe a running agent.

To achieve the performance we needed in order to train an
agent on dozens of simultaneous remote environments at 60FPS, we wrote a
`custom client-side VNC driver <https://github.com/openai/go-vncdriver>`__
in go. The remote VNC server that we use in most of our runtimes is `TigerVNC <http://tigervnc.org/>`__

More information about the Remote Frame Buffer protocol can be found
in the official `IETF RFC <https://tools.ietf.org/html/rfc6143>`__
spec, and in other tutorials elsewhere on the internet.

Rewarder protocol
=================

The Rewarder protocol is a Universe-specific bi-directional JSON
protocol run over WebSockets. In addition to the actions and
observations provided by the VNC connection, the rewarder connection
allows the agent to submit control commands to the environment, and to
receive rewards and other informational messages. This section details
the format of the Rewarder protocol.

Message format
--------------

Each message is serialized as a JSON object with the following
structure:

.. code::
		  
    {
      "method": [string],
      "headers": [object],
      "body": [object]
    }

For example, a ``v0.env.describe`` message might look as follows:

.. code::

    {
      "method": "v0.env.describe",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
        "episode_id": "1.2",
      },
      "body": {
        "env_id": "internet.SlitherIO-v0",
        "env_state": "running",
        "fps": 60
      }
    }


Each message should have a unique ``message_id`` header and a ``sent_at``
header (which should be the current UNIX timestamp with at least
millisecond precision).

Server to client messages
-------------------------

env.describe
~~~~~~~~~~~~

.. code:: 
		  
    {
      "method": "v0.env.describe",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
        "episode_id": "1.2",
      },
      "body": {
        "env_id": "internet.SlitherIO-v0",
        "env_state": "running",
      }
    }

env.reward
~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.env.reward",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
        "episode_id": "1.2",
      },
      "body": {
        "reward": -3,
        "done": False,
    	"info": {},
      }
    }

env.text
~~~~~~~~

.. code::
		  
    {
      "method": "v0.env.text",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
        "episode_id": "1.2",
      },
      "body": {
        "text": "this is some text"
      }
    }

env.observation
~~~~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.env.observation",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
        "episode_id": "1.2"
      },
      "body": {
        "observation": [0.12, 0.51, 2, 12]
      }
    }

connection.close
~~~~~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.connection.close",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15
      },
      "body": {
        "message": "Disconnected since time limit reached"
      }
    }

reply.error
~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.reply.error",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
    	"parent_message_id": "26"
      },
      "body": {
        "message": "No such environment: llama"
      }
    }

reply.env.reset
~~~~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.reply.env.reset",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
    	"parent_message_id": "26",
    	"episode_id": "1.2"
    	
      },
      "body": {}
    }
    
reply.control.ping
~~~~~~~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.reply.control.ping",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15,
    	"parent_message_id": "26"
      },
      "body": {}
    }

Client to server messages
-------------------------

agent.action
~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.agent.action",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15
      },
      "body": {
        "action: [["JoystickAxisXEvent", 0.1],
                  ["JoystickAxisZEvent", 0.1]]
      }
    }

env.reset
~~~~~~~~~

.. code::
		  
    {
      "method": "v0.env.reset",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15
      },
      "body": {
        "env_id': "flashgames.DuskDrive-v0"
      }
    }

control.ping
~~~~~~~~~~~~

.. code::
		  
    {
      "method": "v0.control.ping",
      "headers": {
        "sent_at": 1479493678.1937322617,
        "message_id": 15
      },
      "body": {}
    }
