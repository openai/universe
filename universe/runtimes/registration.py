import collections
import json

import six
from gym import error


class UnregisteredRuntime(error.Unregistered):
    """Raised when the user requests a runtime from the registry that does
    not actually exist.
    """
    pass

class DockerRuntime(object):
    """Lightweight struct for our DockerImage configuration"""
    def __init__(self, id=id, image=None, command=None, host_config=None, default_params=None, server_registry_file=None):
        """
        Args:
            id: The short identifier for this runtime
            image: The full docker image name including a tag
            command: A list of commands to be passed to docker
            host_config: A dict that will be fed to docker.Client().create_host_config
            default_params: The default parameter values for this environment
            server_registry: A file containing a JSON dump of the server registry. The format will be runtime-specific.
        """
        self.id = id
        self.image = image
        self.command = command or []
        self.host_config = host_config or {}
        self.default_params = default_params or {}

        self._server_registry = None
        self._server_registry_file = server_registry_file

    @property
    def server_registry(self):
        if self._server_registry is None:
            with open(self._server_registry_file) as f:
                self._server_registry = json.load(f)
        return self._server_registry

    @property
    def _cli_flags(self):
        # Not everything maps in a straightforward way, e.g. cap_add => '--cap-add' but ipc_mode => '--ipc
        api_to_cli = {
            'ipc_mode': 'ipc'
        }

        cli_flags = []
        for api_key, api_value in self.host_config.items():
            if isinstance(api_value, (six.string_types, bool)):
                cli_values = [api_value]
            else:
                cli_values = api_value

            for cli_value in cli_values:
                if api_key in api_to_cli:
                    api_key = api_to_cli[api_key]
                cli_flag = '--{}'.format(api_key.replace('_', '-'))
                if isinstance(cli_value, bool):
                    # boolean flag, like --privileged
                    cli_flags += [cli_flag]
                else:
                    cli_flags += [cli_flag, cli_value]

        return cli_flags

    def cli_command(self, vnc_port, rewarder_port, extra_flags=[]):
        return ['docker', 'run',
           '-p', '{}:5900'.format(vnc_port),
           '-p', '{}:15900'.format(rewarder_port)] + \
           extra_flags + \
           self._cli_flags + \
           [self.image] + self.command


class WindowsRuntime(object):
    # TODO: Spawn windows runtimes (right now managed manually)
    def __init__(self, id=id, default_params=None):
        """
        Args:
            id: The short identifier for this runtime
        """
        self.id = id
        self.default_params = default_params


class Registry(object):
    def __init__(self):
        self.runtimes = collections.OrderedDict()

    def register_runtime(self, id, kind, **kwargs):
        if kind == "docker":
            self.runtimes[id] = DockerRuntime(id, **kwargs)
        elif kind == "windows":
            self.runtimes[id] = WindowsRuntime(id, **kwargs)
        else:
            raise error.Error("No runtime of kind {} . \n Valid options are ['docker']".format(kind))

    def runtime_spec(self, id):
        """
        id is a string describing the runtime, e.g 'flashgames

        Returns a configured DockerRuntime object
        """
        try:
            return self.runtimes[id]
        except KeyError:
            raise UnregisteredRuntime('No registered runtime with name: {}'.format(id))


registry = Registry()
register_runtime = registry.register_runtime
runtime_spec = registry.runtime_spec
