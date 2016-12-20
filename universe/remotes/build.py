import re
from universe import error
from universe.remotes.allocator_remote import AllocatorManager
from universe.remotes.docker_remote import DockerManager
from universe.remotes.hardcoded_addresses import HardcodedAddresses

def build(client_id, remotes, runtime=None, start_timeout=None, **kwargs):
    if isinstance(remotes, int):
        remotes = str(remotes)
    elif not isinstance(remotes, str):
        raise error.Error('remotes argument must be a string, got {} which is of type {}'.format(remotes, type(remotes)))

    if re.search('^\d+$', remotes): # an integer, like -r 20
        n = int(remotes)
        return DockerManager(
            runtime=runtime,
            start_timeout=start_timeout,
            reuse=kwargs.get('reuse', False),
            n=n,
        ), n
    elif remotes.startswith('vnc://'):
        return HardcodedAddresses.build(
            remotes,
            start_timeout=start_timeout)
    elif remotes.startswith('http://') or remotes.startswith('https://'):
        if runtime is None:
            raise error.Error('Must provide a runtime. HINT: try creating your env instance via gym.make("flashgames.DuskDrive-v0")')

        manager, n = AllocatorManager.from_remotes(
            client_id,
            remotes,
            runtime_id=runtime.id,
            tag=runtime.image.split(':')[-1],
            start_timeout=start_timeout,
            api_key=kwargs.get('api_key'),
            use_recorder_ports=kwargs.get('use_recorder_ports', False),
        )
        manager.start()
        return manager, n
    else:
        raise error.Error('Invalid remotes: {!r}. Must be an integer or must start with vnc:// or https://'.format(remotes))
