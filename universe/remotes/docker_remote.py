from __future__ import absolute_import

import base64
import logging
import os
import pipes
import sys
import threading
import uuid

import docker
import six.moves.urllib.parse as urlparse
from gym.utils import closer
from universe import error
from universe.remotes import healthcheck, remote
from universe import error, utils
from universe.remotes.compose import container, log_printer, progress_stream

logger = logging.getLogger(__name__)

docker_closer = closer.Closer()

def random_alphanumeric(length=14):
    buf = []
    while len(buf) < length:
        entropy = base64.encodestring(uuid.uuid4().bytes).decode('utf-8')
        bytes = [c for c in entropy if c.isalnum()]
        buf += bytes
    return ''.join(buf)[:length]

def pretty_command(command):
    return ' '.join(pipes.quote(c) for c in command)

class DockerManager(object):
    def __init__(self, runtime, n, reuse=False, start_timeout=None):
        super(DockerManager, self).__init__()

        self.runtime = runtime

        self.supports_reconnect = False
        self.connect_vnc = True
        self.connect_rewarder = True
        self._assigner = PortAssigner(reuse=reuse)

        self._popped = False

        self.lock = threading.Lock()
        self.envs = []

        self._n = n
        if start_timeout is None:
            start_timeout = 2 * self._n + 5
        self.start_timeout = start_timeout
        self._start()

    def allocate(self, handles, initial=False, params={}):
        self._handles = handles

    def pop(self, n=None):
        """Call from main thread. Returns the list of newly-available (handle, env) pairs."""
        if self._popped:
            assert n is None
            return []
        self._popped = True

        envs = []
        for i, instance in enumerate(self.instances):
            env = remote.Remote(
                handle=self._handles[i],
                vnc_address='{}:{}'.format(instance.host, instance.vnc_port),
                vnc_password='openai',
                rewarder_address='{}:{}'.format(instance.host, instance.rewarder_port),
                rewarder_password='openai',
            )
            envs.append(env)
        return envs

    def _start(self):
        self.instances = [DockerInstance(self._assigner, self.runtime, label=str(i)) for i in range(self._n)]

        [instance.start() for instance in self.instances]
        self.start_logging(self.instances)
        self.healthcheck(self.instances)

    def close(self):
        with self.lock:
            [instance.close() for instance in self.instances]

    def start_logging(self, instances):
        containers = [instance._container for instance in instances]
        labels = [str(instance.label) for instance in instances]
        if all(instance.reusing for instance in instances):
            # All containers are being reused, so only bother showing
            # a subset of the backlog.
            tail = 0
        else:
            # At least one container is new, so just show
            # everything. It'd be nice to have finer-grained control,
            # but this would require patching the log printer.
            tail = 'all'
        log_printer.build(containers, labels, log_args={'tail': tail})

    def healthcheck(self, instances):
        # Wait for boot
        healthcheck.run(
            ['{}:{}'.format(instance.assigner.info['host'], instance.vnc_port) for instance in instances],
            ['{}:{}'.format(instance.assigner.info['host'], instance.rewarder_port) for instance in instances],
            start_timeout=30,
        )

def get_client():
    info = {}
    host = os.environ.get('DOCKER_HOST')

    client_api_version = os.environ.get('DOCKER_API_VERSION')
    if not client_api_version:
        client_api_version = "auto"

    # IP to use for started containers
    if host:
        info['host'] = urlparse.urlparse(host).netloc.split(':')[0]
    else:
        info['host'] = 'localhost'

    verify = os.environ.get('DOCKER_TLS_VERIFY') == '1'
    if verify: # use TLS
        assert_hostname = None
        cert_path = os.environ.get('DOCKER_CERT_PATH')
        if cert_path:
            client_cert = (os.path.join(cert_path, 'cert.pem'), os.path.join(cert_path, 'key.pem'))
            ca_cert = os.path.join(cert_path, 'ca.pem')
        else:
            client_cert = ca_cert = None

        tls_config = docker.tls.TLSConfig(
            client_cert=client_cert,
            ca_cert=ca_cert,
            verify=verify,
            assert_hostname=assert_hostname,
        )
        return docker.Client(base_url=host, tls=tls_config, version=client_api_version), info
    else:
        return docker.Client(base_url=host, version=client_api_version), info

class PortAssigner(object):
    def __init__(self, reuse=False):
        self.reuse = reuse
        self.instance_id = 'universe-' + random_alphanumeric(length=6)
        self.client, self.info = get_client()
        self._refresh_ports()

    def _refresh_ports(self):
        ports = {}
        for container in self.client.containers():
            for port in container['Ports']:
                # {u'IP': u'0.0.0.0', u'Type': u'tcp', u'PublicPort': 5000, u'PrivatePort': 500}
                if port['Type'] == 'tcp' and 'PublicPort' in port:
                    ports[port['PublicPort']] = container['Id']
        self._ports = ports
        self._next_port = 5900

    def allocate_ports(self, num):
        if self.reuse and self._next_port in self._ports:
            vnc_id = self._ports[self._next_port]
            rewarder_id = self._ports.get(self._next_port+10000)

            # Reuse an existing docker container if it exists
            if (self._next_port+10000) not in self._ports:
                raise error.Error("Port {} was allocated but {} was not. This indicates unexpected state with spun-up VNC docker instances.".format(self._next_port, self._next_port+1))
            elif vnc_id != rewarder_id:
                raise error.Error("Port {} is exposed from {} while {} is exposed from {}. Both should come from a single Docker instance running your environment.".format(vnc_id, self._next_port, rewarder_id, self._next_port+10000))

            base = self._next_port
            self._next_port += 1
            return base, base+10000, vnc_id
        elif not self.reuse:
            # Otherwise, allocate find the lowest free pair of
            # ports. This doesn't work for the reuse case since on
            # restart we won't remember where we spun up our
            # containers.
            while self._next_port in self._ports or (self._next_port+10000) in self._ports:
                self._next_port += 1

        base = self._next_port
        self._next_port += 1

        # And get started!
        return base, base+10000, None

class DockerInstance(object):
    def __init__(self, assigner, runtime, label='main'):
        self._docker_closer_id = docker_closer.register(self)

        self.label = label
        self.assigner = assigner
        self.name='{}-{}'.format(self.assigner.instance_id, self.label),

        self.runtime = runtime

        self._container_id = None
        self._closed = False

        self._container = None

        self.host = self.assigner.info['host']
        self.vnc_port = None
        self.rewarder_port = None
        self.reusing = None
        self.started = False

    def start(self, attempts=None):
        if attempts is None:
            # If we're reusing, we don't scan through ports for a free
            # one.
            if not self.assigner.reuse:
                attempts = 5
            else:
                attempts = 1

        for attempt in range(attempts):
            self._spawn()
            e = self._start()
            if e is None:
                return
        raise error.Error('[{}] Could not start container after {} attempts. Last error: {}'.format(self.label, attempts, e))

    def _spawn(self):
        if self.runtime.image is None:
            raise error.Error('No image specified')
        assert self._container_id is None

        self.vnc_port, self.rewarder_port, self._container_id = self.assigner.allocate_ports(2)
        if self._container_id is not None:
            logger.info('[%s] Reusing container %s on ports %s and %s', self.label, self._container_id[:12], self.vnc_port, self.rewarder_port)
            self.reusing = True
            self.started = True
            return

        self.reusing = False
        logger.info('[%s] Creating container: image=%s. Run the same thing by hand as: %s',
                    self.label,
                    self.runtime.image,
                    pretty_command(self.runtime.cli_command(self.vnc_port, self.rewarder_port)))
        try:
            container = self._spawn_container()
        except docker.errors.NotFound as e:
            # Looks like we need to pull the image
            assert 'No such image' in e.explanation.decode('utf-8'), 'Expected NotFound error message message to include "No such image", but it was: {}. This is probably just a bug in this assertion and the assumption was incorrect'.format(e.explanation)

            logger.info('Image %s not present locally; pulling', self.runtime.image)
            self._pull_image()
            # Try spawning again
            container = self._spawn_container()

        self._container_id = container['Id']

    def _pull_image(self):
        output = self.client.pull(self.runtime.image, stream=True)
        return progress_stream.get_digest_from_pull(
            progress_stream.stream_output(output, sys.stdout))

        # docker-compose uses this:
        # try:
        # except StreamOutputError as e:
        #     if not ignore_pull_failures:
        #         raise
        #     else:
        #         log.error(six.text_type(e))

    def _spawn_container(self):
        # launch instance, and refresh if error
        container = self.client.create_container(
            image=self.runtime.image,
            command=self.runtime.command,
            # environment=self.runtime.environment,
            name=self.name,
            host_config=self.client.create_host_config(
                port_bindings={
                    5900: self.vnc_port,
                    15900: self.rewarder_port,
                 },
                **self.runtime.host_config),
            labels={
                'com.openai.automanaged': 'true',
            }
        )
        return container

    def _start(self):
        # Need to start up the container!
        if not self.started:
            logger.debug('[%s] Starting container: id=%s', self.label, self._container_id)
            try:
                self.client.start(container=self._container_id)
            except docker.errors.APIError as e:
                if 'port is already allocated' in str(e.explanation):
                    logger.info('[%s] Could not start container: %s', self.label, e)
                    self._remove()
                    return e
                else:
                    raise
            else:
                self.started = True

        self._container = container.Container.from_id(self.client, self._container_id)
        return None

    def _remove(self):
        logger.info("Killing and removing container: id=%s. (If this command errors, you can always kill all automanaged environments on this Docker daemon via: docker rm -f $(docker ps -q -a -f 'label=com.openai.automanaged=true')", self._container_id)
        self.client.remove_container(container=self._container_id, force=True)
        self._container_id = None

    def __del__(self):
        self.close()

    def close(self):
        if self._closed:
            return

        docker_closer.unregister(self._docker_closer_id)

        # Make sure 1. we were the onse who started it, 2. it's
        # actually been started, and 3. we're meant to kill it.
        if self._container_id and not self.assigner.reuse:
            self._remove()

        self._closed = True

    @property
    def client(self):
        return self.assigner.client

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    from universe.runtimes import registration

    # docker run --name test --rm -ti -p 5900:5900 -p 15900:15900 quay.io/openai/universe.gym-core
    instance = DockerManager(
        runtime=registration.runtime_spec('gym-core'),
        n=2,
    )
    instance.start()
    import ipdb;ipdb.set_trace()
