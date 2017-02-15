import getpass
import logging
import os
import random
import uuid

import universe
from gym.utils import reraise
from universe import error, pyprofile, rewarder, spaces, twisty, vectorized, vncdriver
from universe import remotes as remotes_module
from universe.envs import diagnostics
from universe.runtimes import registration
from universe.vncdriver import libvnc_session

# The Go driver is the most supported one. So long as the Go driver
# turns out to be easy to install, we'll continue forcing the Go
# driver here.
# noinspection PyUnresolvedReferences
import go_vncdriver

logger = logging.getLogger(__name__)
extra_logger = logging.getLogger('universe.extra.'+__name__)

if os.environ.get('UNIVERSE_VNCDRIVER', None) == 'go':
    # Importing the Go driver early is desirable, so that people get
    # errors if it's not present. Also sometimes if go_vncdriver is
    # loaded after TensorFlow it will crash.
    import go_vncdriver

def default_client_id():
    return '{}-{}'.format(uuid.uuid4(), getpass.getuser())

def rewarder_session(which):
    if which is None:
        which = rewarder.RewarderSession

    if isinstance(which, type):
        return which
    else:
        raise error.Error('Invalid RewarderSession driver: {!r}'.format(which))

def go_vncdriver():
    import go_vncdriver
    return go_vncdriver.VNCSession

def py_vncdriver():
    return vncdriver.VNCSession

def libvnc_vncdriver():
    return libvnc_session.LibVNCSession

def vnc_session(which=None):
    # Short circuit so long as we're forcing the Go driver. Other code
    # left behind for the future if we need to support the other
    # drivers again.
    if isinstance(which, type):
        # Used in the tests to pass a custom VNC driver
        return which

    logger.info('Using the golang VNC implementation')
    return go_vncdriver()

    if which is None:
        which = os.environ.get('UNIVERSE_VNCDRIVER')

    if isinstance(which, type):
        return which
    if which == 'go':
        logger.info('Using the golang VNC implementation')
        return go_vncdriver()
    elif which == 'py':
        logger.info('Using the Python VNC implementation')
        return py_vncdriver()
    elif which == 'libvnc':
        logger.info('Using the libvnc VNC implementation')
        return libvnc_vncdriver()
    elif which is None:
        try:
            go = go_vncdriver()
            logger.debug('Using golang VNC implementation')
            return go
        except ImportError as e:
            logger.info("Go driver failed to import: {}".format(e))
            logger.info("Using pure Python vncdriver implementation. Run 'pip install go-vncdriver' to install the more performant Go implementation. Optionally set the environment variable UNIVERSE_VNCDRIVER='go' to force its use.")
            return py_vncdriver()
    else:
        raise error.Error('Invalid VNCSession driver: {!r}'.format(which))

def build_observation_n(visual_observation_n, info_n):
    observation_n = []
    for visual, info in zip(visual_observation_n, info_n):
        text = info.pop('env.text', [])
        obs = {
            'vision': visual,
            'text': text,
        }
        if 'env.generic' in info:
            obs['generic'] = info.pop('env.generic')
        observation_n.append(obs)
    return observation_n

class VNCEnv(vectorized.Env):
    metadata = {
        'render.modes': ['human'], # we wrap with a Render which can render to rgb_array
        'semantics.async': True,
        'semantics.autoreset': True,
        'video.frames_per_second' : 60,
        'runtime.vectorized': True,
        'configure.required': True,
    }

    def __init__(self, fps=None, probe_key=None):
        self.metadata = dict(self.metadata)
        if fps is not None:
            self.metadata['video.frames_per_second'] = fps

        self._started = False

        self.observation_space = spaces.VNCObservationSpace()
        self.action_space = spaces.VNCActionSpace()

        self._seed_value = None
        self._remotes_manager = None

        self._probe_key = probe_key or 0xbeef1

        self.vnc_session = None
        self.rewarder_session = None

        self._send_actions_over_websockets = False
        self._skip_network_calibration = False


    def _seed(self, seed):
        self._seed_value = seed
        return [seed]

    def _configure(self, remotes=None,
                   client_id=None,
                   start_timeout=None, docker_image=None,
                   ignore_clock_skew=False, disable_action_probes=False,
                   vnc_driver=None, vnc_kwargs=None,
                   rewarder_driver=None,
                   replace_on_crash=False, allocate_sync=True,
                   observer=False, api_key=None,
                   record=False,
                   sample_env_ids=None,
    ):
        """Standard Gym hook to configure the environment.

        Args:

          ignore_clock_skew (bool): Assume remotes are on the same machine as us,
            for the purposes of diagnostics measurement.

            If true, we skip measuring the clock skew over the network,
            and skip generating diagnostics which rely on it.

            True when used by the rewarder to measure latency between
            the VNC frame and its calculation of reward for that
            frame.  In this case we share a common clock with the env
            generating the VNC frame, so we don't need to send/receive
            probes.  Clock skew is zero in this case.

            False when remotes are potentially different machines
            (such as an agent, or a demonstrator), and we will be
            sending probe keys and measuring network ping rountrip
            times to calculate clock skew.
        """
        if self._started:
            raise error.Error('{} has already been started; cannot change configuration now.'.format(self))

        universe.configure_logging()

        twisty.start_once()

        if self.spec is not None:
            runtime = registration.runtime_spec(self.spec.tags['runtime'])
            # Let the user manually set the docker_image version
            if docker_image:
                # TODO: don't support this option?
                runtime.image = docker_image
        else:
            runtime = None

        if remotes is None:
            remotes = os.environ.get('GYM_VNC_REMOTES', '1')

        if client_id is None:
            client_id = default_client_id()

        if vnc_kwargs is None:
            vnc_kwargs = {}

        self.remote_manager, self.n = remotes_module.build(
            client_id=client_id,
            remotes=remotes, runtime=runtime, start_timeout=start_timeout,
            api_key=api_key,
            use_recorder_ports=record,
        )
        self.connection_names = [None] * self.n
        self.connection_labels = [None] * self.n
        self.crashed = {}

        self.allow_reconnect = replace_on_crash and self.remote_manager.supports_reconnect
        if self.remote_manager.connect_vnc:
            cls = vnc_session(vnc_driver)
            vnc_kwargs.setdefault('start_timeout', self.remote_manager.start_timeout)
            if runtime == 'gym-core':
                vnc_kwargs.setdefault('encoding', 'zrle')
            else:
                vnc_kwargs.setdefault('encoding', 'tight')
                vnc_kwargs.setdefault('fine_quality_level', 50)
                vnc_kwargs.setdefault('subsample_level', 2)
            # Filter out None values, since some drivers may not handle them correctly
            vnc_kwargs = {k: v for k, v in vnc_kwargs.items() if v is not None}
            logger.info('Using VNCSession arguments: %s. (Customize by running "env.configure(vnc_kwargs={...})"', vnc_kwargs)
            self.vnc_kwargs = vnc_kwargs
            self.vnc_session = cls()
        else:
            self.vnc_session = None

        self._observer = observer
        if self.remote_manager.connect_rewarder:
            cls = rewarder_session(rewarder_driver)
            self.rewarder_session = cls()
        else:
            self.rewarder_session = None

        if ignore_clock_skew:
            logger.info('Printed stats will ignore clock skew. (This usually makes sense only when the environment and agent are on the same machine.)')

        if self.rewarder_session or ignore_clock_skew:
            # Don't need rewarder session if we're ignoring clock skew
            if self.spec is not None:
                metadata_encoding = self.spec.tags.get('metadata_encoding')
            else:
                metadata_encoding = None
            self.diagnostics = diagnostics.Diagnostics(self.n, self._probe_key, ignore_clock_skew, metadata_encoding=metadata_encoding, disable_action_probes=disable_action_probes)
        else:
            self.diagnostics = None

        self._sample_env_ids = sample_env_ids

        self._reset_mask()
        self._started = True

        self.remote_manager.allocate([str(i) for i in range(self.n)], initial=True)
        if allocate_sync:
            # Block until we've fulfilled n environments
            self._handle_connect(n=self.n)
        else:
            # Handle any backends which synchronously fufill their
            # allocation.
            self._handle_connect()

    def connect(self, i, name, vnc_address, rewarder_address, vnc_password=None, rewarder_password=None):
        logger.info('[%s] Connecting to environment: vnc://%s password=%s. If desired, you can manually connect a VNC viewer, such as TurboVNC. Most environments provide a convenient in-browser VNC client: http://%s/viewer/?password=%s', name, vnc_address, vnc_password, rewarder_address, vnc_password)

        extra_logger.info('[%s] Connecting to environment details: vnc_address=%s vnc_password=%s rewarder_address=%s rewarder_password=%s', name, vnc_address, vnc_password, rewarder_address, rewarder_password)
        self.connection_names[i] = name
        self.connection_labels[i] = '{}:{}'.format(name, vnc_address)
        if self.vnc_session is not None:
            kwargs = {
                'name': name,
                'address': vnc_address,
                'password': vnc_password,
            }
            kwargs.update(self.vnc_kwargs)

            try:
                self.vnc_session.connect(**kwargs)
            except TypeError as e:
                reraise(suffix="(HINT: this error was while passing arguments to the VNCSession driver: {})".format(kwargs))

            # TODO: name becomes index:pod_id
            # TODO: never log index, just log name

        if self.rewarder_session is not None:
            if self.spec is not None:
                env_id = self.spec.id
            else:
                env_id = None

            if self._seed_value is not None:
                # Once we use a seed, we clear it so we never
                # accidentally reuse the seed. Seeds are an advanced
                # feature and aren't supported by most envs in any
                # case.
                seed = self._seed_value[i]
                self._seed_value[i] = None
            else:
                seed = None

            assert rewarder_password, "Missing rewarder password: {}".format(rewarder_password)
            network = self.rewarder_session.connect(
                name=name, address=rewarder_address,
                seed=seed, env_id=env_id,
                fps=self.metadata['video.frames_per_second'],
                password=rewarder_password,
                label=self.connection_labels[i],
                start_timeout=self.remote_manager.start_timeout,
                observer=self._observer,
                skip_network_calibration=self._skip_network_calibration
            )
        else:
            network = None

        if self.diagnostics is not None:
            self.diagnostics.connect(i, network, label=name)
        self.crashed.pop(i, None)

    def _close(self, i=None):
        if i is not None:
            name = self.connection_names[i]
            if self.rewarder_session:
                self.rewarder_session.close(name)
            if self.vnc_session:
                self.vnc_session.close(name)
            if self.diagnostics:
                self.diagnostics.close(i)
            self.mask.close(i)
            self.connection_names[i] = None
            self.connection_labels[i] = None
        else:
            if hasattr(self, 'rewarder_session') and self.rewarder_session:
                self.rewarder_session.close()
            if hasattr(self, 'vnc_session') and self.vnc_session:
                self.vnc_session.close()
            if hasattr(self, 'diagnostics') and self.diagnostics:
                self.diagnostics.close()
            if hasattr(self, 'remotes_manager') and self._remotes_manager:
                self._remotes_manager.close()

    def _reset(self):
        self._handle_connect()

        if self.rewarder_session:
            if self._sample_env_ids:
                env_id = random.choice(self._sample_env_ids)
                logger.info("Randomly sampled env_id={}".format(env_id))
            else:
                env_id = None
            self.rewarder_session.reset(env_id=env_id)
        else:
            logger.info("No rewarder session exists, so cannot send a reset via the rewarder channel")
        self._reset_mask()
        return [None] * self.n

    def _reset_mask(self):
        self.mask = Mask(self.connection_labels, initially_masked=bool(self.rewarder_session))

    def _pop_rewarder_session(self, peek_d):
        with pyprofile.push('vnc_env.VNCEnv.rewarder_session.pop'):
            reward_d, done_d, info_d, err_d = self.rewarder_session.pop(peek_d=peek_d)

        reward_n = []
        done_n = []
        info_n = []
        err_n = []
        for name in self.connection_names:
            reward_n.append(reward_d.get(name, 0))
            done_n.append(done_d.get(name, False))
            info_n.append(info_d.get(name, {'env_status.disconnected': True}))
            err_n.append(err_d.get(name))
        return reward_n, done_n, info_n, err_n

    def _step_vnc_session(self, compiled_d):
        if self._send_actions_over_websockets:
            self.rewarder_session.send_action(compiled_d, self.spec.id)
            vnc_action_d = {}
        else:
            vnc_action_d = compiled_d

        with pyprofile.push('vnc_env.VNCEnv.vnc_session.step'):
            observation_d, info_d, err_d = self.vnc_session.step(vnc_action_d)

        observation_n = []
        info_n = []
        err_n = []
        for name in self.connection_names:
            observation_n.append(observation_d.get(name))
            info_n.append(info_d.get(name))
            err_n.append(err_d.get(name))

        return observation_n, info_n, err_n

    def _compile_actions(self, action_n):
        compiled_n = []
        peek_d = {}
        try:
            for i, action in enumerate(action_n):
                compiled = []
                compiled_n.append(compiled)
                for event in action:
                    # Handle any special control actions
                    if event == spaces.PeekReward:
                        name = self.connection_names[i]
                        peek_d[name] = True
                        continue

                    # Do a generic compile
                    compiled.append(compile_action(event))
        except Exception as e:
            raise error.Error('Could not compile actions. Original error: {} ({}). action_n={}'.format(e, type(e), action_n))
        else:
            return compiled_n, peek_d

    def _action_d(self, action_n):
        action_d = {}
        for i, action in enumerate(action_n):
            action_d[self.connection_names[i]] = action
        return action_d

    def _step(self, action_n):
        self._handle_connect()

        # Compile actions to something more palatable by drivers
        # written in other language.
        action_n, peek_d = self._compile_actions(action_n)

        # We pop from the rewarder session first since we need to
        # determine if any of the current VNC actions need to be
        # masked. (If the environment is resetting, we should
        # definitely not send it any actions.)
        #
        # It's a bit counterintuitive to check for rewards first,
        # since we haven't submitted the actions yet, but keep in mind
        # that everything here is asynchronous!
        if self.rewarder_session:
            reward_n, done_n, info_n, err_n = self._pop_rewarder_session(peek_d)
        else:
            reward_n = done_n = [None] * self.n
            info_n = [{} for _ in range(self.n)]
            err_n = [None] * self.n

        action_mask = observation_mask = self.mask.maintain_mask(done_n, info_n)

        # Drop any actions to resetting environments.
        #
        # We pass the mask to add_probe so it doesn't try to schedule
        # probes which are potentially harmful and will just time out
        # anyway.
        self.mask.apply_to_actions(action_n, info_n, action_mask)
        # Send our actions and return the current framebuffer
        if self.vnc_session:
            if self.diagnostics:
                self.diagnostics.clear_probes_when_done(done_n)
                self.diagnostics.add_probe(action_n, action_mask)
            action_d = self._action_d(action_n)

            visual_observation_n, obs_info_n, vnc_err_n = self._step_vnc_session(action_d)
            # Merge in any keys from the observation
            self._propagate_obs_info(info_n, obs_info_n)
        else:
            visual_observation_n = [None] * self.n
            vnc_err_n = [None] * self.n

        observation_n = build_observation_n(visual_observation_n, info_n)
        self.mask.apply_to_return(observation_n, reward_n, done_n, info_n, observation_mask)

        self._handle_initial_n(observation_n, reward_n)
        self._handle_err_n(err_n, vnc_err_n, info_n, observation_n, reward_n, done_n)
        self._handle_crashed_n(info_n)

        return observation_n, reward_n, done_n, {'n': info_n}

    def _handle_initial_n(self, observation_n, reward_n):
        if self.rewarder_session is None:
            return

        for i, reward in enumerate(reward_n):
            if reward is None:
                # Index hasn't come up yet, so ensure the observation
                # is blanked out.
                observation_n[i] = None

    def _handle_err_n(self, err_n, vnc_err_n, info_n, observation_n=None, reward_n=None, done_n=None):
        # Propogate any errors upwards.
        for i, (err, vnc_err) in enumerate(zip(err_n, vnc_err_n)):
            if err is None and vnc_err is None:
                # All's well at this index.
                continue

            if observation_n is not None:
                observation_n[i] = None
                done_n[i] = True

            # Propagate the error
            if err is not None and vnc_err is not None:
                # Both the rewarder and vnc failed at the same
                # time. What are the odds?
                info_n[i]['error'] = 'Both VNC and rewarder sessions failed: {}; {}'.format(vnc_err, err)
            elif err is not None:
                info_n[i]['error'] = 'Rewarder session failed: {}'.format(err)
            else:
                info_n[i]['error'] = 'VNC session failed: {}'.format(vnc_err)

            extra_logger.info('[%s] %s', self.connection_names[i], info_n[i]['error'])

            if self.allow_reconnect:
                logger.info('[%s] Making a call to the allocator to replace crashed index: %s', self.connection_names[i], info_n[i]['error'])
                self.remote_manager.allocate([str(i)])

            self.crashed[i] = self.connection_names[i]
            self._close(i)

    def _handle_connect(self, n=None):
        # Connect to any environments which are ready
        for remote in self.remote_manager.pop(n=n):
            if remote.name is not None:
                name = '{}:{}'.format(remote.handle, remote.name)
            else:
                name = remote.handle

            self.connect(
                int(remote.handle), name=name,
                vnc_address=remote.vnc_address, vnc_password=remote.vnc_password,
                rewarder_address=remote.rewarder_address, rewarder_password=remote.rewarder_password)

    def _handle_crashed_n(self, info_n):
        # for i in self.crashed:
        #     info_n[i]['env_status.crashed'] = True

        if self.allow_reconnect:
            return

        if len(self.crashed) > 0:
            errors = {}
            for i, info in enumerate(info_n):
                if 'error' in info:
                    errors[self.crashed[i]] = info['error']

            if len(errors) == 0:
                raise error.Error('{}/{} environments have crashed. No error key in info_n: {}'.format(len(self.crashed), self.n, info_n))
            else:
                raise error.Error('{}/{} environments have crashed! Most recent error: {}'.format(len(self.crashed), self.n, errors))

    def _propagate_obs_info(self, info_n, obs_info_n):
        for obs_info, info in zip(obs_info_n, info_n):
            # obs_info keys take precedence over info keys
            if obs_info is not None:
                info.update(obs_info)

    def _render(self, mode='human', close=False):
        if close:
            # render(close) is not currently supported by the Go VNCSession
            return

        if mode is 'human' and self.vnc_session is not None:
            if self.connection_names[0]:
                self.vnc_session.render(self.connection_names[0])

    def __str__(self):
        return 'VNCEnv<{}>'.format(self.spec.id)

class Mask(object):
    """Blocks the agent from interacting with the environment while the
    environment is resetting.

    On the boundaries:

    - Mask will *drop* actions to environments which have just started
      resetting (i.e. returning done=True in this iteration and have
      env_state=resetting).

    - Mask will *allow* actions to environments which have just
      finished resetting (i.e. their env_state=running).

    - Mask will *allow* rewards from environments which have just
      started resetting (i.e. returning done=True in this iteration
      and have env_state=resetting), but mask observations from them.

    - Mask will *allow* observations from environments which have just
      finished resetting (i.e. their env_state is running)
    """
    def __init__(self, connection_labels, initially_masked=True):
        self.connection_labels = connection_labels
        self.n = len(connection_labels)

        self.episode_ids = [None] * self.n
        if initially_masked:
            self.mask = [None] * self.n
        else:
            self.mask = [True] * self.n

    def close(self, i):
        self.mask[i] = None
        self.episode_ids[i] = None

    def maintain_mask(self, done_n, info_n):
        for i, ok in enumerate(self.mask):
            if info_n[i].get('peek'):
                env_state = info_n[i].get('env_status.peek.env_state', 'resetting')
                episode_id = info_n[i].get('env_status.peek.episode_id')

                if info_n[i].get('env_status.episode_id') != episode_id:
                    completed_episode_id = info_n[i].get('env_status.episode_id')
                else:
                    completed_episode_id = None
            else:
                env_state = info_n[i].get('env_status.env_state', 'resetting')
                episode_id = info_n[i].get('env_status.episode_id')
                completed_episode_id = info_n[i].get('env_status.complete.episode_id')

            # Either:
            # 1. The env is currently masked (not ok)
            # 2. We have an active episode (self.episode_ids[i])
            # 3. We didn't connect the rewarder (done_n[i] is None)
            assert not ok or self.episode_ids[i] is not None or done_n[i] is None, "ok={} episode_id={} i={}".format(ok, episode_id, i)
            if not ok and env_state == 'running':
                extra_logger.info('[%s] Episode began: episode_id=%s env_state=%s', self.connection_labels[i], episode_id, env_state)
                self.mask[i] = True
                # this should change only for initial reset
                self.episode_ids[i] = episode_id
            elif ok and self.episode_ids[i] != episode_id and env_state == 'running':
                extra_logger.info('[%s] Episode ended (and began, so not masking): episode_id=%s->%s env_state=%s', self.connection_labels[i], completed_episode_id, episode_id, env_state)
                self.episode_ids[i] = episode_id
            elif ok and self.episode_ids[i] != episode_id:
                extra_logger.info('[%s] Episode ended: episode_id=%s->%s env_state=%s', self.connection_labels[i], completed_episode_id, episode_id, env_state)
                self.mask[i] = False
                self.episode_ids[i] = episode_id
        return self.mask

    def apply_to_actions(self, action_n, info_n, mask):
        for i, ok in enumerate(mask):
            if ok:
                continue

            action_n[i] = []
            info_n[i]['mask.masked.action'] = True
        return self.mask

    def apply_to_return(self, observation_n, reward_n, done_n, info_n, observation_mask):
        # Next, mask any environments which are resetting. We are
        # guaranteed to get done=True messages prior to getting the
        # v0.env.describe message telling us it's resetting, so the
        # conservative route (block upon done=True, unblock upon
        # v0.env.describe with env_state=running) locks us out of the
        # maximum surface area of environment reset possible.
        for i, ok in enumerate(observation_mask):
            if ok:
                continue

            if len(observation_n[i]['text']) > 0 and ok is False:
                logger.warn('[%s] WARNING: Masking text observation for environment %d: %r. This means we received text data before the environment finished resetting; the text observation has been lost. This is not expected and should be reported.', self.connection_labels[i], i, observation_n[i]['text'])

            observation_n[i] = None
            info_n[i]['mask.masked.observation'] = True

def compile_action(event):
    if isinstance(event, tuple):
        if event[0] == 'KeyEvent':
            name, down = event[1:]
            return spaces.KeyEvent.by_name(name, down=down).compile()
        elif event[0] == 'PointerEvent':
            x, y, buttonmask = event[1:]
            return spaces.PointerEvent(x, y, buttonmask).compile()
    else:
        return event.compile()
