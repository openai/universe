import logging
import os

from twisted.internet import defer, endpoints

from universe import error, utils
from universe.twisty import reactor
from universe.vncdriver import screen, vnc_client

PYGAME_INSTALLED = None
def load_pygame():
    global PYGAME_INSTALLED, pygame
    if PYGAME_INSTALLED is not None:
        return

    try:
        import pygame
        PYGAME_INSTALLED = True
    except ImportError:
        PYGAME_INSTALLED = False

logger = logging.getLogger(__name__)


class LibVNCSession(object):
    def __init__(self, remotes, error_buffer, encoding=None, compress_level=None, fine_quality_level=None, subsample_level=None):
        """compress_level: 0-9 [9 is highest compression]
        fine_quality_level: 0-100 [100 is best quality]
        subsample_level: 0-3 [0 is best quality]

        Lots of references for this, but
        https://github.com/TurboVNC/turbovnc/blob/master/doc/performance.txt
        is decent.
        """

        load_pygame()
        import libvncdriver

        if encoding is None:
            encoding = os.environ.get('LIBVNC_ENCODING', 'tight')
        if compress_level is None:
            compress_level = int(os.environ.get('LIBVNC_COMPRESS_LEVEL', '0'))
        if fine_quality_level is None:
            fine_quality_level = int(os.environ.get('LIBVNC_FINE_QUALITY_LEVEL', '100'))
        if subsample_level is None:
            subsample_level = int(os.environ.get('LIBVNC_SUBSAMPLE_LEVEL', '0'))

        if not hasattr(libvncdriver, 'VNCSession'):
            raise error.Error('''
 *=================================================*
|| libvncdriver is not installed                   ||
|| Try installing with "pip install libvncdriver"  ||
|| or use the go or python driver by setting       ||
|| UNIVERSE_VNCDRIVER=go                                ||
|| UNIVERSE_VNCDRIVER=py                                ||
 *=================================================*''')
        logger.info("Using libvncdriver's %s encoding" % encoding)
        self.driver = libvncdriver.VNCSession(
            remotes=remotes,
            error_buffer=error_buffer,
            encoding=encoding,
            compress_level=compress_level,
            fine_quality_level=fine_quality_level,
            subsample_level=subsample_level,
        )
        self.screen = None
        self.render_called_once = False
        if PYGAME_INSTALLED:
            pygame.init()

    def flip(self):
        return self._guard(self.driver.flip)

    def step(self, action):
        return self.driver.step(action)

    def render(self):
        self._guard(self._render)

    def _guard(self, fn):
        try:
            return fn()
        except (KeyboardInterrupt, SystemExit):
            self.close()

    def _render(self):
        self.before_render()
        if not PYGAME_INSTALLED:
            return
        # For some reason pygame wants X and Y swapped
        aray, n = self.driver.flip()
        if self.screen is None:
            self.screen = pygame.display.set_mode(aray[0].shape[:2][::-1])
        surf = pygame.surfarray.make_surface(aray[0].swapaxes(0, 1))
        rect = surf.get_rect()
        self.screen.blit(surf, rect)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()

    def before_render(self):
        if not self.render_called_once:
            self.render_called_once = True
            if not PYGAME_INSTALLED:
                logger.warn('''
 *================================================================*
||                                                                ||
|| Rendering disabled when using libvnc without pygame installed. ||
|| Consider viewing over VNC or running "pip install pygame".     ||
||                                                                ||
 *================================================================*''')


    def close(self):
        if PYGAME_INSTALLED:
            pygame.quit()
        self.driver.close()
