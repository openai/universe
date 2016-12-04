from setuptools import setup, find_packages

setup(name='universe',
      version='0.20.0',
      packages=[package for package in find_packages()
                if package.startswith('universe')],
      install_requires=[
          'autobahn',
          'docker-py==1.10.3',
          'fastzbarlight>=0.0.13',
          'go-vncdriver>=0.4.8',
          'gym>=0.5.2',
          'Pillow',
          'PyYAML',
          'six',
          'twisted',
          'ujson',
      ],
      tests_require=['pytest'],
      extras_require={
          'atari': 'gym[atari]',
      }
)
