from setuptools import setup, find_packages

setup(name='universe',
      version='0.21.0',
      description="Universe: a software platform for measuring and training an AI's general intelligence across the world's supply of games, websites and other applications.",
      url='https://github.com/openai/universe',
      author='OpenAI',
      author_email='universe@openai.com',
      packages=[package for package in find_packages()
                if package.startswith('universe')],
      install_requires=[
          'autobahn',
          'docker-py==1.10.3',
          'docker-pycreds==0.2.1',
          'fastzbarlight>=0.0.13',
          'go-vncdriver>=0.4.8',
          'gym>=0.6.0',
          'Pillow',
          'PyYAML',
          'six',
          'twisted',
          'ujson',
      ],
      package_data={'universe': ['runtimes.yml']},
      tests_require=['pytest'],
      extras_require={
          'atari': 'gym[atari]',
      }
      )
