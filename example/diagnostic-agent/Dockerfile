FROM docker.openai.com/universe:base

RUN apt-get update && apt-get install -y netcat

ADD . /

ENTRYPOINT ["/diagnostic-agent.py"]
CMD ["-A", "forward", "-r", "vnc://localhost:5900+15900"]
