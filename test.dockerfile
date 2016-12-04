FROM quay.io/openai/universe:base

RUN pip install tox

# Run tox. Keep printing so Travis knows we're alive.
CMD ["bash", "-c", "( while true; do echo '.'; sleep 60; done ) & tox"]
