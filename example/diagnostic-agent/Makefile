URL ?= docker.openai.com/diagnostic-agent

build:
	docker build -t $(URL) .

push: build
	docker push $(URL)

pod.yaml: pod.yaml.template make-podspec.py
	./make-podspec.py flashgames.NeonRace-v0 1 /tmp/demo > pod.yaml

