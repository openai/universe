upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*

test:
	find . -name '*.pyc' -delete
	docker build -f test.dockerfile -t quay.io/openai/universe:test .
	docker run -v /usr/bin/docker:/usr/bin/docker -v /root/.docker:/root/.docker -v /var/run/docker.sock:/var/run/docker.sock --net=host quay.io/openai/universe:test

build:
	find . -name '*.pyc' -delete
	docker build -t quay.io/openai/universe .
	docker build -f test.dockerfile -t quay.io/openai/universe:test .

push:
	find . -name '*.pyc' -delete
	docker build -t quay.io/openai/universe .
	docker build -f test.dockerfile -t quay.io/openai/universe:test .

	docker push quay.io/openai/universe
	docker push quay.io/openai/universe:test

test-push:
	docker build -f test.dockerfile -t quay.io/openai/universe:test .
	docker push quay.io/openai/universe:test
