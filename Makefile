upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*

test:
	docker run -v /usr/bin/docker:/usr/bin/docker -v /root/.docker:/root/.docker -v /var/run/docker.sock:/var/run/docker.sock --net=host quay.io/openai/universe:test

push:
	docker build -t quay.io/openai/universe .
	docker build -f test.dockerfile -t quay.io/openai/universe:test .

	docker push quay.io/openai/universe
	docker push quay.io/openai/universe:test

test-push:
	docker build -f test.dockerfile -t quay.io/openai/universe:test .
	docker push quay.io/openai/universe:test
