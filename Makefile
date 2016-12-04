upload:
	rm -rf dist
	python setup.py sdist
	twine upload dist/*

test:
	docker run -v $(pwd):/usr/local/universe -v /usr/bin/docker:/usr/bin/docker -v ~/.docker:/root/.docker -v /var/run/docker.sock:/var/run/docker.sock --net=host quay.io/openai/universe:test

base:
	docker build -t quay.io/openai/universe:base .
	docker build -f test.dockerfile -t quay.io/openai/universe:test .

	docker push quay.io/openai/universe:base
	docker push quay.io/openai/universe:test

test-push:
	docker push quay.io/openai/universe:test
