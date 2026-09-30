.PHONY : run-checks
run-checks :
	isort --check .
	black --check .
	ruff check .

.PHONY : format
format :
	isort .
	black .

.PHONY : test
test :
	python -m pytest hero/tests
