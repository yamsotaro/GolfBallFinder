.PHONY: help bootstrap project python-check python-test shell-check seed-model export-seed

help:
	@printf '%s\n' \
	  'make bootstrap     # macOS one-stop setup' \
	  'make project       # generate Xcode project' \
	  'make python-check  # syntax-check Python utilities' \
	  'make python-test   # run Python utility/config tests' \
	  'make shell-check   # syntax-check shell scripts' \
	  'make seed-model    # fetch pinned third-party golf-ball seed checkpoint' \
	  'make export-seed   # export fetched seed checkpoint to Core ML'

bootstrap:
	./scripts/bootstrap_mac.sh

project:
	xcodegen generate

python-check:
	python3 -m py_compile training/*.py scripts/*.py Tests/Python/*.py

python-test:
	python3 -m unittest discover -s Tests/Python -v

shell-check:
	bash -n scripts/*.sh

seed-model:
	python3 scripts/fetch_seed_model.py

export-seed:
	python3 training/export_coreml.py --weights training/models/seed_golf_ball_yolov8n.pt --output GolfBallFinder/Resources/GolfBall.mlpackage
