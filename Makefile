.PHONY: all build ui py test install dev clean

BIN := bin/herd

all: py ui build

py:			# install the Python backend into .venv
	uv venv
	uv pip install -e .

ui:			# build the web UI into ui/dist
	cd ui && npm install && npm run build

build: $(BIN)		# build the Go CLI
$(BIN): cli/*.go
	cd cli && go build -o ../$(BIN) .

test: build		# math self-check + CLI smoke test
	.venv/bin/python tests/test_vram.py
	.venv/bin/python tests/test_scheduler.py
	.venv/bin/python tests/test_backends.py
	@./$(BIN) detect >/dev/null && ./$(BIN) recommend --json >/dev/null && echo "  ok  cli smoke"

install: build		# put herd on your PATH
	install -Dm755 $(BIN) $(HOME)/.local/bin/herd
	@mkdir -p $(HOME)/.config/herd
	@echo "$(PWD)/.venv/bin/python" > $(HOME)/.config/herd/python
	@echo "installed to ~/.local/bin/herd"
	@echo "backend recorded: $(PWD)/.venv/bin/python"

dev:			# API on :8787 plus the Vite dev server on :5173
	./$(BIN) serve & cd ui && npm run dev

clean:
	rm -rf bin ui/dist .venv
