.PHONY: setup run clean

setup:
	@echo "Checking OS and setting up..."
	@if [ "$(OS)" = "Windows_NT" ]; then \
		cmd.exe /c setup.bat; \
	else \
		bash setup.sh; \
	fi

run:
	@if [ "$(OS)" = "Windows_NT" ]; then \
		venv\Scripts\python main.py; \
	else \
		venv/bin/python main.py; \
	fi

clean:
	@if [ "$(OS)" = "Windows_NT" ]; then \
		if exist venv rmdir /s /q venv; \
	else \
		rm -rf venv; \
	fi
