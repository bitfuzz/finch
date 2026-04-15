.PHONY: setup run clean

ifeq ($(OS),Windows_NT)
    SETUP_CMD = cmd.exe /c setup.bat
    RUN_CMD = venv\Scripts\python main.py
    CLEAN_CMD = if exist venv rmdir /s /q venv
else
    SETUP_CMD = bash setup.sh
    RUN_CMD = venv/bin/python main.py
    CLEAN_CMD = rm -rf venv
endif

setup:
	@echo "Setting up..."
	@$(SETUP_CMD)

run:
	@$(RUN_CMD)

clean:
	@$(CLEAN_CMD)
