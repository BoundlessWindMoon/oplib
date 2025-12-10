NCU := $(shell which ncu)
NCU_UI := $(shell which ncu-ui)
PYTHON := $(shell which python)

ifeq ($(NCU),)
$(error "ncu not found, please install Nsight Compute")
endif
ifeq ($(PYTHON),)
$(error "python not found, please install python")
endif

TARGET_SCRIPT := ./run.py
REPORT_DIR := ./build/ncu

.PHONY: profile-full ui
profile-full:
	@echo "Profiling with full option..."
	@mkdir -p $(REPORT_DIR)
	TIMESTAMP=$$(date +'%Y%m%d_%H%M%S'); \
	sudo $(NCU) --set full -o $(REPORT_DIR)/$${TIMESTAMP}_full $(PYTHON) $(TARGET_SCRIPT)
	@echo "Report generated at: $(REPORT_DIR)/$${TIMESTAMP}_full.ncu-rep"

ui:
	@echo "Opening .ncu-rep file..."
	@if [ -n "$(FILE)" ]; then \
		if [ -f "$(FILE)" ]; then \
			echo "Using user-specified file: $(FILE)"; \
			$(NCU_UI) "$(FILE)" & \
		else \
			echo "Error: File '$(FILE)' not found"; \
			exit 1; \
		fi; \
	else \
		echo "No file specified, searching for the latest report in $(REPORT_DIR)"; \
		if [ -d "$(REPORT_DIR)" ]; then \
			LATEST_REPORT=$$(ls -t $(REPORT_DIR)/*_full.ncu-rep 2>/dev/null | head -n 1); \
			if [ -n "$$LATEST_REPORT" ]; then \
				if [ -f "$(NCU_UI)" ]; then \
					$(NCU_UI) "$$LATEST_REPORT" & \
				else \
					echo "Error: ncu-ui not found at $(NCU_UI)"; \
					exit 1; \
				fi; \
			else \
				echo "Error: No .ncu-rep files found in $(REPORT_DIR)"; \
				exit 1; \
			fi; \
		else \
			echo "Error: Directory $(REPORT_DIR) does not exist"; \
			exit 1; \
		fi; \
	fi
