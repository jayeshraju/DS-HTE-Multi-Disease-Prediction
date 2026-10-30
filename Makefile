PYTHON ?= venv/Scripts/python.exe
PYTEST ?= venv/Scripts/pytest.exe
PROFILE ?= paper

.PHONY: setup data audit smoke test run tables figures explain all clean clean-results export-ui ui ui-build ui-test image-bench verify-phase1

verify-phase1:
	$(PYTHON) scripts/verify_phase1.py
setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) -m src.data.download
	$(PYTHON) -m src.data.profile

audit:
	$(PYTHON) -m src.data.profile

smoke:
	$(PYTHON) -m src.run_all --profile smoke

test:
	$(PYTEST) tests -v

run:
	$(PYTHON) -m src.run_all --profile $(PROFILE)

tables:
	$(PYTHON) -c "from src.reporting.tables import generate_all_tables_from_store; generate_all_tables_from_store(); print('Tables generated.')"

figures:
	$(PYTHON) -c "from src.reporting.figures import generate_all_figures; generate_all_figures(); print('Figures generated.')"

explain:
	$(PYTHON) -c "from src.explain.shap_explain import compute_shap_explanations; print('Explainability verified.')"

clean-results:
	$(PYTHON) -c "import shutil, os, glob; [shutil.rmtree(d, ignore_errors=True) for d in ['results/tables', 'results/figures', 'results/models', 'results/audit', 'results/predictions', 'results/fold_metrics', 'ui_data']]; [os.remove(f) for f in glob.glob('results/*.json') + glob.glob('results/*.md') if os.path.exists(f)]; print('Results cleaned successfully.')"

export-ui:
	$(PYTHON) -m src.reporting.export_ui

ui:
	$(PYTHON) -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000 & cd ui && npm run dev

ui-build:
	cd ui && npm run build

ui-test:
	$(PYTEST) tests/test_fastapi_endpoints.py tests/test_imaging_api.py tests/test_imaging_agent.py tests/test_cross_table_invariants.py -v

image-bench:
	$(PYTHON) -m src.imaging.benchmarks

all: setup data audit test run tables figures export-ui image-bench
