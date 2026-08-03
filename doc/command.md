# Run all tests with coverage
pytest tests/ -v --cov=src --cov=neurofusion --cov-report=html

# Run unit tests only
pytest tests/unit/ -v

# Run integration tests only
pytest tests/integration/ -v -m integration

# Run specific test file
pytest tests/unit/test_models/test_neurofusion.py -v

# Run with verbose output and debug
pytest tests/ -v --tb=long --maxfail=1

# Run tests with GPU
pytest tests/ --device=cuda

# Run tests with specific marker
pytest tests/ -m "not slow"