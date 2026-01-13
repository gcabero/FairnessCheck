# Mock Classifier Service

A simple Fast api REST API for testing the fairness-check CLI tool locally.

## Features

- **`/classify`** - Main endpoint (deterministic based on features)
- **`/classify/random`** - Random predictions (for testing)
- **`/classify/biased`** - Intentionally biased (always returns positive class)

## Installation

```bash
cd mock_server
uv pip install -r requirements.txt
```

## Running the Server

```bash
cd mock_server
uv run  app.py
```

The server will start on `http://localhost:8000`

## API Endpoints

### GET / - Service Info
```bash
curl http://localhost:8000/
```

### GET /health - Health Check (good for K8s probes!)
```bash
curl http://localhost:8000/health
```

### POST /classify - Main Classifier
Deterministic predictions based on features hash.

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"features": "test_input_1"}'
```

Response:
```json
{
  "prediction": 1,
  "features": "test_input_1",
   "note": null
}
```

### POST /classify/random - Random Classifier
Random predictions for testing.

```bash
curl -X POST http://localhost:8000/classify/random \
  -H "Content-Type: application/json" \
  -d '{"features": "any_input"}'
```

### POST /classify/biased - Biased Classifier
Always returns positive class (for testing fairness issues).

```bash
curl -X POST http://localhost:8000/classify/biased \
  -H "Content-Type: application/json" \
  -d '{"features": "any_input"}'
```


## Test Fairness with Mock Server

1. **Start server:**
   ```bash
   python mock_server/app.py
   ```
2. Make sure there is a compliant dataset

3**Run fairness check:**
   ```bash
   fairness-check report config.yaml --verbose
   ```

3. **Check results:**
   The CLI will show fairness metrics for the mock classifier.

To demonstrate fairness issues you may want to use the biased endpoint:

 * Update `config.test.yaml` to use the biased endpoint:
   ```yaml
   endpoint:
     url: "http://localhost:8000/classify/biased"
   ```

 * Run the fairness check from the project root:

   ```bash
   # Test with the local mock server
   fairness-check report config.test.yaml --verbose
   ```


