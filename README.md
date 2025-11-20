# 🧠 Hybrid GNN-LLM Recommendation System

A **production-grade recommendation engine** combining Graph Neural Networks, Large Language Models, and Computer Vision into a unified multimodal system. Built with real-time serving, preference alignment (DPO), and full MLOps automation.

---

## 📌 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        HYBRID RECOMMENDATION ENGINE                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐               │
│   │  User-Item    │   │  Product     │   │  Product Image   │               │
│   │  Interactions │   │  Descriptions│   │  Catalog         │               │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────────┘               │
│          │                  │                   │                            │
│          ▼                  ▼                   ▼                            │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────────┐               │
│   │  GraphSAGE   │   │  LLM Encoder │   │  CLIP ViT        │               │
│   │  (GNN)       │   │  (TinyLlama/ │   │  (Vision)        │               │
│   │              │   │   Mistral)   │   │                  │               │
│   └──────┬───────┘   └──────┬───────┘   └──────┬───────────┘               │
│          │                  │                   │                            │
│          ▼                  ▼                   ▼                            │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │           Multimodal Attention Fusion Layer              │               │
│   │                                                         │               │
│   │   α_gnn · E_gnn + α_llm · E_llm + α_vis · E_vision    │               │
│   │                                                         │               │
│   │   (Learned attention weights, softmax-normalized)       │               │
│   └─────────────────────┬───────────────────────────────────┘               │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │              DPO Preference Alignment                    │               │
│   │                                                         │               │
│   │   Fine-tune with (prompt, chosen, rejected) pairs       │               │
│   │   LoRA adapters (rank=16, α=32) for efficiency          │               │
│   └─────────────────────┬───────────────────────────────────┘               │
│                         │                                                   │
│                         ▼                                                   │
│   ┌─────────────────────────────────────────────────────────┐               │
│   │              FastAPI Serving Layer                        │               │
│   │                                                         │               │
│   │   POST /recommend      → Single-user recommendations   │               │
│   │   POST /recommend/batch → Batch recommendations         │               │
│   │   POST /feedback        → User feedback collection      │               │
│   │   GET  /health          → Health probe                  │               │
│   │   GET  /metrics         → Prometheus metrics            │               │
│   └─────────────────────────────────────────────────────────┘               │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│  STREAMING PIPELINE          │  MLOPS AUTOMATION                            │
│                              │                                              │
│  Kafka Consumer              │  MLflow Experiment Tracking                  │
│    → Event Processor         │  Airflow DAG Orchestration                   │
│    → User Buffer (per-user)  │  Prometheus Monitoring (p50/p95/p99)         │
│    → Model Update Trigger    │  Alert Manager (threshold + cooldown)        │
│    → Kafka Producer          │  Model Registry (versioning + A/B)           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 How It Works

### 1. Graph-Based Collaborative Filtering (GNN)
The system constructs a **bipartite user-item interaction graph** from purchase history. GraphSAGE learns node embeddings by aggregating neighborhood information:

- **User nodes** capture purchasing patterns via multi-hop neighbor aggregation
- **Item nodes** encode co-purchase relationships and popularity signals
- **Link prediction** with BPR (Bayesian Personalized Ranking) loss trains the model to rank positive interactions above negative samples
- Architecture: `[SAGEConv → BatchNorm → ReLU → Dropout] × L layers`, L2-normalized embeddings

### 2. LLM Semantic Understanding
Natural language product descriptions are encoded via pre-trained language models (TinyLlama / Mistral-7B):

- Mean-pooled hidden states create dense semantic representations
- Lazy loading and 8-bit quantization keep memory footprint manageable
- Text embeddings capture product attributes, categories, and descriptions that graphs miss

### 3. CLIP Vision Features
Product images are processed through **OpenAI CLIP ViT** to extract visual embeddings:

- Captures visual product attributes (color, shape, packaging)
- Text-to-image CLIP scores enable zero-shot visual search
- Particularly useful for fashion, furniture, and food categories

### 4. Multimodal Fusion
A **learned attention mechanism** dynamically weights the three modalities:

```
output = α_gnn × proj(GNN_emb) + α_llm × proj(LLM_emb) + α_vis × proj(Vision_emb)
```

- Attention weights are data-driven — the model learns which modality matters per prediction
- Active users get more graph signal; cold-start users rely on text/vision
- Full interpretability: you can inspect which modality drove each recommendation

### 5. DPO Preference Alignment
Instead of just supervised fine-tuning (SFT), the model is aligned to user preferences via **Direct Preference Optimization**:

- Generates `(prompt, chosen, rejected)` triplets from ground truth vs. model errors
- Three negative sampling strategies: model-based, random, and hard negatives
- LoRA adapters (rank=16, α=32) enable efficient fine-tuning of large LLMs
- Reward accuracy and margin convergence tracked throughout training

### 6. Real-Time Serving
A **FastAPI** application serves recommendations with sub-100ms latency:

- Health probes (`/health`, `/ready`) for Kubernetes deployments
- Batch endpoint for bulk recommendation generation
- Feedback loop for online learning signals
- Request tracking middleware (latency, request ID)

### 7. Streaming Pipeline
**Kafka-based** event processing enables real-time model updates:

- User interaction events consumed from Kafka topics
- Per-user event buffers with configurable flush thresholds
- Event handlers trigger embedding re-computation on significant events (purchases > clicks)
- Recommendation updates published to downstream topics

### 8. MLOps Automation
Full experiment lifecycle management:

- **Experiment Tracking**: Parameters, metrics (with history), artifacts, best-run selection
- **Pipeline DAGs**: Airflow-compatible DAGs with topological sort, cycle detection, and auto-export
- **Monitoring**: Latency percentiles (p50/p95/p99), QPS, Prometheus exposition format
- **Alerting**: Threshold-based rules with cooldown to prevent alert fatigue

---

## 📁 Project Structure

```
├── src/
│   ├── data/
│   │   ├── dataset_loader.py            # Multi-format data loading, ID mapping, temporal split
│   │   └── preference_data_generator.py # DPO preference pair generation (3 strategies)
│   ├── graph/
│   │   ├── graph_builder.py             # Bipartite graph construction with PyTorch Geometric
│   │   └── gnn_model.py                # Multi-layer GraphSAGE with link prediction
│   ├── models/
│   │   ├── hybrid_model.py             # Gated GNN-LLM fusion (α-weighted)
│   │   ├── llm_wrapper.py              # LLM embedding extractor (lazy load, quantization)
│   │   └── multimodal.py               # CLIP vision encoder + trimodal attention fusion
│   ├── train/
│   │   ├── gnn_trainer.py              # BPR loss, AUC-ROC, Hit Rate, MRR, early stopping
│   │   ├── dpo_config.py               # LoRA + quantization + DPO hyperparameters
│   │   └── dpo_trainer.py              # TRL DPOTrainer wrapper with reward computation
│   ├── evaluation/
│   │   ├── metrics.py                  # NDCG@K, Precision@K, Recall@K, MAP, MRR
│   │   └── reward_analyzer.py          # Reward convergence, win rate, margin tracking
│   ├── serving/
│   │   ├── app.py                      # FastAPI server (6 endpoints + middleware)
│   │   ├── schemas.py                  # Pydantic request/response validation
│   │   ├── stream_processor.py         # Kafka event consumer + processor
│   │   └── model_registry.py           # Versioned model management + A/B testing
│   └── mlops/
│       ├── experiment_tracker.py        # MLflow-compatible experiment tracking
│       ├── monitoring.py               # MetricsCollector + AlertManager
│       └── pipeline_dag.py             # Airflow DAG builder with cycle detection
├── tests/                              # 195 unit tests
│   ├── test_graph_builder.py
│   ├── test_gnn_model.py
│   ├── test_hybrid_model.py
│   ├── test_preference_data.py
│   ├── test_dpo_trainer.py
│   ├── test_metrics.py
│   ├── test_api.py
│   ├── test_streaming.py
│   ├── test_multimodal.py
│   └── test_mlops.py
├── scripts/
│   ├── train_gnn.py                    # GNN training CLI
│   ├── train_dpo.py                    # DPO training CLI (with dry-run mode)
│   ├── run_evaluation.py               # Dataset & model evaluation
│   └── compare_sft_dpo.py             # SFT vs DPO benchmark comparison
├── requirements.txt
└── .gitignore
```

---

## 🚀 Local Setup

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/<your-username>/hybrid-gnn-llm-recsys.git
cd hybrid-gnn-llm-recsys

# Create virtual environment
python -m venv .venv

# Activate
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run Tests

```bash
python -m pytest tests/ -v --tb=short
# Expected: 195 passed
```

---

## 💻 Usage

### Train the GNN

```bash
python scripts/train_gnn.py \
    --data-dir ./ \
    --epochs 100 \
    --lr 0.001 \
    --embedding-dim 128
```

### Train with DPO Alignment

```bash
# Dry run (generate data + config only, no GPU needed)
python scripts/train_dpo.py \
    --data-dir ./ \
    --beta 0.1 \
    --strategy mixed \
    --dry-run

# Full training (requires GPU)
python scripts/train_dpo.py \
    --data-dir ./ \
    --beta 0.1 \
    --epochs 3 \
    --lora-rank 16
```

### Compare SFT vs DPO

```bash
python scripts/compare_sft_dpo.py --data-dir ./
```

### Start the API Server

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Then open `http://localhost:8000/docs` for the interactive Swagger UI.

### API Examples

```bash
# Health check
curl http://localhost:8000/health

# Get recommendations
curl -X POST http://localhost:8000/recommend \
    -H "Content-Type: application/json" \
    -d '{"user_id": "user_123", "num_items": 5}'

# Batch recommendations
curl -X POST http://localhost:8000/recommend/batch \
    -H "Content-Type: application/json" \
    -d '{"user_ids": ["u1", "u2", "u3"], "num_items": 5}'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
    -H "Content-Type: application/json" \
    -d '{"user_id": "u1", "item_id": "i1", "rating": 4.5}'
```

---

## 🧪 Testing

| Module | Tests | Coverage |
|--------|-------|----------|
| Graph Builder | 18 | Graph construction, bidirectional edges, splitting |
| GNN Model | 16 | Forward pass, gradients, link prediction, top-K |
| Hybrid Fusion | 18 | Gate values, gradient flow, determinism |
| Preference Data | 18 | Pair generation, strategies, splitting, accuracy |
| DPO Config | 12 | Validation, serialization, LoRA config |
| Metrics | 28 | NDCG, Precision, Recall, MRR, MAP, rewards |
| API | 20 | All endpoints, schemas, middleware headers |
| Streaming | 18 | Event processing, buffering, Kafka config, registry |
| Multimodal | 11 | CLIP encoder, fusion layer, attention weights |
| MLOps | 36 | Tracker lifecycle, monitoring, DAG validation |
| **Total** | **195** | **All passing** |

---

## ⚠️ Limitations

1. **No pre-trained model weights included** — Training requires GPU (at least 8GB VRAM for LoRA fine-tuning with TinyLlama). DPO dry-run mode works on CPU.

2. **Single dataset** — Currently built around the Amazon Prime Pantry dataset (~12K users, ~8K items). Adapting to other domains requires updating the data loader.

3. **Kafka/Airflow not bundled** — The streaming and pipeline DAG modules define the integration interfaces. Actual Kafka brokers and Airflow schedulers must be deployed separately for production use.

4. **CLIP model not pre-loaded** — Vision features require downloading the CLIP model (~600MB). The encoder uses lazy loading so memory is only consumed when vision features are explicitly requested.

5. **Cold-start limitation** — New users with zero interactions get fallback popularity-based recommendations until sufficient data accumulates for graph-based signals.

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **ML Models** | PyTorch, PyTorch Geometric, Transformers, PEFT, TRL |
| **GNN** | GraphSAGE (SAGEConv), BPR Loss, Negative Sampling |
| **LLM** | TinyLlama / Mistral-7B, LoRA (rank=16, α=32) |
| **Vision** | CLIP ViT-B/32, OpenAI |
| **Alignment** | Direct Preference Optimization (DPO) |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Streaming** | Kafka (kafka-python) |
| **Monitoring** | Prometheus exposition, Custom AlertManager |
| **Orchestration** | Airflow DAG generation |
| **Tracking** | MLflow-compatible ExperimentTracker |
| **Testing** | pytest (195 tests) |

---

## 📄 License

MIT
