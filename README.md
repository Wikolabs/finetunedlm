# finetunedlm — Domain-Specific LLM Fine-Tuning as a Service

> Wikolabs AI Platform · GCP Cloud Run GPU · Next.js 14 + FastAPI · LoRA / QLoRA · Mistral-7B / Llama-3.1-8B

---

## Table of Contents

1. [Vision produit](#1-vision-produit)
2. [User Stories](#2-user-stories)
3. [Business Rules](#3-business-rules)
4. [Architecture Overview](#4-architecture-overview)
5. [Mono-repo Structure](#5-mono-repo-structure)
6. [Fine-Tuning Pipeline Specification](#6-fine-tuning-pipeline-specification)
7. [UML Diagrams](#7-uml-diagrams)
8. [API Reference](#8-api-reference)
9. [UI Simulation Guide](#9-ui-simulation-guide)
10. [Database Schema](#10-database-schema)
11. [Infrastructure & Deployment](#11-infrastructure--deployment)
12. [CI/CD Pipeline](#12-cicd-pipeline)
13. [Kaggle Dataset](#13-kaggle-dataset)
14. [Local Development](#14-local-development)

---

## 1. Vision produit

### Problème métier

Les modèles de langage généraux (GPT-4, Mistral, Llama) offrent d'excellentes performances générales mais échouent sur les tâches très spécialisées :

- **Support client** : terminologie propriétaire, procédures internes, tonalité de marque
- **Juridique** : jurisprudence spécifique, clauses contractuelles d'un secteur
- **Médical** : protocoles cliniques, nomenclature ICD-10, pharmacopée spécialisée
- **Finance** : réglementations prudentielles, reporting IFRS, scoring de crédit interne

Fine-tuner un LLM de 7-8 milliards de paramètres sur un dataset métier permet d'atteindre des performances de pointe sur ces tâches à une fraction du coût d'un modèle propriétaire.

### Solution

**finetunedlm** est une plateforme end-to-end de fine-tuning de LLM. Upload d'un dataset JSONL, configuration LoRA ou QLoRA, lancement du job sur GPU Cloud Run, suivi des métriques, comparaison avant/après, et déploiement en endpoint OpenAI-compatible.

### Proposition de valeur

| Approche actuelle | finetunedlm |
|---|---|
| Prompt engineering long et fragile | Modèle entraîné sur la terminologie métier |
| API propriétaire coûteuse à l'usage | Modèle auto-hébergé, coût fixe |
| Latence réseau vers API externe | Inférence locale, sous 200ms |
| Données confidentielles envoyées à des tiers | 100% données sur infrastructure propre |
| Pas de versioning des comportements | Artefacts versionnés dans GCS |

### Cas d'usage cibles

- Éditeurs de logiciels métier (chatbot intégré dans leur SaaS)
- Cabinets de conseil (assistant documentaire interne)
- Assureurs / banques (scoring et analyse de contrats)
- Hôpitaux et cliniques (aide à la décision clinique)
- E-commerce B2B (générateur de fiches produits sectorielles)

---

## 2. User Stories

### Rôles

- **ML Engineer** : configure et lance les jobs de fine-tuning
- **Product Manager** : valide la qualité avant/après fine-tuning
- **Data Engineer** : prépare et valide les datasets
- **CTO / DSI** : supervise les coûts GPU et la gouvernance des modèles

### Epic 1 — Gestion des datasets

| ID | En tant que | Je veux | Afin de |
|---|---|---|---|
| US-001 | Data Engineer | uploader un fichier JSONL instruction/input/output | préparer le jeu d'entraînement |
| US-002 | Data Engineer | voir une validation automatique du format et du nombre de samples | détecter les erreurs avant lancement |
| US-003 | ML Engineer | prévisualiser 10 exemples aléatoires du dataset | vérifier la qualité des données |
| US-004 | Data Engineer | marquer les colonnes PII pour les exclure du fine-tuning | respecter les obligations RGPD |

### Epic 2 — Configuration du fine-tuning

| ID | En tant que | Je veux | Afin de |
|---|---|---|---|
| US-005 | ML Engineer | choisir le modèle de base Mistral-7B ou Llama-3.1-8B | adapter le modèle au cas d'usage |
| US-006 | ML Engineer | configurer les paramètres LoRA r, alpha, target_modules | contrôler le trade-off qualité/coût |
| US-007 | ML Engineer | voir l'estimation du coût GPU avant de lancer | valider le budget de formation |
| US-008 | CTO | voir la liste des jobs par statut et leur coût réel | piloter les dépenses de compute |

### Epic 3 — Entraînement & évaluation

| ID | En tant que | Je veux | Afin de |
|---|---|---|---|
| US-009 | ML Engineer | suivre les courbes de loss en temps réel via W&B | détecter l'overfitting ou underfitting |
| US-010 | ML Engineer | comparer les métriques ROUGE-L base vs fine-tuned | quantifier l'amélioration |
| US-011 | Product Manager | voir une comparaison avant/après sur mes propres prompts | valider la pertinence métier |
| US-012 | ML Engineer | activer l'early stopping automatique | éviter le gaspillage GPU |

### Epic 4 — Déploiement & serving

| ID | En tant que | Je veux | Afin de |
|---|---|---|---|
| US-013 | ML Engineer | merger les adaptateurs LoRA dans le modèle de base | créer un modèle production autonome |
| US-014 | CTO | avoir un endpoint OpenAI-compatible /v1/chat/completions | intégrer sans modifier le code client |
| US-015 | ML Engineer | quantifier le modèle final en GGUF 4-bit | réduire les besoins mémoire en inférence |

---

## 3. Business Rules

### BR-001 — Validation du dataset

```
Format requis : JSONL (une ligne = un exemple)
Chaque ligne : {"instruction": "...", "input": "...", "output": "..."}
Minimum : 500 samples pour lancer le fine-tuning
Maximum : 100 000 samples par job (chunking automatique au-dela)
Split auto : 90% train / 10% eval
```

### BR-002 — Configuration LoRA

```python
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
```

### BR-003 — Configuration QLoRA économique

```python
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)
```

### BR-004 — Configuration SFTTrainer

```python
training_args = TrainingArguments(
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    max_seq_length=2048,
    fp16=True,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)
```

### BR-005 — Early Stopping

Si eval_loss ne s'améliore pas pendant 3 époques consécutives, arrêt du job.
Metric monitored : eval/loss. Min delta : 0.001.

### BR-006 — Métriques d'évaluation

```
ROUGE-L recall  : sur le jeu eval 10%
Perplexity      : exp(eval_loss) — attendu < 15 pour un bon fine-tuning
Hallucination   : proportion de réponses non factuelles via LLM-as-judge
Domain accuracy : score sur benchmark de questions métier
```

### BR-007 — Versioning des artefacts GCS

```
gcs://finetunedlm-artifacts/{job_id}/
  checkpoints/epoch-1/
  checkpoints/epoch-2/
  final/merged_model/
  final/model.gguf
  eval_results.json
  training_config.json
```

### BR-008 — Estimation du coût GPU

```python
def estimate_cost(dataset_size: int, epochs: int) -> float:
    samples_per_hour = 1200  # batch_size=4, seq_len=2048, NVIDIA L4
    hours = (dataset_size * epochs) / samples_per_hour
    gpu_cost_per_hour = 0.54  # NVIDIA L4 Cloud Run
    return hours * gpu_cost_per_hour
```

Confirmation requise si coût estimé > 50 USD.

### BR-009 — Merge LoRA

```python
model = model.merge_and_unload()
model.save_pretrained("final/merged_model")
```

### BR-010 — Serving OpenAI-compatible

POST /v1/chat/completions format identique à l'API OpenAI.
Rate limiting : 60 requêtes/minute free tier, illimité paid.

### BR-011 — Quantification GGUF

Conversion 4-bit pour déploiement léger via llama.cpp.
Méthode : q4_k_m (meilleur ratio qualité/taille).

### BR-012 — Catégories benchmark Dolly-15k

brainstorming, classification, closed_qa, generation,
information_extraction, open_qa, summarization, creative_writing.

---

## 4. Architecture Overview

```
Browser / Client
Next.js 14 · TypeScript · Tailwind CSS
        |
        | HTTPS / WebSocket
        v
FastAPI Backend (Python 3.11)
/api/datasets  /api/jobs  /api/eval  /api/inference
transformers · peft (LoRA) · trl (SFTTrainer)
bitsandbytes (QLoRA) · datasets · accelerate
        |                       |
        v                       v
PostgreSQL               GCP Cloud Run GPU
Jobs, Datasets           Fine-Tuning Jobs (A100/L4)
Model Versions                |
        |                     v
        v                     GCS
Weights & Biases         Model Checkpoints
(wandb tracking)         Merged Models, GGUF
```

---

## 5. Mono-repo Structure

```
finetunedlm/
├── frontend/
│   ├── src/app/
│   │   ├── page.tsx                    # Dashboard jobs
│   │   ├── datasets/page.tsx           # Upload et validation
│   │   ├── jobs/[id]/page.tsx          # Suivi training live
│   │   ├── compare/page.tsx            # Before/After comparison
│   │   └── models/page.tsx             # Model registry
│   └── src/components/
│       ├── DatasetUploader.tsx         # Drag and drop JSONL
│       ├── LoraConfigurator.tsx        # Form config LoRA
│       ├── LossChart.tsx               # Recharts live loss
│       ├── RougeComparison.tsx         # Métriques ROUGE
│       ├── PromptComparator.tsx        # Base vs Fine-tuned
│       └── CostEstimator.tsx           # Estimation GPU cost
│
├── backend/
│   └── app/
│       ├── main.py
│       ├── api/
│       │   ├── datasets.py             # Upload, validation
│       │   ├── jobs.py                 # Training job management
│       │   ├── evaluation.py           # ROUGE, perplexity
│       │   ├── inference.py            # Model serving
│       │   └── models.py              # Model registry
│       └── ml/
│           ├── trainer.py              # SFTTrainer wrapper
│           ├── lora_config.py          # LoRA/QLoRA configs
│           ├── evaluator.py            # ROUGE, benchmarks
│           ├── merger.py               # LoRA merge + export
│           └── quantizer.py           # GGUF conversion
│
├── .github/workflows/
│   ├── ci.yml
│   └── deploy-cloud-run.yml
├── cloudbuild.yaml
├── docker-compose.yml
└── README.md
```

---

## 6. Fine-Tuning Pipeline Specification

```
1. Dataset Upload
   Validation format JSONL, check 500+ samples, train/eval split 90/10

2. Configuration
   Choix modèle base, méthode LoRA ou QLoRA, hyperparamètres, estimation coût

3. Entraînement Cloud Run GPU
   Chargement modèle HuggingFace, SFTTrainer cosine LR, logging wandb,
   checkpoints GCS, early stopping

4. Évaluation
   ROUGE-L eval set, perplexity base vs fine-tuned, rapport JSON GCS

5. Export & Déploiement
   Merge LoRA, quantification GGUF 4-bit, endpoint vLLM OpenAI-compatible
```

---

## 7. UML Diagrams

### 7.1 LoRA Training Pipeline

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#7c3aed', 'tertiaryColor': '#0f172a'}}}%%
flowchart TD
    classDef input fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef process fill:#14532d,stroke:#22c55e,color:#fff
    classDef ml fill:#4a1d96,stroke:#a78bfa,color:#fff
    classDef storage fill:#7c2d12,stroke:#f97316,color:#fff

    A[Dataset JSONL Upload]:::input --> B[Validation\n500+ samples\nformat check]:::process
    B --> C[Train 90% / Eval 10%]:::process
    C --> D[Load Base Model\nMistral-7B ou Llama-3.1-8B]:::ml
    D --> E{Méthode}
    E -->|LoRA| F[LoraConfig\nr=16 alpha=32]:::ml
    E -->|QLoRA| G[BitsAndBytes 4-bit NF4\n+ LoraConfig]:::ml
    F --> H[SFTTrainer\ncosine LR warmup 3%]:::ml
    G --> H
    H --> I[Log W&B\ntrain_loss eval_loss]:::storage
    H --> J{Early Stop?\n3 epochs no improve}
    J -->|No| H
    J -->|Yes| K[Save Best Checkpoint GCS]:::storage
    K --> L[Evaluation ROUGE-L\nPerplexity]:::process
    L --> M[Merge LoRA\nmerge_and_unload]:::ml
    M --> N[Quantify GGUF 4-bit]:::ml
    N --> O[Deploy vLLM Endpoint]:::process
```

### 7.2 Model Lifecycle State Machine

```mermaid
%%{init: {'theme': 'dark'}}%%
stateDiagram-v2
    [*] --> DatasetPending
    DatasetPending --> DatasetValidated : validation OK
    DatasetPending --> DatasetError : format invalid or under 500 samples
    DatasetError --> DatasetPending : user corrects file
    DatasetValidated --> CostEstimated : config submitted
    CostEstimated --> Training : user confirms cost
    CostEstimated --> Cancelled : user rejects cost
    Training --> Evaluating : epochs completed
    Training --> Failed : GPU OOM or timeout
    Training --> Training : checkpoint saved
    Evaluating --> MergeReady : ROUGE-L improvement confirmed
    Evaluating --> Archived : no improvement vs base
    MergeReady --> Quantizing : merge triggered
    Quantizing --> Serving : GGUF ready + endpoint deployed
    Serving --> Archived : model deprecated
    Serving --> Training : drift detected retrain
```

### 7.3 Architecture Diagram

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'lineColor': '#a78bfa'}}}%%
graph TB
    classDef fe fill:#1e3a5f,stroke:#3b82f6,color:#fff
    classDef be fill:#14532d,stroke:#22c55e,color:#fff
    classDef gpu fill:#4a1d96,stroke:#a78bfa,color:#fff
    classDef storage fill:#7c2d12,stroke:#f97316,color:#fff
    classDef ext fill:#374151,stroke:#9ca3af,color:#fff

    UI["Next.js 14 Frontend"]:::fe
    API["FastAPI Python 3.11"]:::be
    Trainer["SFTTrainer LoRA/QLoRA"]:::gpu
    VLLM["vLLM Serving OpenAI API"]:::gpu
    HF["HuggingFace Hub Base Models"]:::ext
    WB["Weights and Biases Tracking"]:::ext
    PG[("PostgreSQL Jobs and Models")]:::storage
    GCS["GCS Checkpoints and GGUF"]:::storage

    UI --> API
    API --> Trainer
    API --> VLLM
    Trainer --> HF
    Trainer --> WB
    Trainer --> GCS
    VLLM --> GCS
    API --> PG
```

### 7.4 Deployment Sequence Diagram

```mermaid
%%{init: {'theme': 'dark'}}%%
sequenceDiagram
    participant User
    participant Frontend
    participant API
    participant CloudRun as Cloud Run GPU
    participant GCS
    participant vLLM

    User->>Frontend: Upload JSONL dataset
    Frontend->>API: POST /datasets/upload
    API-->>Frontend: dataset_id, validation_report
    User->>Frontend: Configure LoRA + confirm cost
    Frontend->>API: POST /jobs/start
    API->>CloudRun: Trigger training job
    CloudRun-->>API: job_id, status training
    loop Every 10 steps
        CloudRun-->>API: SSE metrics loss eval_loss
        API-->>Frontend: Stream metrics
    end
    CloudRun->>GCS: Save checkpoints
    CloudRun-->>API: status completed
    API->>CloudRun: Trigger merge + quantize
    CloudRun->>GCS: Save merged + GGUF
    API->>vLLM: Deploy endpoint
    vLLM-->>API: endpoint_url
    API-->>Frontend: model ready endpoint_url
```

### 7.5 Entity-Relationship Diagram

```mermaid
%%{init: {'theme': 'dark'}}%%
erDiagram
    DATASET {
        uuid id PK
        string name
        string gcs_path
        int sample_count
        jsonb column_stats
        string status
        timestamp uploaded_at
    }
    TRAINING_JOB {
        uuid id PK
        uuid dataset_id FK
        string base_model
        string method
        jsonb lora_config
        jsonb training_args
        float estimated_cost_usd
        float actual_cost_usd
        string status
        timestamp started_at
        timestamp completed_at
    }
    CHECKPOINT {
        uuid id PK
        uuid job_id FK
        int epoch
        float train_loss
        float eval_loss
        string gcs_path
        timestamp saved_at
    }
    MODEL_VERSION {
        uuid id PK
        uuid job_id FK
        int version
        string gcs_merged_path
        string gcs_gguf_path
        float rouge_l_base
        float rouge_l_finetuned
        float perplexity
        string endpoint_url
        string status
        timestamp deployed_at
    }
    INFERENCE_LOG {
        uuid id PK
        uuid model_version_id FK
        text prompt
        text response
        float latency_ms
        int tokens_generated
        timestamp ts
    }

    DATASET ||--o{ TRAINING_JOB : trains
    TRAINING_JOB ||--o{ CHECKPOINT : saves
    TRAINING_JOB ||--o| MODEL_VERSION : produces
    MODEL_VERSION ||--o{ INFERENCE_LOG : logs
```

---

## 8. API Reference

### Base URL : `https://api.finetunedlm.wikolabs.com/v1`

#### POST /datasets/upload

```json
Response 200:
{
  "dataset_id": "ds_4f9a2b",
  "sample_count": 3240,
  "format_valid": true,
  "split": {"train": 2916, "eval": 324},
  "estimated_tokens": 1850000
}
```

#### POST /jobs/start

```json
{
  "dataset_id": "ds_4f9a2b",
  "base_model": "mistralai/Mistral-7B-v0.1",
  "method": "qlora",
  "lora_config": {"r": 16, "lora_alpha": 32, "target_modules": ["q_proj", "v_proj"], "lora_dropout": 0.1},
  "training_args": {"num_train_epochs": 3, "per_device_train_batch_size": 4, "learning_rate": 2e-4}
}

Response 202:
{
  "job_id": "job_c7e1a4",
  "status": "queued",
  "estimated_cost_usd": 12.40,
  "estimated_minutes": 85
}
```

#### GET /jobs/{job_id}/stream

SSE stream:
```
data: {"step": 100, "epoch": 1, "train_loss": 2.41, "eval_loss": 2.38, "lr": 1.8e-4}
data: {"step": 200, "epoch": 1, "train_loss": 2.19, "eval_loss": 2.14, "lr": 1.5e-4}
```

#### POST /eval/compare

```json
Request:
{"job_id": "job_c7e1a4", "prompt": "Expliquez la procédure de remboursement."}

Response 200:
{
  "base_model": {"response": "...", "latency_ms": 320, "tokens": 87},
  "finetuned_model": {"response": "...", "latency_ms": 290, "tokens": 112},
  "rouge_l_base": 0.41,
  "rouge_l_finetuned": 0.73
}
```

#### POST /models/{model_id}/deploy

```json
Response 200:
{
  "endpoint_url": "https://inf.finetunedlm.wikolabs.com/v1",
  "model_name": "customer-support-fr-v1",
  "status": "serving"
}
```

#### POST /v1/chat/completions (serving endpoint — format OpenAI)

```json
{
  "model": "customer-support-fr-v1",
  "messages": [{"role": "user", "content": "Comment retourner une commande ?"}],
  "max_tokens": 256,
  "temperature": 0.7
}
```

---

## 9. UI Simulation Guide

### Écran 1 — Dataset Upload & Validation

- Zone de drag and drop avec preview des 5 premiers exemples
- Badge : nombre de samples, format valide / invalide
- Tableau : distribution des longueurs instruction/output
- Avertissement si PII détectés (emails, téléphones, IBAN)

### Écran 2 — Configuration Fine-Tuning

- Select : Mistral-7B ou Llama-3.1-8B
- Toggle : LoRA vs QLoRA 4-bit
- Sliders : r (8/16/32), learning rate, batch size, epochs
- Card Estimation en temps réel : GPU hours × tarif = coût USD
- Confirmation requise si coût > 50 USD

### Écran 3 — Training Live Dashboard

- LineChart train_loss et eval_loss en temps réel via SSE
- Barre de progression steps / total_steps
- Badge statut : QUEUED / TRAINING / CONVERGED / FAILED
- Lien W&B pour le run complet
- Tableau checkpoints : époque, eval_loss, bouton restaurer

### Écran 4 — Before / After Comparison

- Champ texte : saisir un prompt de test
- Deux colonnes : Modèle de base vs Modèle fine-tuné
- Scores ROUGE-L côte à côte
- Highlight visuel des différences de réponse
- Bouton Déployer si résultats satisfaisants

### Écran 5 — Model Registry

- Tableau : version, date, ROUGE-L, coût, endpoint, statut
- Toggle actif/inactif par version
- Bouton Comparer deux versions
- Lien vers le run W&B

---

## 10. Database Schema

```sql
CREATE TABLE datasets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    gcs_path VARCHAR(500) NOT NULL,
    sample_count INTEGER,
    column_stats JSONB,
    status VARCHAR(50) DEFAULT 'uploaded',
    uploaded_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE training_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_id UUID REFERENCES datasets(id),
    base_model VARCHAR(255) NOT NULL,
    method VARCHAR(50) NOT NULL,
    lora_config JSONB NOT NULL,
    training_args JSONB NOT NULL,
    estimated_cost_usd FLOAT,
    actual_cost_usd FLOAT,
    status VARCHAR(50) DEFAULT 'queued',
    wandb_run_id VARCHAR(255),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ
);

CREATE TABLE checkpoints (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES training_jobs(id),
    epoch INTEGER NOT NULL,
    train_loss FLOAT,
    eval_loss FLOAT,
    gcs_path VARCHAR(500),
    saved_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id UUID REFERENCES training_jobs(id),
    version INTEGER NOT NULL,
    gcs_merged_path VARCHAR(500),
    gcs_gguf_path VARCHAR(500),
    rouge_l_base FLOAT,
    rouge_l_finetuned FLOAT,
    perplexity FLOAT,
    endpoint_url VARCHAR(500),
    status VARCHAR(50) DEFAULT 'ready',
    deployed_at TIMESTAMPTZ
);

CREATE TABLE inference_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_version_id UUID REFERENCES model_versions(id),
    prompt TEXT,
    response TEXT,
    latency_ms FLOAT,
    tokens_generated INTEGER,
    ts TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_jobs_dataset ON training_jobs(dataset_id);
CREATE INDEX idx_checkpoints_job ON checkpoints(job_id, epoch);
CREATE INDEX idx_inference_model ON inference_logs(model_version_id, ts);
```

---

## 11. Infrastructure & Deployment

### cloudbuild.yaml

```yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/finetunedlm-backend:$SHORT_SHA', './backend']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/finetunedlm-backend:$SHORT_SHA']
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args:
      - run
      - deploy
      - finetunedlm-backend
      - '--image=gcr.io/$PROJECT_ID/finetunedlm-backend:$SHORT_SHA'
      - '--region=us-central1'
      - '--platform=managed'
      - '--gpu=1'
      - '--gpu-type=nvidia-a100-80gb'
      - '--memory=64Gi'
      - '--cpu=8'
      - '--timeout=7200'
      - '--concurrency=1'
      - '--no-allow-unauthenticated'
```

### Backend Dockerfile

```dockerfile
FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel
WORKDIR /app
RUN apt-get update && apt-get install -y git
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/tmp/huggingface
EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### requirements.txt

```
fastapi==0.111.0
uvicorn[standard]==0.30.0
transformers==4.41.0
peft==0.11.1
bitsandbytes==0.43.1
trl==0.9.4
datasets==2.19.2
accelerate==0.30.1
torch==2.3.0
wandb==0.17.0
sqlalchemy==2.0.30
asyncpg==0.29.0
psycopg2-binary==2.9.9
alembic==1.13.1
pydantic==2.7.1
rouge-score==0.1.2
google-cloud-storage==2.17.0
```

### docker-compose.yml

```yaml
version: '3.9'
services:
  backend:
    build: ./backend
    ports: ["8000:8080"]
    environment:
      - DATABASE_URL=postgresql://ftlm:ftlm@db:5432/finetunedlm
      - GCS_BUCKET=finetunedlm-artifacts-dev
    depends_on: [db]
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:8000
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: ftlm
      POSTGRES_PASSWORD: ftlm
      POSTGRES_DB: finetunedlm
    volumes: [pgdata:/var/lib/postgresql/data]
volumes:
  pgdata:
```

---

## 12. CI/CD Pipeline

```yaml
name: CI finetunedlm
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test-backend:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:16
        env:
          POSTGRES_USER: test
          POSTGRES_PASSWORD: test
          POSTGRES_DB: finetunedlm_test
        ports: ["5432:5432"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: '3.11'}
      - run: pip install -r backend/requirements.txt pytest pytest-asyncio httpx
      - run: pytest backend/tests/ -v
        env:
          DATABASE_URL: postgresql://test:test@localhost:5432/finetunedlm_test
          MOCK_GPU: "true"

  test-frontend:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: {node-version: '20'}
      - run: npm ci && npm run build
        working-directory: frontend

  deploy:
    needs: [test-backend, test-frontend]
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}
      - uses: google-github-actions/setup-gcloud@v2
      - run: gcloud builds submit --config cloudbuild.yaml
```

---

## 13. Kaggle Dataset

**Dataset : databricks-dolly-15k**

| Attribut | Valeur |
|---|---|
| Source | Databricks, Inc. via Kaggle / HuggingFace |
| Lignes | 15 011 paires instruction-réponse |
| Catégories | 8 : brainstorming, classification, closed_qa, generation, information_extraction, open_qa, summarization, creative_writing |
| Licence | CC BY-SA 3.0 |

**Téléchargement :**
```bash
pip install kaggle
kaggle datasets download -d databricks/databricks-dolly-15k
```

**Conversion au format finetunedlm :**
```python
import json
from datasets import load_dataset

ds = load_dataset("databricks/databricks-dolly-15k")
with open("backend/data/dolly_finetune.jsonl", "w") as f:
    for row in ds["train"]:
        line = {"instruction": row["instruction"], "input": row.get("context", ""), "output": row["response"]}
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
```

---

## 14. Local Development

### Installation

```bash
git clone https://github.com/Wikolabs/finetunedlm.git
cd finetunedlm

# Via Docker Compose
docker compose up --build

# Backend seul mode mock CPU
cd backend && pip install -r requirements.txt
MOCK_GPU=true uvicorn app.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

### Variables d'environnement

| Variable | Description | Exemple |
|---|---|---|
| DATABASE_URL | PostgreSQL connection | postgresql://user:pass@localhost:5432/finetunedlm |
| GCS_BUCKET | Bucket artefacts | finetunedlm-artifacts |
| HF_TOKEN | HuggingFace API token | hf_xxxx |
| WANDB_API_KEY | Weights and Biases | optionnel |
| GCP_PROJECT_ID | Projet GCP | wikolabs-prod |
| MOCK_GPU | Mode simulation CPU | true en dev |

---

*Wikolabs — finetunedlm · Domain-Specific LLM Fine-Tuning as a Service · v1.0.0*
