# Omics‑DashApp

An interactive Dash dashboard for exploring enriched metabolite biomarker data derived from multi‑omics analysis of colorectal cancer.

---

## 🚀 Overview

Omics‑DashApp provides a user‑friendly web interface to visualize and filter metabolite intensity data, enabling rapid insights into biomarker expression patterns.

Key features:

* **Interactive Plots:** Scatter, bar, and heatmap visualizations of metabolite abundances
* **Dynamic Filtering:** Select subsets by sample group, metabolite class, or intensity thresholds
* **Deployment‑Ready:** Docker container and Fly.io configuration for one‑click deployment

---

## 🔧 Installation

### 1. Clone the repo

```bash
git clone https://github.com/Makayacine/Masters-Dissertation.git
cd Masters-Dissertation/Omics-DashApp
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running Locally

### 1. Set environment variables

Ensure your API keys or config (if any) go into `dashenv` (ignored by Git).

### 2. Start the app

```bash
python app.py
```

Open your browser at [http://localhost:8050](http://localhost:8050).

---

## 🐳 Docker Deployment

Build and run with Docker:

```bash
# Build image
docker build -t omics-dashapp .

# Run container
docker run -p 8050:8050 omics-dashapp
```

---

## ☁️ Fly.io Deployment

Configured via `fly.toml`. To deploy:

```bash
fly deploy
```

---

## 📁 File Structure

```plaintext
Omics-DashApp/
├── app.py                 # Main Dash application
├── enriched_metabolite_data.csv  # Input data(Original Study Data/)
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build definition
├── fly.toml               # Fly.io deployment config
└── .gitignore             # Ignored files (env, logs)
```

---

## 📝 License

MIT © 2025 Vince Mbanze
