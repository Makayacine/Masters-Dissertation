# Masters-Dissertation: Multi-Omics Biomarker Discovery and Analysis

A comprehensive collection of analytical workflows, dashboards, and security evaluations supporting the thesis *“Exploring Metabolomic Signatures in Colorectal Cancer: A Data-Driven Approach for Biomarker Identification.”*

---

## 📜 Abstract

This dissertation integrates:

* **Exploratory Data Analysis (EDA):** Metabolomics, cheminformatics, and transcriptomics workflows to identify candidate biomarkers.
* **Machine Learning (ML):** Classification models and model interpretation to pinpoint diagnostic metabolites.
* **Interactive DashApp:** A web dashboard for real-time data exploration and visualization.
* **Security Auditing:** Automated penetration testing to validate web app robustness against OWASP Top 10 vulnerabilities.

Each component is self-contained for reproducibility, transparency, and future extension.

---

## 📂 Repository Structure

```plaintext
Masters-Dissertation/
├── EDA/                     # Metabolomics exploratory analysis
├── Machine Learning/        # Classification & biomarker ML
├── Omics-DashApp/           # Dash application for interactive visualization
├── Penetration Test/        # Automated security testing notebook
└── Original Study Data/     # Raw data files and metadata
```

---

## 🚀 Quick Start

1. **Clone this repository**

   ```bash
   git clone https://github.com/Makayacine/Masters-Dissertation.git
   cd Masters-Dissertation
   ```

2. **Create & activate a Python environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

3. **Run each module**

   * **EDA:**

     ```bash
     cd EDA
     jupyter notebook EDA.ipynb
     ```
   * **ML:**

     ```bash
     cd "Machine Learning"
     jupyter notebook "Machine Learning.ipynb"
     ```
   * **DashApp:**

     ```bash
     cd Omics-DashApp
     python app.py   # or use Docker: docker build -t omics-dashapp . && docker run -p 8050:8050 omics-dashapp
     ```
   * **Security Test:**

     ```bash
     cd "Penetration Test"
     jupyter notebook "Web App Penetration Test.ipynb"
     ```

---

## 📖 Module Descriptions

1. **EDA** – Data cleaning, annotation, pathway enrichment, and transcriptomic integration.
2. **Machine Learning** – Train ANN, FT-Transformer, XGBoost; interpret with SHAP; compare to univariate.
3. **Omics-DashApp** – Interactive Dash dashboard with Docker and Fly.io deployment.
4. **Penetration Test** – Automated OWASP Top 10 audit, including fuzzing, header checks, and IDOR simulation.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome. Please open a Pull Request against the `main` branch.

---

*© 2025 Vince Mbanze — Metascience Architect*
