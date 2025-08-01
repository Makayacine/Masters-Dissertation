NB Please note that the notebook may take up to a minute to fully render due to the large amount of whitespace output produced by PySpark when displaying DataFrame previews.
# Machine Learning for Colorectal Cancer Metabolomic Biomarker Identification

This repository contains the Jupyter Notebook (`Machine Learning.ipynb`) detailing the machine learning workflow used to classify colorectal cancer (CRC) vs. normal tissue samples and identify influential metabolomic biomarkers. This work supports the thesis _“Exploring Metabolomic Signatures in Colorectal Cancer: A Data-Driven Approach for Biomarker Identification.”_

---

## 🎯 Aim

- **Primary:** Classify samples as CRC or Normal Control (NC) using metabolomic profiles.  
- **Secondary:** Identify key metabolites driving the classification and compare them to biomarkers from traditional statistical methods.

---

## 📊 Dataset

- **Source:** Kang et al. (2023), MetaboLights _MTBLS8090_  
- **Dimensions:** 70 samples (36 CRC, 34 NC) × 927 features  
- **Preprocessing:**  
  1. Log₂ transformation  
  2. Z-score standardization  
  3. Stratified 80/20 train / test split  

---

## ⚙️ Requirements

- **Python 3.8+**  
- Key libraries:  
  ```bash
  pip install pyspark torch xgboost shap scikit-learn pandas numpy matplotlib seaborn openpyxl


* (Optionally maintain exact versions via `requirements.txt`.)

---

## 🔄 Workflow

1. **Setup & Data Loading**

   * Initialize Spark session
   * Load raw metabolomics, sample info, and filtered statistical hits

2. **Data Preparation**

   * Reshape & join data tables in PySpark
   * Apply transformations to produce ML-ready feature matrix

3. **Model Training**

   * **ANN** (Multi-Layer Perceptron)
   * **FT-Transformer** (Tabular transformer)
   * **XGBoost** (Gradient-boosted trees)

4. **Model Evaluation**

   * Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   * Select best model (XGBoost, Recall = 0.8571)

5. **Model Interpretation**

   * Compute feature importances with SHAP
   * Identify top-24 metabolites

6. **Biomarker Comparison**

   * Venn diagram of SHAP vs. statistical hits

---

## 📈 Key Results

* **Best Classifier:** XGBoost correctly labeled 6/7 CRC test cases.
* **Biomarker Discovery:** SHAP uncovered 20 unique candidates not found via univariate filtering.
* **Insight:** XAI reveals multivariate metabolomic signatures invisible to traditional tests.

---

## 🚀 How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/Makayacine/Masters-Dissertation.git
   cd Machine Learning
   ```
2. **Place data files**

   * `ALL_sample_data.csv`
   * `sample_info.csv`
   * `NC_vs_CRC_filter.xlsx`
     in the same folder as `Machine Learning.ipynb`.
3. **Run the notebook**

   ```bash
   jupyter notebook "Machine Learning.ipynb"
   ```
4. **Execute cells** from top to bottom. All results and figures will be generated automatically.

---


## 📂 Repository Structure



Masters-Dissertation/
├── README.md
├── EDA/
│   ├── EDA.ipynb
│   ├── README.md
│   ├── requirements.txt
│   ├── kegg\_kgml\_files/
│   ├── TCGA\_COAD\_RNASeq2Gene\_counts.csv
│   ├── TCGA\_COAD\_RNASeq2Gene\_sampleInfo.csv
│   ├── trrust\_rawdata.human.tsv
│   └── smart\_motifs\_programmatic.py
│
├── Machine Learning/
│   ├── Machine Learning.ipynb
│   ├── README.md
│   └── requirements.txt
│
├── Original Study Data/
│   ├── ALL\_sample\_data.csv
│   ├── ALL\_sample\_data.xlsx
│   ├── Article.pdf
│   ├── hmdb\_anno.xlsx
│   ├── sample\_info.csv
│   ├── sample\_info.xlsx
│   ├── NC\_vs\_CRC\_filter.xlsx
│   ├── NC\_vs\_CRC\_filter\_anno.xlsx
│   ├── NC\_vs\_CRC\_info.xlsx
│   ├── NC\_vs\_CRC\_pca\_eigenvec.xlsx
│   ├── NC\_vs\_CRC\_pca\_eigenval.xlsx
│   └── NC\_vs\_CRC.AUC.xls
│
└── .gitignore

---

## ✍️ License

This work is licensed under the MIT License.

```
::contentReference[oaicite:0]{index=0}
```
