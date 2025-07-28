
# Multi-Layered Exploratory Analysis of Colorectal Cancer Metabolomics

A reproducible, end-to-end exploratory data analysis (EDA) workflow to identify and prioritize candidate metabolic biomarkers in colorectal cancer (CRC) by integrating metabolomics, cheminformatics, and transcriptomics data.

---

## 📋 Contents

1. **Notebooks**
   - `EDA/EDA.ipynb`  
     Main Jupyter notebook implementing:
     - Data curation & preprocessing of raw metabolite measurements  
     - Cheminformatic annotation (SMILES, physicochemical & quantum-mechanical properties)  
     - KEGG pathway enrichment & SMARTS-guided enzyme inference  
     - Integration with TCGA transcriptomic data  
     - Data consolidation and export of analysis-ready datasets  

2. **Original Study Data**  
   - `ALL_sample_data.xlsx`  
   - `sample_info.xlsx`  
   - etc...

3. **Additional Datasets**  
   - `EDA/TCGA_COAD_RNASeq2Gene_counts.csv`  
   - `EDA/trrust_rawdata.human.tsv`

---

## 🛠️ Installation

1. **Clone this repository**  
   ```bash
   git clone https://github.com/Makayacine/Masters-Dissertation.git
   cd Masters-Dissertation


2. **Create & activate a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   Use the `requirements.txt` file to install all necessary packages:

   ```bash
   pip install -r requirements.txt``

---

## 🚀 Usage

Launch JupyterLab or Jupyter Notebook:

```bash
jupyter lab
```

Open and run `EDA/EDA.ipynb` from top to bottom.


