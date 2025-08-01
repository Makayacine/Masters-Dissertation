````markdown
# Web Application Penetration Test for the Comprehensive Multi-Omics Dashboard

This directory contains the Jupyter Notebook (`Web App Penetration Test.ipynb`) used to perform an automated security audit of the “Comprehensive Multi-Omics Dashboard” (a Python-Dash web app). The goal is to establish a reproducible framework for testing OWASP Top 10 vulnerabilities in a research context.

---

## 🎯 Overview & Methodology

We automate five core stages of a penetration test:

1. **Reconnaissance & Endpoint Discovery**  
   - Crawl all routes, static assets, and Dash callbacks.  
2. **Vulnerability Fuzzing**  
   - Inject payloads to detect XSS, SQLi, Command Injection, etc.  
3. **Security Configuration Analysis**  
   - Audit HTTP headers (CSP, HSTS, etc.) and sensitive file exposure.  
4. **Access Control Simulation**  
   - Test for IDOR by requesting unauthorized object IDs.  
5. **Client-Side Validation**  
   - Use headless Selenium to verify CSP enforcement and script-blocking.

---

## 🚀 How to Run

1. **Prerequisites**  
   - Python 3.9+  
   - Running app at `http://localhost:8280`  
   - Install dependencies:  
     ```bash
     pip install requests beautifulsoup4 selenium webdriver-manager
     ```
2. **Configure**  
   - In the first cell of `Web App Penetration Test.ipynb`, set:  
     ```python
     BASE_URL = "http://localhost:8280"
     ```
3. **Execute**  
   - Open the notebook in Jupyter:  
     ```bash
     jupyter notebook "Web App Penetration Test.ipynb"
     ```  
   - Run all cells in order. The final cell runs the full suite and prints results.

---

## 📊 Summary of Findings

- **Info Leakage**: Fixed by adding a global error handler to `app.py`.  
- **Injection Resistance**: No XSS/SQLi, thanks to strict CSP and in-memory data handling.  
- **Secure Headers**: All OWASP-recommended HTTP headers present; no sensitive files exposed.  
- **Access Control**: IDOR simulation confirmed only public data is accessible.  

The dashboard passed all automated checks, demonstrating a robust security posture suitable for public scientific use.

---

## 📂 File Structure

````

Web App Penetration Test/
├── Web App Penetration Test.ipynb
└── README.md


