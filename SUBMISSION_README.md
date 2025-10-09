# BLP Homework Submission Package

**Course:** Economics 600a, Fall 2025  
**Student:** Marek Chadim (marek.chadim@yale.edu)

## Files Included

### 1. BLP_hw_chadim.tex
Main LaTeX source file containing all homework questions, answers, and analysis.

**Key Features:**
- 15 questions covering BLP demand estimation and merger simulation
- Complete mathematical derivations and economic interpretations
- Tables with numerical results from Python implementation
- Bibliography with 5 academic references (BLP 1994, 1995, Berry & Haile 2021, Conlon & Mortimer 2021, Conlon & Gortmaker 2020)
- Citations using natbib package with ecta bibliography style

**To compile:**
```bash
pdflatex BLP_hw_chadim.tex
bibtex BLP_hw_chadim
pdflatex BLP_hw_chadim.tex
pdflatex BLP_hw_chadim.tex
```

### 2. BLP_hw_chadim.bib
BibTeX bibliography file with all academic references cited in the LaTeX document.

### 3. BLP_homework_code.py
Complete Python implementation extracted from the Jupyter notebook.

**Contents:**
- Data generation with simulated demand and cost shocks (Questions 1-2)
- Market equilibrium price solving using contraction mappings (Question 5)
- OLS estimation (Question 6)
- 2SLS instrumental variables estimation (Question 7)
- Nested Logit estimation (Question 8)
- BLP random coefficients logit with pyBLP (Question 9)
- Derivative convergence analysis (Question 10)
- Supply-side estimation and diversion analysis (Question 11)
- Merger simulation: within-nest (Firms 1&2) (Question 12)
- Cross-nest merger (Firms 1&3) (Question 13)
- Merger with efficiency gains (Question 14)
- Welfare analysis (Question 15)

**To run:**
```bash
# Install pyblp if needed
pip install pyblp

# Run the code (takes 3-5 minutes for BLP estimation)
python3 BLP_homework_code.py
```

**Note:** The code runs successfully and produces results matching the LaTeX document. Key results:
- Merger price effects: Sat 1 +6.34%, Sat 2 +6.07%
- BLP estimates with optimal instruments converge successfully
- Welfare decomposition with consumer/producer surplus changes

### 4. blp.ipynb
Original Jupyter notebook with all code and outputs (for reference).

## Verification

The Python code has been validated to:
1. ✓ Compile without syntax errors
2. ✓ Execute successfully with all necessary imports (numpy, pandas, matplotlib, scipy, statsmodels, pyblp)
3. ✓ Produce numerical results matching the LaTeX document
4. ✓ Generate all tables and figures referenced in the homework

## Key Results Summary

| Question | Topic | Key Result |
|----------|-------|------------|
| Q6 | OLS Estimation | Price coefficient: -1.058 |
| Q9 | BLP with Optimal IV | σ_satellite: 1.058, σ_wired: 1.019 |
| Q12 | Merger Simulation | Sat 1: +6.34%, Sat 2: +6.07% |
| Q15 | Welfare Analysis | ΔCS: -43,500, ΔPS: +53,400, ΔW: +9,900 |

All results in the Python code match the tables in the LaTeX document.

## Software Requirements

- **Python 3.9+** with packages:
  - numpy
  - pandas
  - matplotlib
  - scipy
  - statsmodels
  - pyblp (version 1.1.2 or higher)

- **LaTeX** with packages:
  - natbib (for citations)
  - amsmath, amssymb (for math)
  - booktabs (for tables)
  - geometry (for page layout)

## Notes

- The Python code is extracted directly from the executed Jupyter notebook using automated extraction
- All code has been validated to ensure proper indentation and syntax
- The LaTeX document compiles cleanly with no errors or warnings
- Bibliography uses econometrics standard (ecta.bst) formatting
