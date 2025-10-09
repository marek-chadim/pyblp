# Publication-Ready Figures for BLP Homework

## Overview

Two publication-quality figures have been created and integrated into the LaTeX document to enhance the visual presentation of key merger analysis results.

---

## Figure 1: Surplus Changes by Scenario

### Files Created
- **surplus_changes_merger.pdf** (38K) - Vector graphics for LaTeX
- **surplus_changes_merger.png** (193K) - Raster preview

### Location in Document
- **Section**: Question 15 (Welfare Analysis)
- **Position**: After welfare analysis table (~line 1256)
- **LaTeX Label**: `\label{fig:surplus_changes}`

### Content
Stacked bar chart comparing welfare effects of satellite merger (Firms 1 & 2) under two scenarios:
- **Left bar**: Merger without efficiency gains
- **Right bar**: Merger with 15% cost reduction

### Visual Elements
- **Stacked bars** showing:
  - Consumer Surplus (ΔCS) in blue (#0173B2)
  - Producer Surplus (ΔPS) in orange (#DE8F05)
- **Total Welfare (ΔW)** prominently labeled above each bar
- **Component values** labeled within bars
- **Zero reference line** for clarity
- **Professional styling**: 300 DPI, serif fonts, colorblind-friendly colors

### Key Results Displayed

| Scenario | ΔCS | ΔPS | ΔW | Verdict |
|----------|-----|-----|-----|---------|
| Without efficiency | -$18,043 | +$7,318 | -$10,725 | Welfare-reducing |
| With 15% cost cut | +$21,805 | +$55,186 | +$76,990 | Welfare-enhancing |

### Interpretation
The figure dramatically illustrates how productive efficiency transforms merger impacts. Without efficiency, consumer harm (-$18,043) exceeds producer gains (+$7,318), yielding net welfare loss. With 15% cost reduction, both consumers and producers benefit, demonstrating cost pass-through dominance over anti-competitive effects.

---

## Figure 2: Within-Nest vs Cross-Nest Merger Comparison

### Files Created
- **merger_comparison.pdf** (35K) - Vector graphics for LaTeX
- **merger_comparison.png** (164K) - Raster preview

### Location in Document
- **Section**: Question 13 (Cross-Nest Merger Analysis)
- **Position**: After merger comparison table (~line 1147)
- **LaTeX Label**: `\label{fig:merger_comparison}`

### Content
Grouped bar chart comparing price effects across all four products for two merger types:
- **Blue bars**: Within-nest merger (Firms 1 & 2, both satellite)
- **Yellow bars**: Cross-nest merger (Firms 1 & 3, satellite + wired)

### Visual Elements
- **Grouped bars** for each product (Sat 1, Sat 2, Wired 1, Wired 2)
- **Percentage labels** on each bar
- **Zero reference line**
- **Professional styling**: 300 DPI, colorblind-friendly palette
- **Informative note** about merger types

### Key Results Displayed

| Product | Within-Nest (%) | Cross-Nest (%) | Difference (pp) |
|---------|-----------------|----------------|-----------------|
| Sat 1 | 6.34 | 2.51 | -3.84 |
| Sat 2 | 6.07 | 0.13 | -5.94 |
| Wired 1 | 0.08 | 2.40 | +2.32 |
| Wired 2 | 0.08 | 0.10 | +0.03 |

### Interpretation
The figure clearly shows that merger impacts depend critically on substitution patterns. Within-nest merger (satellite competitors) produces large price increases (6.34%, 6.07%) for merging products due to high diversion ratios. Cross-nest merger (satellite + wired) yields moderate increases (2.51%, 2.40%) as lower cross-category substitution limits recapture incentives. Non-merging firms show asymmetric responses, confirming competitive pressure operates primarily within product categories.

---

## Technical Specifications

### Both Figures
- **Resolution**: 300 DPI
- **Format**: PDF (vector) + PNG (raster)
- **Font**: Serif (publication standard)
- **Colors**: Colorblind-friendly palette
- **Grid**: Light background grid for readability
- **Borders**: Black edge lines on bars (0.5pt)
- **Legend**: Clean, borderless design

### LaTeX Integration
Both figures use:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{filename.pdf}
\caption{...}
\label{fig:...}
\end{figure}
```

### Package Requirements
Added to preamble:
```latex
\usepackage{graphicx}
```

---

## Python Implementation

### Notebook Cells
- **Cell 74**: Welfare surplus figure (lines 1988-2094)
- **Cell 69**: Merger comparison figure (lines 1667-1761)

### Code Features
- Sets publication-quality matplotlib parameters
- Uses colorblind-friendly color schemes
- Adds detailed labels and annotations
- Saves both PDF and PNG formats
- Resets matplotlib parameters after saving

### Files Updated
- **blp.ipynb**: Contains both figure generation cells
- **BLP_homework_code.py** (76K): Now includes 43 code blocks with both figures

### To Regenerate
```bash
# Run the complete code
python3 BLP_homework_code.py

# Or run specific cells in the notebook
# Cell 74 for surplus changes
# Cell 69 for merger comparison
```

---

## File Inventory

All files located in: `/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/`

### Figure Files
1. surplus_changes_merger.pdf (38K)
2. surplus_changes_merger.png (193K)
3. merger_comparison.pdf (35K)
4. merger_comparison.png (164K)

### Document Files
5. BLP_hw_chadim.tex (65K) - Updated with both figures
6. BLP_hw_chadim.bib (1.6K) - Bibliography
7. blp.ipynb - Notebook with figure generation code
8. BLP_homework_code.py (76K) - Standalone Python with 43 code blocks

### Documentation
9. FIGURE_SUMMARY.md (4.1K) - Original welfare figure summary
10. FIGURES_SUMMARY.md (this file) - Complete figures documentation
11. SUBMISSION_README.md (3.4K) - Overall submission guide

---

## Benefits for Publication

### Visual Impact
Both figures provide immediate visual understanding of complex numerical results:
- Welfare tradeoffs become visually obvious
- Merger type differences are instantly apparent
- Professional quality suitable for academic publication

### Pedagogical Value
The figures effectively teach key concepts:
- **Figure 1**: Efficiency defense in merger analysis
- **Figure 2**: Role of substitution patterns in competitive effects

### Accessibility
- Colorblind-friendly palettes ensure universal readability
- Clear labels reduce reliance on color alone
- Professional styling meets journal standards

### Integration
- Figures placed immediately after relevant tables
- Detailed captions explain interpretation
- Cross-referenceable via LaTeX labels

---

## Cross-References in Text

To reference the figures in your LaTeX text:

```latex
% For welfare surplus figure
As shown in Figure~\ref{fig:surplus_changes}, the merger...

% For merger comparison figure
Figure~\ref{fig:merger_comparison} illustrates that within-nest...
```

---

## Summary Statistics

### Figure 1 (Welfare)
- Shows: 2 scenarios × 2 components (CS, PS) = 4 data points
- Highlights: Total welfare changes (+$76,990 vs -$10,725)
- Message: Efficiency can transform welfare-reducing to welfare-enhancing

### Figure 2 (Merger Comparison)
- Shows: 4 products × 2 merger types = 8 bars
- Highlights: 6.34% vs 2.51% for Sat 1 (within vs cross)
- Message: Substitution patterns drive merger impacts

Both figures use real data from the BLP estimation with optimal instruments, ensuring accuracy and consistency with the written analysis.

