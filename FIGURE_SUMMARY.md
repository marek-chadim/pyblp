# Publication-Ready Figure: Surplus Changes by Scenario

## Created Files

1. **surplus_changes_merger.pdf** (38K)
   - High-resolution PDF figure for LaTeX inclusion
   - 300 DPI, vector graphics
   - Publication-ready quality

2. **surplus_changes_merger.png** (193K)
   - PNG preview version
   - 300 DPI raster image

## Figure Details

### Content
The figure visualizes welfare effects of the satellite merger (Firms 1 & 2) under two scenarios:
- **Left bar**: Merger without efficiency gains
- **Right bar**: Merger with 15% cost reduction

### Visual Elements
- **Stacked bars** showing:
  - Consumer Surplus changes (ΔCS) in blue (#0173B2)
  - Producer Surplus changes (ΔPS) in orange (#DE8F05)
- **Total Welfare (ΔW)** labeled above each bar
- **Component values** labeled within each bar segment
- **Zero reference line** for easy interpretation
- **Professional styling**:
  - Colorblind-friendly palette
  - Clean grid background
  - Serif fonts for academic publication
  - Black edge lines on bars

### Key Results Displayed

**Scenario 1: Without Efficiency**
- ΔCS: -$18,043 (consumer harm)
- ΔPS: +$7,318 (producer gain)
- ΔW: -$10,725 (net welfare loss)

**Scenario 2: With 15% Cost Cut**
- ΔCS: +$21,805 (consumer benefit)
- ΔPS: +$55,186 (producer benefit)
- ΔW: +$76,990 (net welfare gain)

## Integration with LaTeX Document

### Location
The figure has been inserted in the LaTeX document at:
- **Section**: Question 15 (Welfare Analysis)
- **Position**: Immediately after the welfare analysis table
- **Line**: ~1256 in BLP_hw_chadim.tex

### LaTeX Code Added
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{surplus_changes_merger.pdf}
\caption{Welfare Effects of Satellite Merger (Firms 1 \& 2). 
Stacked bars show consumer surplus (blue) and producer surplus (orange) 
changes for two scenarios: merger without efficiency gains (left) and 
merger with 15\% cost reduction (right). Total welfare change ($\Delta W$) 
is shown above each bar. Without efficiency, the merger reduces total 
welfare by \$10,725 as consumer harm exceeds producer gain. With 15\% 
cost reduction, both consumer and producer surplus increase, yielding a 
total welfare gain of \$76,990. Values represent aggregate changes across 
all 600 markets with 1,000 consumers per market.}
\label{fig:surplus_changes}
\end{figure}
```

### Package Requirements
Added to preamble:
```latex
\usepackage{graphicx}
```

## Python Code

The figure generation code has been added to:
- **Notebook cell 74** in blp.ipynb
- **BLP_homework_code.py** (updated with 42 code blocks total)

### Code Features
- Sets publication-quality matplotlib parameters (300 DPI)
- Uses colorblind-friendly color scheme
- Adds detailed labels and annotations
- Saves both PDF (for LaTeX) and PNG (for preview)
- Resets matplotlib parameters after saving

### To Regenerate
```python
# Run cell 74 in the notebook, or:
python3 BLP_homework_code.py
```

## Figure Interpretation

The figure clearly illustrates the trade-off between anti-competitive effects 
and efficiency gains in merger analysis:

1. **Without efficiency**: The merger harms consumers through higher prices 
   (-$18,043 CS), benefiting producers through increased markups (+$7,318 PS). 
   Net effect is welfare-reducing (-$10,725).

2. **With 15% cost reduction**: Cost pass-through dominates unilateral effects. 
   Both consumers and producers benefit (+$21,805 CS, +$55,186 PS), yielding 
   substantial welfare gains (+$76,990).

The visual contrast between the two scenarios effectively demonstrates how 
productive efficiency can transform a harmful merger into a beneficial one.

## File Locations

All files are in: `/Users/marek/Library/CloudStorage/Dropbox/github/pyblp/`

- surplus_changes_merger.pdf
- surplus_changes_merger.png
- BLP_hw_chadim.tex (updated with figure)
- blp.ipynb (cell 74 has generation code)
- BLP_homework_code.py (updated)

## Notes

- The figure uses data from the welfare analysis in Question 15
- Values assume M_t = 1,000 consumers per market across T = 600 markets
- Figure label is \label{fig:surplus_changes} for cross-referencing
- Figure placement uses [h] (here) specifier for proximity to text

