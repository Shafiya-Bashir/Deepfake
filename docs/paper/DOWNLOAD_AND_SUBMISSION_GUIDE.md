# Research Paper Compilation and Submission Guide

This guide describes how to compile the IEEE research paper draft, generate high-resolution vector figures, and prepare the submission package.

## 1. Prerequisites
- LaTeX distribution: MacTeX / TeX Live (`pdflatex`, `bibtex`)
- Python plotting tools for figure generation: `matplotlib`, `seaborn`

## 2. Compiling the LaTeX Paper
```bash
cd docs/paper
pdflatex IEEE_TRANSACTIONS_RESEARCH_PAPER.tex
bibtex IEEE_TRANSACTIONS_RESEARCH_PAPER
pdflatex IEEE_TRANSACTIONS_RESEARCH_PAPER.tex
pdflatex IEEE_TRANSACTIONS_RESEARCH_PAPER.tex
```

## 3. Generating Paper Figures
Run the visualization script to regenerate all figures formatted specifically for IEEE two-column layout (3.5 inch column width, 300 DPI):
```bash
python3 scripts/generate_visualizations.py --paper-mode
```
Figures will be deposited into `docs/paper/figures/`.
