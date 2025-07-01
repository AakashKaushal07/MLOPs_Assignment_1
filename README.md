# MLOPs Assignment 1

This assignment demands us to design, implement and automate a complete machine learning workflow to predict house prices using classical machine learning models.The dataset used is `boston_housing`,

The metric we are comparing here are MSE/R² in both cases i.e. with and without adjusting hyper parameters

## Overview
`regression.py` is the file which containts our regression models and the comparisons.
`utils.py` contains the utilities which are required to perform the assignment smoothly.
## Prerequisites

- Python 3.9+

## Directory Structure

```bash/
|-- .github/workflows/
| |-- ci.yml
|-- utils.py
|-- regression.py
|-- requirements.txt
|-- README.md
```

## How to Run
1. Use `pip install -r requirements.txt`. This will install all the required packages.
2. Run the regression models by using command `python regresssion.py`.