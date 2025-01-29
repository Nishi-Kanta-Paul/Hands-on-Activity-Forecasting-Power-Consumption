# Power Consumption Forecasting Project

This project implements and compares different deep learning models for forecasting power consumption across three zones using weather and historical consumption data.

## Table of Contents

- [Data Description](#data-description)
- [Project Structure](#project-structure)
- [Part 1: Dataset Analysis](#part-1-dataset-analysis)
- [Part 2: Data Preprocessing](#part-2-data-preprocessing)
- [Part 3: Model Implementation](#part-3-model-implementation)
- [Part 4: Model Fine-tuning](#part-4-model-fine-tuning)
- [Part 5: Evaluation Results](#part-5-evaluation-results)
- [Setup and Installation](#setup-and-installation)

## Data Description

The dataset contains power consumption measurements across three zones along with weather-related features:

- **Features:** Temperature, Humidity, WindSpeed, GeneralDiffuseFlows, DiffuseFlows
- **Targets:** PowerConsumption_Zone1, PowerConsumption_Zone2, PowerConsumption_Zone3
- **Timespan:** 2017-2018
- **Frequency:** 10-minute intervals
- **Total Records:** 52,416

## Project Structure

```
├── data/
│   ├── powerconsumption.csv
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── vanilla_transformer.py
│   ├── patchtst.py
│   └── train_evaluate.py
├── notebooks/
│   └── model_analysis.ipynb
└── requirements.txt
```

## Part 1: Dataset Analysis

### Key Features Analysis
- **Temperature range:** 3.25°C to 40.01°C
- **Humidity range:** 11.34% to 93.60%
- **Wind Speed range:** 0.05 to 4.98 m/s
- **Power consumption patterns vary significantly across zones**

### Time Series Challenges
1. Seasonal patterns in power consumption
2. Weather dependency variations
3. Different consumption patterns across zones

### Weather Impact Analysis
- Temperature shows strong correlation with consumption
- Humidity has moderate impact
- Wind speed shows minimal correlation

## Part 2: Data Preprocessing

### Data Processing Steps
1. Missing value handling
2. Normalization using min-max scaling
3. Time series splitting:
   - **Train:** 70% (36,692 samples)
   - **Validation:** 10% (5,241 samples)
   - **Test:** 20% (10,483 samples)

### Data Tokenization
- **Sequence length:** 168 (1 week of 10-minute intervals)
- **Feature dimension:** 8 (5 weather features + 3 zone consumptions)
- **Sliding window approach with 50% overlap**

## Part 3: Model Implementation

### Models Implemented
1. **Vanilla Transformer**
   - Multi-head attention layers
   - Position-wise feed-forward networks
   - Positional encoding

2. **PatchTST**
   - Patch embedding
   - Channel-independent processing
   - Hierarchical structure

### Architecture Comparison
| Feature                | Vanilla Transformer | PatchTST       |
|------------------------|---------------------|----------------|
| **Input Processing**   | Sequential          | Patched        |
| **Attention Mechanism**| Global              | Local + Global |
| **Parameter Efficiency**| Lower               | Higher         |
| **Training Speed**     | Slower              | Faster         |

## Part 4: Model Fine-tuning

### Hyperparameters
- **Learning rate:** 0.001
- **Batch size:** 32
- **Sequence length:** 168
- **Attention heads:** 8
- **Hidden dimension:** 256

### Training Configuration
- **Optimizer:** AdamW
- **Learning rate scheduler:** ReduceLROnPlateau
- **Early stopping patience:** 10
- **Dropout rate:** 0.1

## Part 5: Evaluation Results

### Metrics Used
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Square Error)
- R² Score
- MAPE (Mean Absolute Percentage Error)

### Performance Summary
| Zone  | Best Model | RMSE  | R² Score |
|-------|------------|-------|----------|
| Zone 1| PatchTST   | 0.0842| 0.9123   |
| Zone 2| Transformer| 0.0756| 0.9245   |
| Zone 3| PatchTST   | 0.0891| 0.9078   |

### Key Findings
1. PatchTST performs better in zones with high variability
2. Transformer shows better stability in regular patterns
3. Both models handle seasonal patterns effectively

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nishi-Kanta-Paul/Hands-on-Activity-Forecasting-Power-Consumption.git
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run preprocessing:**
   ```bash
   python src/data_preprocessing.py
   ```

4. **Run model training and evaluation:**
   ```bash
   python src/train_evaluate.py
   ```

5. **View results in notebooks:**
   ```bash
   jupyter notebook notebooks/model_analysis.ipynb
   ```
```