# 💳 Fraud Detection System - E-commerce Transaction Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

## 📊 Project Overview

A comprehensive fraud detection system analyzing 100,000 e-commerce transactions to identify fraudulent patterns and build predictive models. This project demonstrates end-to-end data science workflow from exploratory analysis to feature engineering for machine learning deployment.

**Key Challenge:** Highly imbalanced dataset (2.56% fraud rate, 1:38 ratio) requiring specialized techniques beyond simple accuracy metrics.

---

## 🎯 Business Impact

**Fraud Prevention Insights:**
- Identified **Product C** with 8.78% fraud rate (3.4x baseline risk)
- Discovered early morning transactions (4-9am) show 2x baseline fraud rate
- Found credit cards are 1.8x riskier than debit cards
- Quantified compound risk factors for targeted intervention

**Potential Value:**
- Reduced false positives through risk-based authentication
- Optimized fraud team resources with predictive scoring
- Improved customer experience with smart friction

---

## 📁 Dataset

**Source:** IEEE-CIS Fraud Detection (Vesta Corporation)
- **Size:** 100,000 transactions (sampled from 590k due to memory constraints)
- **Features:** 394 columns including transaction details, card info, device data
- **Target:** Binary classification (isFraud: 0 = legitimate, 1 = fraud)
- **Time Period:** 6 months of real e-commerce data

---

## 🔍 Key Findings

### 1. Temporal Patterns
- **High-risk hours:** 4am-9am (fraud rate up to 5.2%)
- **Safe hours:** 1pm-5pm (fraud rate 1.3%)
- **High-risk days:** Monday/Tuesday (2.9% fraud)
- **Pattern:** Fraudsters exploit low-activity periods when cardholders sleep

### 2. Product Type Analysis
| Product | Fraud Rate | Avg Amount | Risk Level |
|---------|-----------|-----------|------------|
| Product C | **8.78%** | $49 | **CRITICAL** |
| Product S | 2.34% | $58 | Low |
| Product H | 1.98% | $72 | Low |
| Product W | 1.85% | $153 | Low |
| Product R | 1.08% | $188 | Very Low |

**Key Insight:** Product C (likely digital goods/gift cards) has highest fraud despite lowest value - fraudsters target items below detection thresholds.

### 3. Transaction Amount Patterns
- **Riskiest range:** $100-500 (2.89% fraud)
- **Safest range:** $1000+ (1.50% fraud)
- **Strategy:** Fraudsters target "sweet spot" - high enough to profit, low enough to avoid flags

### 4. Email Domain Risk
- **Highest risk:** outlook.com (6.91%), hotmail.com (4.63%)
- **Lowest risk:** ISP emails like att.net, comcast.net (0.86%)
- **Critical finding:** Same purchaser/recipient email (P=R) shows 5.17% fraud vs 1.69% for different emails

### 5. Card Information
- **Credit cards:** 3.60% fraud
- **Debit cards:** 2.00% fraud
- **American Express:** 1.14% fraud (premium security)
- **Mastercard:** 2.88% fraud (highest among brands)

### 6. Meta-Pattern Discovery
**Universal trend:** Low transaction volume/unusual behavior correlates with high fraud
- Observed across temporal, product, amount, and email features
- Exception: Premium products (American Express) with robust security

---

## 🛠️ Technical Approach

### Phase 1: Exploratory Data Analysis ✅
```
✓ Target variable analysis (class imbalance assessment)
✓ Transaction amount distributions and binning
✓ Temporal pattern analysis (hour/day of week)
✓ Missing data assessment (319/394 columns affected)
✓ Product type analysis
✓ Email domain analysis  
✓ Card information analysis
✓ Meta-pattern identification
```

### Phase 2: Feature Engineering 🔄 (In Progress)
```
✓ Temporal features (hour, day, risk scores, categories)
✓ Missing data flags (9 predictive columns)
✓ Amount-based features (risk categories and scores)
✓ Email features (domain risk, match flags)
✓ Product features (risk scores, binary flags)
✓ Card features (type/brand risk scores)
⏳ Interaction features (compound risk factors)
⏳ Categorical encoding (one-hot, label encoding)
```

### Phase 3: Model Building ⏳ (Upcoming)
```
⏳ Handle class imbalance (SMOTE, class weights)
⏳ Train multiple models (Logistic Regression, Random Forest, XGBoost)
⏳ Hyperparameter tuning
⏳ Model evaluation (Precision, Recall, F1, ROC-AUC)
⏳ Feature importance analysis
```

### Phase 4: Deployment & Documentation ⏳ (Upcoming)
```
⏳ Risk scoring system implementation
⏳ Business recommendations
⏳ Final documentation
```

---

## 📦 Technologies Used

**Programming & Libraries:**
- Python 3.11
- Pandas (data manipulation)
- NumPy (numerical computing)
- Matplotlib & Seaborn (visualization)
- Scikit-learn (machine learning)

**Environment:**
- Anaconda (conda environments)
- Jupyter Notebook
- VS Code

**Version Control:**
- Git & GitHub

---

## 📊 Key Visualizations

### Fraud Rate by Hour of Day
![Hour Analysis](visualization/hour_by_fraud_rate.png)
*Early morning (4-9am) shows 2x baseline fraud rate*

### Fraud Rate by Product Type
![Product Analysis](visualization/product_category_by_fraud_rate.png)
*Product C exhibits 8.78% fraud rate - 3.4x baseline*

### Fraud Rate by Day of Week
![Day Analysis](visualization/day_in_week_by_fraud_rate.png)
*Monday shows highest fraud; weekends safer despite higher volume*

---

## 🚀 Getting Started

### Prerequisites
```bash
- Python 3.11+
- Anaconda or pip
- 8GB+ RAM recommended
```

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Create conda environment
```bash
conda create -n fraud_env python=3.11
conda activate fraud_env
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download dataset
- Place `train_transaction.csv` and `train_identity.csv` in `data/raw/` folder

5. Run analysis
```bash
jupyter notebook notebooks/01_data_exploration.ipynb
```

---

## 📂 Project Structure
```
fraud-detection/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Processed datasets
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA & insights
│   ├── 02_feature_engineering.ipynb   # Feature creation
│   ├── 03_model_building.ipynb        # Model training
│   └── 04_evaluation.ipynb            # Results & metrics
├── models/                     # Saved models
├── visualizations/            # Charts and graphs
├── README.md
└── requirements.txt
```

---

## 📈 Results & Insights Summary

### Critical Risk Factors Identified:

**Maximum Risk Scenario (15-20% fraud probability):**
- Product C purchase
- Amount: $100-500
- Time: 10am-2pm on Monday
- Email: outlook.com with P=R match
- Card: Mastercard credit

**Minimum Risk Scenario (<0.5% fraud probability):**
- Product R purchase
- Amount: $1000+
- Time: 3pm on Thursday
- Email: ISP domain with P≠R
- Card: American Express

---

## 🎓 Skills Demonstrated

### Technical Skills
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Handling Imbalanced Datasets
- Statistical Analysis
- Data Visualization
- Machine Learning Classification
- Python Programming

### Business Skills
- Pattern Recognition
- Risk Assessment
- Business Impact Quantification
- Stakeholder Communication
- Strategic Recommendations

---

## 🔮 Future Enhancements

- [ ] Real-time fraud detection API
- [ ] Interactive dashboard (Streamlit/Dash)
- [ ] Deep learning models (LSTM for sequence patterns)
- [ ] Explainable AI (SHAP values for model interpretability)
- [ ] A/B testing framework for fraud rules
- [ ] Cost-benefit analysis calculator

---

## 👤 Author

**[IGEIN EMIATAEHI HOPE]**
- LinkedIn: [linkedin](linkedin.com/in/emi-igein-b024a8147/)
- Portfolio: [GitHub](https://github.com/emiataehi/fraud-detection)
- Email: [emi.igein@gmail.com]

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- Dataset: IEEE-CIS Fraud Detection / Vesta Corporation
- Inspiration: Real-world fraud detection challenges in e-commerce
- Learning: Part of Data Science portfolio development

---

## 📝 Project Status

**Current Phase:** Feature Engineering (60% complete)

**Last Updated:** [25/10/2025]

**Next Milestone:** Model Building & Evaluation