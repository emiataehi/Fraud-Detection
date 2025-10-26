# Data Folder

## Dataset

This project uses the IEEE-CIS Fraud Detection dataset.

**Dataset size:** 590K+ transactions  
**Features:** 394 transaction attributes

### Data Structure:
- `processed/` - Processed data files
- Raw transaction data files

### Data Files:
- `train_transaction.csv` - Transaction data (590K+ records)
- `train_identity.csv` - Identity information
- Additional feature files (V, C, D, M columns)

**Note:** Data files not included in repository due to size constraints (>100MB GitHub limit).

**Data Source:** Kaggle IEEE-CIS Fraud Detection Competition

## Data Processing

Refer to:
- `notebooks/01_data_exploration.ipynb` - EDA and fraud pattern analysis
- `notebooks/02_Feature_Engineering.ipynb` - Feature engineering steps
```

5. Scroll down to "Commit new file"
6. Commit message: `Add data folder documentation`
7. Click **"Commit new file"**

### **Step 3: Create the processed subfolder**

1. Click **"Add file"** → **"Create new file"** again
2. Type: `data/processed/.gitkeep`
3. Leave the file empty
4. Commit message: `Add processed data folder`
5. Click **"Commit new file"**

---

## ✅ DONE! 

Now your GitHub repo will have:
```
fraud-detection/
├── data/
│   ├── README.md
│   └── processed/
│       └── .gitkeep
├── notebooks/
├── README.md
├── requirements.txt
└── .gitignore
