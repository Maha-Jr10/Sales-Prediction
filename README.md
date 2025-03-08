# **📊 Sales Prediction Using Machine Learning**  

This project utilizes **machine learning** to predict sales revenue based on advertising budgets across **TV, Radio, and Newspaper** channels. The entire workflow, including **data analysis, model selection, hyperparameter tuning, and evaluation**, is implemented in a **Jupyter Notebook**.  

---

## **📂 Dataset**  
The dataset consists of advertising budgets and their corresponding **sales revenue**.

| Feature   | Description |
|-----------|------------|
| `TV`      | Budget spent on TV ads |
| `Radio`   | Budget spent on radio ads |
| `Newspaper` | Budget spent on newspaper ads |
| `Sales`   | Total sales revenue (target variable) |

---

## **🛠 Machine Learning Models Used**
The notebook implements and compares multiple **regression models** with **GridSearchCV** for hyperparameter tuning:

✅ **Decision Tree Regressor**  
✅ **Random Forest Regressor**  
✅ **Support Vector Machine (SVM) Regressor**  
✅ **Gradient Boosting Regressor**  
✅ **XGBoost Regressor**  
✅ **Lasso Regression**  
✅ **Ridge Regression**  

Each model is evaluated using **cross-validation**, and the best-performing model is selected.

---

## **📊 Model Evaluation Metrics**
The models are compared using:  

✔ **R-Squared Score (R²)** – Measures accuracy.  
✔ **Mean Squared Error (MSE)** – Measures error magnitude.  
✔ **Feature Importance Analysis** – Identifies key predictors.  
✔ **Residual Analysis** – Ensures model validity.  
✔ **Actual vs. Predicted Scatter Plot** – Visualizes performance.  

The **best model** is automatically selected based on the highest **R² score**.

---

## **📝 Notebook Workflow**
### **1️⃣ Install Dependencies**
Ensure the required Python libraries are installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### **2️⃣ Load and Explore Data**
- The dataset is loaded using **Pandas**.
- Basic statistics and visualizations are generated, including **histograms, boxplots, pairplots, and a heatmap** to understand correlations.

### **3️⃣ Train Machine Learning Models**
- The dataset is split into **training (80%)** and **testing (20%)** sets.
- Several **machine learning models** are trained and fine-tuned using **GridSearchCV**.
- The best model is selected automatically based on **cross-validation performance**.

### **4️⃣ Model Evaluation and Visualization**
- The selected model is tested on the **unseen test data**.
- **Predictions** are compared against actual sales.
- Various plots are generated:
  - 📉 **Actual vs. Predicted Sales Plot**  
  - 📊 **Residual Distribution Plot**  
  - 📈 **Residuals vs. Predicted Sales Plot**  
  - 🔍 **Feature Importance (for tree-based models)**  

### **5️⃣ Make New Predictions**
Once the best model is selected, you can predict **sales revenue** for new advertising budgets:
```python
import pandas as pd

# Example: TV = 257, Radio = 50, Newspaper = 44
new_data = pd.DataFrame([[257, 50, 44]], columns=['TV', 'Radio', 'Newspaper'])
prediction = best_model.predict(new_data)
print("Predicted Sales:", prediction)
```

---

## **📌 How to Run the Notebook**
1️⃣ Open the Jupyter Notebook:  
```bash
jupyter notebook
```
2️⃣ Load `sales_prediction.ipynb`.  
3️⃣ Run each cell step by step.  
4️⃣ View **model performance and visualizations**.  
5️⃣ Use the **best model** to make new sales predictions.

---

## **📬 Contact**
If you have any questions or suggestions, feel free to reach out! 🚀
