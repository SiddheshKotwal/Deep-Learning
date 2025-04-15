# 🚀 Demand Forecasting in Supply Chain Management using LSTM

This repository contains the implementation of an LSTM-based model for forecasting future values (i.e., demand) in a supply chain environment using retail inventory data. The project is motivated by the research paper *"Evaluation of Deep Learning with Long Short-Term Memory Networks for Time Series Forecasting in Supply Chain Management"* and applies similar techniques to a *Retail Store Inventory Forecasting* dataset from Kaggle.

---

## 📖 Overview

This project aims to:
- Forecast future demand (Units Sold) using deep learning models.
- Implement and compare two approaches:
  - **Standard LSTM**
  - **Bidirectional LSTM (BiLSTM)**
- Evaluate model performance using RMSE and MAE metrics.
- Provide visual insights by plotting actual vs. predicted values.

---

## 📚 Research Paper Summary

- **Title:** Evaluation of Deep Learning with Long Short-Term Memory Networks for Time Series Forecasting in Supply Chain Management
- **Focus:** The paper explores the effectiveness of LSTM networks to forecast demand in supply chain scenarios.
- **Methodology:** It emphasizes data preprocessing, normalization, sequence generation, and LSTM architecture design to handle nonlinear dependencies and irregular data fluctuations.
- **Results:** LSTM models demonstrated improved accuracy over traditional forecasting approaches, as confirmed via RMSE/MAE metrics.

---

## 🗃️ Dataset

- **Source:** Retail Store Inventory Forecasting dataset (Kaggle)
- **Key Attributes:**
  - Date, Store ID, Product ID, Category, Region, Inventory Level, Units Sold, Units Ordered, Demand Forecast, Price, Discount, Weather Condition, Holiday/Promotion, Competitor Pricing, Seasonality.
- **Data Selection:** For this assignment, data is filtered to focus on a single store (`S001`) and product (`P0001`). The primary focus is on forecasting the `Units Sold`.

---

## 🛠️ Methodology

1. **Data Exploration & Preprocessing:**
   - Load and inspect the dataset.
   - Filter the dataset for a specific store-product combination.
   - Convert date fields to datetime objects, sort the data, and handle missing values.
   
2. **Normalization:**
   - Normalize the `Units Sold` data using MinMaxScaler.

3. **Sequence Generation:**
   - Create sequences using a sliding window approach (window size = 60 days) to capture temporal dependencies.

4. **Model Development:**
   - Build a **Standard LSTM** model and a **Bidirectional LSTM (BiLSTM)** model.
   - Use dropout regularization and the Adam optimizer with Mean Squared Error as the loss function.

5. **Training and Evaluation:**
   - Split data into training (80%) and testing (20%) sets.
   - Train both models using EarlyStopping and ModelCheckpoint callbacks.
   - Evaluate using RMSE and MAE metrics.

6. **Visualization:**
   - Plot prediction vs. actual values for visual comparison.

---

## 📊 Visualizations

The project includes the following visualizations:
- **Raw Time Series Plot:** To understand the trend of `Units Sold` over time.
- **Scaled Data Plot:** To verify proper normalization of the data.
- **Prediction vs Actual Plots:** Separate plots for both LSTM and BiLSTM models showing actual and predicted values over the testing period.

---

## 📈 Our Model Performance

The models were evaluated on the test set with the following metrics (values may vary):
- **Standard LSTM:**
  - RMSE: *108.84*
  - MAE: *91.49*
- **Bidirectional LSTM:**
  - RMSE: *108.60*
  - MAE: *92.23*

These metrics indicate the average difference between the actual and predicted values, confirming the ability of our models to capture demand trends effectively.

---

## 🔍 Comparison with Research Paper Results

- **Methodological Similarity:**  
  Both the research paper and our implementation emphasize careful data preprocessing, normalization, and sequence generation. Our LSTM architectures, with dropout and the choice of hyperparameters, closely follow the paper’s guidelines.
  
- **Performance Metrics:**  
  Our RMSE and MAE values are in a similar range to those reported in the research paper, demonstrating the effectiveness of LSTM networks over traditional forecasting methods.
  
- **Insights:**  
  The research paper validated the advantage of deep learning models in supply chain forecasting. Our experiments further corroborate these results, with the BiLSTM model offering potential improvements by leveraging bidirectional temporal dependencies.

---

## 🎯 Conclusion

This project successfully demonstrates how LSTM-based models can be used for demand forecasting in supply chain management. Key conclusions are:
- **Effective Forecasting:**  
  Both the standard LSTM and BiLSTM models capture the complex, nonlinear behavior of retail demand.
- **Research Validation:**  
  Our results validate the findings of the referenced research paper, offering competitive performance as measured by RMSE and MAE.
- **Future Work:**  
  Future enhancements could include hyperparameter tuning, integration of additional features (e.g., weather data, promotional events), and exploring ensemble methods for further improvement.

---

## 🙏 Acknowledgements

- Research Paper: *"Pacella, Massimo & Papadia, Gabriele. (2021). Evaluation of deep learning with long short-term memory networks for time series forecasting in supply chain management. Procedia CIRP. 99. 604-609. 10.1016/j.procir.2021.03.081."*  
