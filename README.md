# Demand Forecasting Using Random Forest

This project uses historical retail data to forecast product demand (i.e., `units_sold`) using machine learning. The aim is to build a robust regression model that helps predict sales based on various features such as store ID, SKU ID, pricing, and promotional flags.

---

## Dataset

The dataset used contains **150,150 rows** and the following columns:
- `record_ID`: Unique identifier (later dropped)
- `week`: Date of transaction (split into day, month, year)
- `store_id`, `sku_id`: Store and product identifiers
- `total_price`, `base_price`: Transaction prices
- `is_featured_sku`, `is_display_sku`: Binary promotional indicators
- `units_sold`: Target variable (continuous)

---

## Data Preprocessing

- Split `week` into `day`, `month`, and `year`
- Dropped `record_ID` as it added no predictive value
- Identified and removed outliers using the 99th percentile of `units_sold`

---

## Model Used

- **RandomForestRegressor** from `sklearn.ensemble`
- Trained/tested with 80/20 split
- Evaluated using:
  - `R² Score` (Accuracy)
  - `Root Mean Squared Error (RMSE)`

---

## Results

| Stage                     | R² Score | RMSE     |
|--------------------------|----------|----------|
| Before outlier removal   | 0.778    | 26.93    |
| After outlier removal    | 0.807    | 18.73    |
| After GridSearchCV tuning| 0.808    | ~        |

Best parameters from **GridSearchCV**:
```python
{'n_estimators': 200, 'min_samples_split': 3}
