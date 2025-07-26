# Second-Hand iPhone Price Prediction

This project aims to predict the price of second-hand iPhones in Indonesia using machine learning techniques. The model leverages historical data from 2021 to 2025, which includes various features such as iPhone model, storage capacity, color, condition, and market category. The goal is to assist both buyers and sellers in determining reasonable prices for used iPhones.

## Project Overview

The dataset contains 1,000 rows and 11 columns, with the following features:

* **Model**: Used iPhone model (e.g., iPhone 11, iPhone 12, etc.)
* **Kapasitas\_GB**: Smartphone storage capacity in gigabytes.
* **Warna**: Color of the iPhone.
* **Tahun\_Pencatatan**: Year the ad for the used iPhone was published.
* **Harga\_IDR**: Selling price in Indonesian Rupiah (target variable).
* **Status**: Current condition status of the iPhone (e.g., "Used", "Like New").
* **Kondisi**: Specific condition of the iPhone (e.g., "Fine", "Usage Scratches").
* **Kategori\_Pasar**: Market category (e.g., "Official", "Ex-Inter").
* **Sumber**: Platform where the used iPhone is listed (e.g., Facebook Marketplace, OLX).
* **Lokasi**: Region where the iPhone is being sold.

## Steps Involved

### 1. Data Cleaning

* Checked for and removed duplicate data entries.
* Handled missing values and irrelevant columns (e.g., dropped the "ID" column since it was unique and not needed for prediction).
* Identified and left outliers that were important for market predictions (e.g., high-end iPhone models in the future).

### 2. Encoding

* Applied **One-Hot Encoding (OHE)** to transform categorical data (e.g., Model, Color, Condition) into binary columns for the machine learning model, ensuring that there was no implied ordinal relationship between categories.

### 3. Modeling

* Evaluated multiple machine learning models, including:

  * **Random Forest**
  * **K-Nearest Neighbors (KNN)**
  * **Decision Tree**
  * **XGBoost**
  * **Linear Regression**

* After hyperparameter tuning, **XGBoost** provided the best performance with an R² value of **0.9985**, showing excellent generalization capabilities.

### 4. Deployment

* Deployed the model using **Streamlit**, making it accessible to everyone without the need to download code or data. Users can simply input the details of a used iPhone to get an accurate price prediction.

You can access the live prediction model [here](https://second-hand-iphone-price-prediction-in-indonesia-raiffaza.streamlit.app/).

## Getting Started

### Prerequisites

To run this project locally, you will need to have the following installed:

* Python 3.x
* Libraries:

  * Pandas
  * NumPy
  * Scikit-learn
  * XGBoost
  * Streamlit

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/second-hand-iphone-price-prediction.git
   cd second-hand-iphone-price-prediction
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

### Usage

Once the app is running locally, you can:

* Input the iPhone details (Model, Storage Capacity, Color, etc.)
* View the predicted price for the used iPhone.

## Contributing

Feel free to fork the repository, create an issue, or submit a pull request if you have improvements, bug fixes, or enhancements to share!

## References

* [Outlier Treatment](https://letsdatascience.com/outlier-treatment/)
* [When Not to Remove Outliers from Data](https://amanxai.com/2025/01/09/when-not-to-remove-outliers-from-data/)
* [Predicting the Price of Used Electronic Devices Using Machine Learning](https://ijcrt.smiu.edu.pk/index.php/smiu/article/view/152/48)

