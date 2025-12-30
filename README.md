# Anomaly Detection in Time Series Datasets

**Authors:** Dawid Sitnik, Władysław Sinkiewicz

## File Description

| File | Description |
| :--- | :--- |
| `LSTM_approach.ipynb` | Jupyter notebook containing the LSTM solution implementation. |
| `LSTM_approach.r` | The corresponding solution implemented in R (includes LSTM learning fragment). |
| `yahoo_notebook.Rmd` | R notebook containing Statistical, One-class SVM, ESD, and Isolation Forest methods. |
| `yahoo_notebook.html` | Knitted R notebook with evaluated results. |

## The Aim of The Project
The primary objective of this project is to detect anomalies in time series datasets. We utilized the *'A Benchmark Dataset for Time Series Anomaly Detection'* provided by Yahoo, which consists of real traffic data from Yahoo servers.

The data can be accessed here: [Yahoo Webscope Program](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

## Our Approach
We aim to compare classical unsupervised modeling approaches against modern deep learning methods:

1.  **Statistical & Machine Learning:** Approaches based on analyzing data distribution, distance from the mean, and clustering.
2.  **Deep Learning (LSTM):** A Recurrent Neural Network (LSTM) trained to predict the next values in a time series. Points where the predicted value differs significantly from the real value are treated as outliers.

## Dataset
This dataset is provided as part of the Yahoo Webscope program. It contains real and synthetic time-series with labeled anomalies. Timestamps are replaced by integers with an increment of 1, where each data point represents 1 hour of data.

> **Note:** The dataset contains anomalies marked by humans; therefore, labels may not be perfectly consistent. We selected this dataset to simulate real-life production issues where data is often imperfect.

Time-series used to evaluate model performance are based on real production traffic. These series vary in scale and length. We tested models on specific time series to simplify the detection task. Finding anomalies across disparate series often requires specialized parameter tuning, as per the "No Free Lunch" theorem.

**The dataset fields are:**
* `timestamp`
* `value`
* `is_anomaly`: Boolean indicating if the current value is considered an anomaly.

**Dataset snippet:**

| timestamp | value | is_anomaly |
| :--- | :--- | :--- |
| 1 | 5.86 | 0 |
| 2 | 5.95 | 0 |
| 3 | 5.92 | 0 |
| 4 | 5.47 | 0 |

### Exploratory Data Analysis
The dataset before preprocessing:

<div align="center">
  <img src="./pictures/before_preprocessing1.png" alt="Data before preprocessing"/>
</div>

Common exploratory data analysis tools include histograms. We also examine differenced data to account for trends.

**Histograms of untransformed (top) and differenced (bottom) data:**

![](https://imgur.com/zDFF4m6.png)
![](https://imgur.com/GDfuazo.png)

The histogram of untransformed data shows a normal distribution, which is expected as this specific sample has no trend. If a trend existed, differencing the data would be required to transform it into a normal-shaped distribution.

### Stationarity Check
We utilize the **Augmented Dickey-Fuller (ADF)** test to check for stationarity (stable statistical properties like mean and variance over time).

**ADF Test Result:**
```text
Augmented Dickey-Fuller Test
data:  series$value
Dickey-Fuller = -10.364, Lag order = 11, p-value = 0.01
alternative hypothesis: stationary
