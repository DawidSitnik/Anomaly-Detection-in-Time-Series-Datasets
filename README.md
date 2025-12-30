# Anomaly Detection in Time Series Datasets
**Authors:** Dawid Sitnik, Władysław Sinkiewicz

## File Description

* **LSTM_approach.ipynb:** Jupyter notebook containing the Long Short-Term Memory (LSTM) solution.
* **LSTM_approach.r:** The R implementation of the LSTM solution, focusing on the network training fragment.
* **yahoo_notebook.Rmd:** R notebook containing Statistical, One-class SVM, ESD, and Isolation Forest methods.
* **yahoo_notebook.html:** Knitted R notebook showcasing the evaluated results.

---

## 1. Project Aim
The primary objective of this project is to detect anomalies within time-series datasets. We utilized the *'Benchmark Dataset for Time Series Anomaly Detection'* provided by Yahoo, which consists of real traffic data from Yahoo servers.

**Data Access:** [Yahoo Research Dataset](https://webscope.sandbox.yahoo.com/catalog.php?datatype=y)

### Our Approach
We compare classical unsupervised modeling approaches against modern deep learning techniques:
1.  **Statistical/Unsupervised:** Analyzing data distribution and distance from the mean (IQR, SVM, Isolation Forests).
2.  **Deep Learning:** Training an LSTM network to predict the next time series value. Discrepancies between predicted and actual values are treated as outliers.

---

## 2. Dataset Overview
The dataset is part of the Yahoo Webscope program. It contains both real and synthetic time-series with labeled anomalies.

* **Granularity:** Timestamps are integers with an increment of 1 (each point represents 1 hour).
* **Labels:** The dataset contains anomalies marked by humans; consistency varies.
* **Rationale:** We selected this dataset to simulate real-world production scenarios.

**Note:** Time series vary in scale and length. We focused on specific series to avoid the pitfalls of the "No Free Lunch" theorem—parameters tuned for one specific traffic pattern may not generalize to others without adjustment.

### Dataset Fields
| Field | Description |
| :--- | :--- |
| `timestamp` | Integer representing time sequence. |
| `value` | The metric value (e.g., server traffic). |
| `is_anomaly` | Boolean (0 or 1) indicating if the value is an anomaly. |

### Data Snippet
    timestamp  value  is_anomaly
    1          1      5.86   0
    2          2      5.95   0
    3          3      5.92   0
    4          4      5.47   0
    5          5      5.77   0
    6          6      5.73   0

<p align="center">
  <img src="./pictures/before_preprocessing1.png" alt="Data Before Preprocessing"/>
</p>



### Exploratory Data Analysis (EDA)
We utilized histograms to understand the data distribution. We also examined differenced data to account for trends.

* **Top Histogram:** Untransformed data showing a normal distribution (indicates no trend).
* **Bottom Histogram:** If trends existed, differencing would normalize the shape.

![](https://imgur.com/zDFF4m6.png)
![](https://imgur.com/GDfuazo.png)

### Stationarity Test
We verified stationarity (stable mean and variance over time) using the **Augmented Dickey-Fuller (ADF)** test.

    Augmented Dickey-Fuller Test
    data:  series$value
    Dickey-Fuller = -10.364, Lag order = 11, p-value = 0.01
    alternative hypothesis: stationary

**Result:** With a $p$-value of 0.01 ($< 0.05$), we reject the null hypothesis. The time series is likely stationary.

---

## 3. Statistical Approach (IQR)
This method detects anomalies by searching for extreme values based on the Interquartile Range (IQR). Since our histograms showed near-normal distributions, IQR is a robust and interpretable choice.

**Methodology:**
* **Parameter:** IQR Coefficient ($k$).
* **Logic:** A data point is an outlier if:
    $$Value > Q3 + k \cdot IQR \quad \text{or} \quad Value < Q1 - k \cdot IQR$$
    Where typically $k=1.5$.



**Results:**
* **F1 Score:** 98.37%
* **Precision:** 100%
* **Recall:** 96.79%

![](https://imgur.com/zmsPJuT.png)
*Confusion Matrix:*
![](https://imgur.com/q247OrW.png)

**Analysis:**
The model identifies all outliers (High Precision) but misses a small fraction (Recall). In production, the trade-off between Precision and Recall depends on the cost of false negatives (e.g., missing a vital health anomaly).

---

## 4. One-Class Support Vector Machine (OCSVM)
OCSVM learns a decision boundary to separate the "normal" class from the origin. It is effective for unsupervised anomaly detection in high-dimensional spaces.

### Embedding Process
Time series are unfolded into phase space using time-delay embeddings:
$$x_E(t) = [x(t-E+1), x(t-E+2), \dots, x(t)]$$
Where $E$ is the embedding dimension.

### Hyperparameters
We utilized Grid Search to optimize parameters, though initial domain knowledge estimates were accurate.
* **Kernel:** RBF (Radial Basis Function) - handles non-linearity well.
* **Nu ($\nu$):** 0.01 (Regularization parameter, upper bound on margin errors).
* **Embedding Dimension:** 5.

**Grid Search Results:**

| Window | Nu | Kernel | Score |
| :--- | :--- | :--- | :--- |
| 10 | 0.01 | Sigmoid | 0.995 |
| 7 | 0.01 | Sigmoid | 0.995 |
| 5 | 0.01 | Sigmoid | 0.994 |
| **5** | **0.01** | **Radial** | **0.993** |

**Results:**
* **F1 Score:** 99.11%
* **Precision:** 99.14%
* **Recall:** 99.07%

![](https://imgur.com/8IM8TlB.png)
*Confusion Matrix:*
![](https://imgur.com/A7p8SPp.png)

---

## 5. Seasonal Hybrid ESD Model
Seasonal Hybrid ESD (Extreme Studentized Deviant) builds upon the **Grubbs test** to detect anomalies while accounting for seasonality via time-series decomposition.

**Results:**
* **F1 Score:** 99.53%
* **Precision:** 100%
* **Recall:** 99.07%

![](https://imgur.com/9sZrzP3.png)
*Confusion Matrix:*
![](https://imgur.com/dMGfnYq.png)

---

## 6. Isolation Forests
Isolation Forest is a tree-based algorithm that isolates observations. Anomalies are "few and different," making them susceptible to isolation in the early stages of random partitioning (shorter path lengths in the tree).



**Results:**
* **F1 Score:** 99.43%
* **Precision:** 99.08%
* **Recall:** 99.78%

![](https://imgur.com/WosnGTf.png)
*Confusion Matrix:*
![](https://imgur.com/yIcH7em.png)

*Precision-Recall Curve (Flat due to class imbalance):*
![](https://imgur.com/JtLPTxB.png)

---

## 7. LSTM Neural Network Approach
We implemented a Long Short-Term Memory (LSTM) network, a Recurrent Neural Network (RNN) architecture capable of learning long-term dependencies in sequence data.



### Data Preprocessing
1.  **Normalization:**
    $$X_{norm} = \frac{X - \mu}{X_{max} - X_{min}}$$
2.  **Smoothing:** Rolling window (size = 5) to reduce noise.
3.  **Reshaping:** Converting data into 3D vectors: `(Samples, TimeSteps, Features)`.

<p align="center">
  <img src="./pictures/after_preprocessing.png" alt="Smoothed Signal"/>
</p>

### Architecture & Training
* **Lookback ($n$):** 5 previous timesteps.
* **Epochs:** 20.
* **Optimizer:** Adam.



**Performance:**
The model learned to predict the signal accurately on both training and testing sets, indicating no overfitting.

*Training Prediction:*
<p align="center"><img src="./pictures/train_r_vs_p.png"/></p>

*Testing Prediction:*
<p align="center"><img src="./pictures/test_r_vs_p.png"/></p>

### Anomaly Detection Logic
Anomalies were identified by calculating the absolute error between the Real ($y$) and Predicted ($\hat{y}$) values.

* **Max Error (Normal):** 0.151
* **Min Error (Anomaly):** 0.1639
* **Threshold:** Set at $5.4 \times \sigma$ (Standard Deviation).

This threshold allowed for near-perfect separation.

---

## Summary and Conclusion

We explored a spectrum of anomaly detection methods, from simple statistical rules to complex deep learning architectures.

### Model Comparison

| Model | F1 Score | Precision | Recall | Complexity |
| :--- | :--- | :--- | :--- | :--- |
| **Statistical (IQR)** | 98.31% | 100% | 96.69% | Low |
| **One-Class SVM** | 99.11% | 99.14% | 99.07% | Medium |
| **Isolation Forest** | 99.43% | 99.08% | 99.78% | Medium |
| **Hybrid ESD** | 99.53% | 100% | 99.07% | Medium |
| **LSTM** | **~100%** | **100%** | **100%** | High |

### Key Takeaways
1.  **Baseline (IQR):** Provided a strong starting point (98.31% F1) with high interpretability. Excellent for scenarios where implementation speed is key.
2.  **Machine Learning (SVM/Isolation Forest):** Offered a significant improvement in Recall over the baseline. Isolation Forest is particularly notable for its speed and lack of scalar assumptions.
3.  **Deep Learning (LSTM):** Achieved the highest performance, correctly identifying all anomalies without false positives. However, this came at the cost of higher computational resources and significant preprocessing effort.

For this specific dataset, the LSTM approach proved superior, but for general-purpose monitoring where resources are constrained, Isolation Forests offer an excellent trade-off between performance and complexity.
