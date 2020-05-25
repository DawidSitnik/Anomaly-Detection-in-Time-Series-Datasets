# Anomaly Detection in Time Series Datasets
*authors: Dawid Sitnik, Władysław Sinkiewicz*

**File Description**

* LSTM_approach.ipynb - Jupiter notebook in which LSTM solution is made
* LSTM_approach.r - the same solution made in R with the extraction of the fragment responsible for learning LSTM network.
* yahoo_notebook.Rmd - R notebook that contains statistcal methods, One-class SVM and ESD methods for time series anomaly detection.
* yahoo_notebook.html - knited R notebook with evaludated results.

## The Aim of The Project
The main goal is to detect anomalies in the time series dataset. As the dataset, we decided to choose data shared by Yahoo called *'A Benchmark Dataset for Time Series Anomaly Detection'*, which is the real traffic data from Yahoo servers. 

The data can be accessed from this URL:
[Yahoo dataset](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

## Our Approach
To detect anomalies we are going to create some models which will be learned on the dataset which doesn't consist of any anomalies. Then we will make predictions on the dataset which consists also the data with the anomaly. Assuming that our models will work properly, predictions in which values are much different than real values will be treated as anomalies.

In our project, we would like to compare classical approaches to modeling with a more modern one. The first group of solutions will be based on transforming dataset into its vector representation of time series (for example - values history for certain period with its eventual aggregation in smaller sub-windows in different variants) and then use one of the classical algorithms like linear regression, random forest classifier, etc. to create the model. In the second approach, we will use the LSTM neural network which will work only on historical values from the time series.

Eventually, we will also try to find anomalies with ESD models. This model is mostly used for forecasting, but their application can be extended to anomaly detection.

## Dataset
This dataset is provided as part of the Yahoo! Webscope program, to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with Yahoo! Dataset contains real and synthetic time-series with labeled anomalies. Timestamps are replaced by integers with an increment of 1, where each data-point represents 1 hour worth of data. The anomalies are marked by humans and therefore may not be consistent.

**The dataset fields are:**
* *timestamp*
* *value*
* *is_anomaly*
    
The is_anomaly field is a boolean indicating if the current value at a given timestamp is considered an anomaly.

**Dataset snippet:**

```
  timestamp value is_anomaly
1         1  5.86          0
2         2  5.95          0
3         3  5.92          0
4         4  5.47          0
5         5  5.77          0
6         6  5.73          0
```

The dataset before preprocessing:

![](https://imgur.com/qkjA3oo.png){width=80%}

Common exploratory data analysis tool in time-series and non-time series data are histograms. We will also look at the differenced data because we want to use our timestamp (time) axis.

The histogram of untransformed data

![](https://imgur.com/M2phBfi.png){width=80%}

![](https://imgur.com/ovzO405.png){width=80%}


The histogram of untransformed data (top) shows normal distributions. This is to be expected given that the underlying sample (1st in our case) has no trend. If series had trend, we would difference the data to remove it, and this would transform the data to a more normal shaped distribution (bottom).

Next, we would like to know if time-series is *stationary*. We do this because many traditional statistical time series models rely on time series with such characteristic. In general, time series is *stationary* when it has fairly stable statistical properties over time, particularly mean and variance. The Augmented Dickey-Fuller (ADF) test is the most commonly used metric to access a time series for stationary problems. That test focuses on whether the mean of a series is changing, a variance isn’t tested here. 

ADF test result of one of the the Yahoo time series.

```
    Augmented Dickey-Fuller Test

data:  series$value
Dickey-Fuller = -10.364, Lag order = 11, p-value = 0.01
alternative hypothesis: stationary
```

Depending on the resulst of the test, null hypothesis can be rejected for a specific significance level - *p*-value. Conventionally, if *p*-value is less 0.05, the time series is likely stationany, whereas a *p* > 0.05 provides no such evidence.

Common exploartory times series methods is identifying where the series has a self-correlation. Self-corelation of a time series is the idea that a value in a time series at one give point in time has a correlation to the value at another point in time. Autocorrelation on the other hand asks more general question of whether there is a correlation between two points in a specific time series with a specific fixed distance between them.

We can apply the Box-Pierce test to the data to know whether or not the data is autocorrelated.

```
    Box-Pierce test

data:  series$value
X-squared = 1128.3, df = 1, p-value < 2.2e-16
```

Same as with ADF test we can see how likely times series is autocorrelated depending on the *p*-values.

We can graph the autocorrelation function to dig further into the data.

![](https://imgur.com/kyFbXmR.png){width=80%}

### Data Preprocessing
Before feeding the network with data we should preprocess our data. The first thing which comes to the mind is to detect trends and seasonality and delete it from the signal. However, in this case, there was no seasonality nor trend. Then we checked if the data doesn't contain any null values and outliers, but fortunately, it didn't.

Another thing which can be done is data normalization. In our purpose we decided to normalize our data due to this formule:
```
df = (df - df.mean()) / (df.max() - df.min())
```

As we can see, our signal is quite complex - it changes its values drastically, so we also have tried to smooth it a little bit using a rolling window technique trying different values of the window. The window size which gave us the best result was 5.

The preprocessed signal can be seen at the picture below:

![](https://imgur.com/oSG2zG7.png){width=80%}

So finally description of our data is:

![](https://imgur.com/BMTJWsV.png){width=30%}


### Statistical approach

We can find anomalies in time seris simply by searching for and extreme values or outliers. In `Dataset` chapter we build histograms of time seris values and most of time series had distrubutions close to normal. This means that we can use interquartile distance to determine the outliers.

Outliers found by the statistical approach(right):

![](https://imgur.com/XLVrwdh){width=80%}

### One-class Support Vector Machine

One-class SVM is an extension of the original SVM algorithms that learns a decision boudary that tries to achive the maximu separation between the sample the known class and the origin. Algorithm allows only small part of the dataset to lie on the other side of the decision boudary. These points are considered as outliers.

To feed OCSVM we have to transform our time series to vector space. For this, we make use of the time delay embeddings. 

Outliers detected by One-class SVM:

![](https://imgur.com/HwL3zRv){width=80%}

### Seasonal Hybrid ESD Model

Season Hybrid ESD (Extreme Studentized Deviant) is well know method for identifying anomales in times series which remains usefull and well performing. Season Hybrid ESD is built on statistical test, the *Grubbs test*, which defines a statistic for testing the hypothesis that there is a single outlier in a dataset. The ESD applies this test repeatedly, first to the most extreme outlier and then to the smaller outliers. ESD also accounts for seasonality in behavior via time series decomposition.

Visualizations anomalies found by ESD model:

![](https://imgur.com/WT4kzGW){width=50%}

We can plot confustion matrix to quantify model performance.

![](https://imgur.com/ikWarRw){width=60%}

### Isolation forests

Isolation Forest is a variation of Random Forest algorithm which creates a random trees until each values is in separate leaf. Outlier are mostly isolated in early stages of the algorithm. Based on the mean of the depth of the leaves we decide whether or value is an anomaly or not. 

Outliers detected by the Isolation forest alogrithm:

![](https://imgur.com/M0jSZNl){width=80%}

### LSTM Neural Network Approach
A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.

Before feeding the network with the data we needed to extend our dataset by assigning new attributes to each value. The new attributes are previous values of time series, so for value recorded at time t, we extended it of values from t-1, t-2 ... t-n. In this case, we got a data frame of size (initial length of data frame x n). We couldn't create all of the attributes for last n observations from each sub dataset so we decided to not include them in our training dataset. 

Because of the LSTM neural network nature, we had to reshape our data one more time to get its final dimension equal to (initial length of data frame x n x 1). It had to be done because those types of networks operate only on 3D vectors. The fraction of the training dataset can be seen in the picture.

![](https://imgur.com/kT1A5pH.png){width=30%}

The final architecture of the LSTM NN looks like:

![](https://imgur.com/oK2PkrI.png"){width=80%}

![](https://imgur.com/kT1A5pH.png){width=30%}

Its graphical ilustration can be seen below:

![](https://imgur.com/339bam0.png"){width=80%}


We repeated the learning process many times, trying different sizes of:
* n - attributes which were value from the previous timestep
* window size for smoothing the signal
* epochs 

The best-obtained result was for:
* lookback n = 5
* smoothing window size = 5
* 20 epochs

The learning process of network can be seen here:

![](https://imgur.com/59ARm9m.png"){width=80%}


Signal prediction on training dataset:

![](https://imgur.com/WbhzOD0.png"){width=80%}



Signal prediction on testing dataset:

![](https://imgur.com/Dx0f3ja.png"){width=80%}


The model isn't overtrained, as it can be seen at the pictures, signal predicted basing on the test dataset is as good as the one predicted on the training dataset.

Then, to point the anomalies we tried to find the threshold which is the absolute value of the difference between real and predicted values. To do that we split our results into two groups - anomaly data and normal data. The summarize of absolute errors for each group can be seen at the picture:

group without anomalies

![](https://imgur.com/4HMz185.png"){width=30%}


group with anomalies

![](https://imgur.com/rwVLJJg.png"){width=30%}

The max error for the group without anomalies equals to 0.151 when min error for a group with anomalies is 0.1639. It means, that we can easily split the difference between those two groups if we choose a tolerated error threshold sensibly.

The threshold value set to differentiate groups correctly is *12 * (standard deviation of the whole dataset)* what equals to 1.559.


### Summary

The experiment proves thet we can use deep neural network for anomaly prediction also. Even though it is not the classical approach for anomaly detection it works properly as well. 

If we would like to use this approach for different dataset we would probably have to change a treshold value. It could be done automatically by the function which minimizes F1 value and takes treshold as the parameter. 

