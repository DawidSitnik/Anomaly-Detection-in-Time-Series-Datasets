# Anomaly Detection in Time Series Datasets
*authors: Dawid Sitnik, Władysław Sinkiewicz*

**File Description**

* LSTM_approach.ipynb - jupyter notebook in which the whole solution is made
* LSTM_approach.r - the same solution made in R with extraction of the fragment responsible for learning LSTM network.

## The Aim of The Project
The main goal is to detect anomalies in time series dataset. As the dataset we decied to choose data shared by Yahoo called *'A Benchmark Dataset for Time Series Anomaly Detection'*, which is the real traffic data from Yahoo servers. 

The data can be accessed from this url:
[Yahoo dataset](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

## Our Approach
To detect anomalies we are going to create some models whihch will be learned on the dataset which doesn't consist any anomalies. Than we will make predictions on the dataset which consists also the data with anomaly. Assuming that our models will work properly, predictions which values are much different than real values will be treated as anomalies.

In our project we would like to compare classical approaches of modeling with more modern one. The first group of sollutions will be based on transforming dataset into its vector representation of time series (for example - values history for certain period with its eventual aggregation in smaller sub-windows in different variants) and then use one of classical alghoritms like linear regression, random forest classifier etc. to create the model. In the second approach we will use LSTM neural network which will work only on historical values from the time series.

Eventually, we will also try to find anomalies with ARIMA and ESD models. This models are mostly used for forecasting, but their application can be extended to anomaly detection.

## Dataset
This dataset is provided as part of the Yahoo! Webscope program, to be used for approved non-commercial research purposes by recipients who have signed a Data Sharing Agreement with Yahoo! Dataset contains real and synthetic time-series with labeled anomalies. Timestamps are replaced by integers with the increment of 1, where each data-point represents 1 hour worth of data. The anomalies are marked by humans and therefore may not be consistent.

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

The dataset consists of 10 independent .csv files which combined signal is presented at the picture below: 
<p align="center">
  <img src = "https://imgur.com/FGLEwGT.png"/>
</p>

Example of one of the signals with highlighted anomalies:
<p align="center">
  <img src="https://imgur.com/y1cyx0a.png">
</p>

Common exploratory data analysis tool in time-series and non-time series data are histograms. We will also look at the differenced data because we want to use our timestamp (time) axis.

The histogram of untransformed data
<div class="row">
  <div class="column">
    <img src="https://imgur.com/M2phBfi.png">
  </div>
  <div class="column">
    <img src="https://imgur.com/ovzO405.png">
  </div>
</div>

The histogram of untransformed data (top) shows normal distributions. This is to be expected given that the underlying sample (13th in our case) has no trend. If series had trend, we would difference the data to remove it, and this would transform the data to a more normal shaped distribution (bottom).

Next, we would like to know if time-series is *stationary*. We do this because many traditional statistical time series models rely on time series being stationary. In general, time series is *stationary* when it has fairly stable statistical properties over time, particularly mean and variance. The Augmented Dickey-Fuller (ADF) test is the most commonly used metric to access a time series for stationary problems. That test focuses on whether the mean of a series is changing, a variance isn’t tested here. 

ADF test result of one of the the Yahoo time series.

```
    Augmented Dickey-Fuller Test

data:  series$value
Dickey-Fuller = -10.364, Lag order = 11, p-value = 0.01
alternative hypothesis: stationary
```

Depending on the resulst of the test, null hypothesis can be rejected for a specific significance level - *p*-value. Conventionally, if *p*-value is less 0.05, the time series is likely stationany, whereas a *p* > 0.05 provides no such evidence.

Common exploartory times series methods is identifying where the series has a self-correlation, which generalizes to autocorrelation. Self-corelation of a time series is the ideat that a value in a time series at one give point in time has a correlation to the value at another point in time. Autocorrelation on the other hand asks more general question of wheher there is a correlation between two points in a specific time series witha a specific fixed distance between them.

We can apply the Box-Pierce test to the data to know wheher or not the data is autocorrelated.

```
    Box-Pierce test

data:  series$value
X-squared = 1128.3, df = 1, p-value < 2.2e-16
```

Same as with ADF test we can how likely times series is autocorrelated depending on the *p*-values.

We can graph the autocorrelation function to dig further into the data.

<p align="center">
  <img src="https://imgur.com/kyFbXmR.png">
</p>

The partial autocorrelation function is another tool for revealing the interpollations in at time seris. However, its interpolation is much less intutive than that of the autocorrelation function. One of the definitions of the partial autocorrelation functions goes as follow:

> The partial correlation between two random variables, X and Y, is the correlation that remains after accounting for the correlation shown by X and Y with all other variables. In the case of time series, the partial autocorrelation at lag k is the correlation between all data points that are exactly k steps apart, after accounting for their correlation with the data between those k steps.

The particular value of partial autocorrelation is that it helps to identify the number of *autoregressions* coefficients, which are used in ARIMA model.

We can visualize partial autocorrelation function at each lag.

<p align="center">
  <img src = "https://imgur.com/gWQenZu.png"/>
</p>

### Data Preprocessing
Before feeding the network with data we should preprocess our data. The first thing which comes to the mind is to detect trends and seasonality and delete it from the signal. However, in this case there is no seasonality nor trend, as we can see at the picture. 

Another thing which can be done is data normalization. In our purpose we decided to normalize our data due to this formule:
```
df = (df - df.mean()) / (df.max() - df.min())
```
So finally description of our data is:
<p align="center">
  <img src = "https://imgur.com/5d2GGWd.png"/>
</p>

We also have checked if the data doesn't contain any null falues and outlayers, but fortunatelly it didn't.

As we can see, our signal is quite complex - it changes its values drastically, so we also have tried to smooth it a little bit using rooling window technique trying different values of window. The signal smoothed with window set to 5, can be seen at the picture bellow:

<p align="center">
  <img src = "https://imgur.com/G9gw5w4.png"/>
</p>

### Liner Regression

The liner regression model is one the most common method for identifyin and quantifying the relashionship between a dependant variables and a single independent variables. This model has a wide range of applications, from a casual inference to predictive analysis.

*Liner Regression anomaly detection model will be implemented in the next steps of the projct*

### Season Hybrid ESD Model

Season Hybrid ESD (Extreme Studentized Deviant) is well know method for identifying anomales in times series which remains usefull and well performing. Season Hybrid ESD is built on statistical test, the *Grubbs test*, which defines a statistic for testing the hypothesis that there is a single outlier in a dataset. The ESD applies this test repeatedly, first to the most extreme outlier and the to the smaller outliers. ESD also accounts for seasonality in behavior vie time series decomposition.

Visualizations of the Yahoo time series with actual anomalies(top) and anomalies found by ESD model.

<div class="row">
  <div class="column">
    <img src="https://imgur.com/y1cyx0a.png">
  </div>
  <div class="column">
    <img src = "https://imgur.com/SIHFsXV.png"/>
  </div>
</div>

We can plot confustion matrix to quantify model performance.

<p align="center">
  <img src="https://imgur.com/5cZgO07.png">
</p>

### ARIMA model

*ARIMA anomaly detection model will be implemented in the next steps of the project*
Implementation of ARIMA forecasting model can be found in [yahoo_notebook.Rmd](yahoo_notebook.Rmd)

### Random Forests

*Random Forest model will be implemented in the next steps of the project*

### Gradient Boosted Machine model

*Gradient Boost Machine anomaly model will be implemented in the next steps of the project*

### LSTM Neural Network Approach
A powerful type of neural network designed to handle sequence dependence is called recurrent neural networks. The Long Short-Term Memory network or LSTM network is a type of recurrent neural network used in deep learning because very large architectures can be successfully trained.

Before feeding the network with the data we needed to extend our dataset by assigning new attributes to each value. The new attributes are previous values of time series, so for value recorded at time t we extended it of values from t-1, t-2 ... t-n. In this case we got dataframe of size (initial length of dataframe x n). We couldn't create all of the attributes for last n observations from each subdataset so we decided to not include them in our training dataset. 

Because of the LSTM neural network nature, we had to reshape our data one more time to get its final dimension equal to (initial length of dataframe x n x 1). It had to be done, because those types of networks opperates only on 3D vectors. The fraction of the training dataset can be seen at the picture.

<p align="center">
  <img src = "https://imgur.com/kT1A5pH.png"/>
</p>

The final architecture of the LSTM NN looks like:
<p align="center">
  <img src = "https://imgur.com/oK2PkrI.png"/>
</p>

Its graphical ilustration can be seen below:
<p align="center">
  <img src = "https://imgur.com/339bam0.png"/>
</p>

We repeated learning process many times, trying different sizes of:
* n - attributes which were value from previous timestep
* window size for smoothing the signal
* epochs 

The best obtained result was for:
* n = 5
* window size = 1, so there was no need for simplyfing the signal
* 100 epochs

The learning process of network can be seen here:
<p align="center">
  <img src = "https://imgur.com/EhXDPQB.png"/>
</p>

Than we tested our model on the data that wasn't used for learning enriched of data with anomalies. The final result:
<p align="center">
  <img src = "https://imgur.com/KllWyqt.png"/>
</p>

Than, to point the anomalies we tried to find the treshold which is the the absolute value of difference between real and predicted values. To do that we splited our results into two groups - anomaly data and normal data. The summarize of absolute errors for each group can be seen at the picture:

group without anomalies
<p align="center">
  <img src = "https://imgur.com/F6gOrQT.png"/>
</p>

group with anomalies
<p align="center">
  <img src = "https://imgur.com/znyw1Up.png"/>
</p>

If we take a look at given standatd deviation, means and max values the result looks quite promising and seems like we could differentiate those two groups without any problem. 

However, there is one drawback. If we take a closer look at destribution of the absolute errors we can realize that those differences arises from the last quantiles of the absolute errors.
<p align="center">
  <img src = "https://imgur.com/rr2esUq.png"/>
</p>

In this case there is no possibility to detect all of the anomalies, because in the most cases their values are simillar to the predicted ones. We could expect that situation, because as it was written in the dataset description: *The anomalies are marked by humans and therefore may not be consistent.* 

The best possible result we could get was positively finding 30 anomalies out of 160, at the same time not classyfing any normal value as anomaly. This result was obtained for treshold equaled to 30. 

To detect more anomalies we could try to escalate the treshold value, but we would have to pay the cost of classyfing normal data as anomalies. 

To obtain better results we could try to define anomalies on our own, looking at its values and gradients. 
