# Anomaly Detection in Time Series Datasets
*authors: Dawid Sitnik, Władysław Sinkiewicz*

# The Aim of The Project
The main goal is to detect anomalies in time series dataset. As the dataset we decied to choose data shared by YAHO called *'A Benchmark Dataset for Time Series Anomaly Detection'*, which is the real traffic data from YAHO servers. 

The data can be accessed from this url:
[YAHO dataset](https://yahooresearch.tumblr.com/post/114590420346/a-benchmark-dataset-for-time-series-anomaly?fbclid=IwAR31SaUo48kFzUCeYPFDfVGRKyqYPW3vmY0XDuci7uIYM-XrrW86QXGerrY)

# The Idea
Our idea
Nowadays there are many techniques which enables anomaly detection. We could split them into classical approach and more modern. The first group of sollutions is based on transforming dataset into its vector representation of time series (for example - values history for certain period with its eventual aggregation in smaller sub-windows in different variants) and then use one of classical alghoritms like linear regression, random forest classifier etc. to create the model. 
