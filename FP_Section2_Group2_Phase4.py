# Databricks notebook source
# MAGIC %md
# MAGIC # Predictive delays in commercial flights
# MAGIC ## Phase 4
# MAGIC
# MAGIC **Team**: Section 2, Team 2
# MAGIC
# MAGIC **Team Members**: Camille Church, Anoop Nair, Dylan Jin, Ken Trinh, Shalini Chawla, Hector Rincon
# MAGIC
# MAGIC **Phase 4 Leaders**: Dylan Jin, Shalini Chawla

# COMMAND ----------

import requests
from IPython.display import display, Image

#![Updated credit assignment on Monday.com](http://13.66.213.192/w261/phasePlan.png)
# URL of the image
image_url = "http://13.66.213.192/w261/team-members.png"

# Use requests to get the image
response = requests.get(image_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Display the image in the notebook
    img = Image(response.content)
    display(img)
else:
     print(f"Image is not accessible. Status code: {response.status_code}")

# COMMAND ----------

# DBTITLE 1,Updated credit assignment on Monday.com
import requests
from IPython.display import display, Image

#![Updated credit assignment on Monday.com](http://13.66.213.192/w261/phasePlan.png)
# URL of the image
image_url = "http://13.66.213.192/w261/monday_week4.png"

# Use requests to get the image
response = requests.get(image_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Display the image in the notebook
    img = Image(response.content)
    display(img)
else:
     print(f"Image is not accessible. Status code: {response.status_code}")

# COMMAND ----------

# MAGIC %run "./configurations/mounting"

# COMMAND ----------

# list all available joined flights data on ADLS 
# display(dbutils.fs.ls(f'{team_blob_url}/OTPW_Flights/'))

# COMMAND ----------

#Install the holidays library if not already installed
!pip install holidays hyperopt

# COMMAND ----------

import pyspark.sql.types as T
import pyspark.sql.functions as F
from pyspark.sql import Window

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import holidays
from datetime import datetime, timedelta, date

sns.set()

# COMMAND ----------

# Load Packages for Modeling

from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder, PCA
from pyspark.ml.stat import Correlation
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml import Pipeline

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

import mlflow
import mlflow.spark

# run grisearch (set this to fasle after you get the param)
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import itertools

from functools import reduce
from pyspark.sql import DataFrame

# set flags
random_split = False
run_corr_matrix = False
run_PCA_analysis = False
perform_resampling = True
resample_by_year = True
write_to_blob = True
grisearch = False
ts_grisearch = False
run_model_flag = False
new_data_flag = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data

# COMMAND ----------

# dataset name
## NOTE replace this with new dataset
dataset_name = "OTPW_60M" #"OTPW_3M"

# COMMAND ----------

# ingest flights dataset
# NOTEO edit this to point to new data
df_flights = spark.read.format('parquet')\
                        .option("inferSchema", True)\
                        .load(f"{team_blob_url}/OTPW_Flights/{dataset_name}").cache()


# COMMAND ----------

# Group data by year and count records for each year
year_counts = df_flights.groupBy('YEAR').count()

# Show the count of records for each year
year_counts.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Project Abstract
# MAGIC ## Introduction
# MAGIC Airlines are a critical part of the transportation model today. Unforeseen and extended flight delays are not just a cause of financial loss for airline carriers, they are also a major inconvenience for the passengers. According to Berkeley News, over half the costs associated with a delayed flight is the responsibility of the passengers. It is also estimated, that fear of flight delays as a 32.9 billion economic impact on vacation hotspots. 
# MAGIC
# MAGIC For this study, the airline passenger is our stakeholder. Being able to get an accurate advance notice of an impending flight delay will allow the passengers to plan their trip better, reducing the financial burden they carry and will elevate the related “flight fears” and increase willingness to travel for vacationing.
# MAGIC
# MAGIC There has been a lot of previous research to use machine learning to create predictive models for flight delay. Some of these researches have focused on limited data sets or specific airports. In this research study, our goal is to implement machine learning at scale using  US domestic flights data collected by the Department of Transporation, that contains more than 30M flights records for years 2015 to 2019.
# MAGIC
# MAGIC In addition to our source data with flight information, we are using the weather data from NOAA’s Quality Controlled Local Climatological Data, which gathers data from weather stations at major airports. We also use supplemental data from US Holiday Calendar to account for holiday rush and its impact of flight delays.
# MAGIC
# MAGIC The goal of this analysis is to predict the binomial indicator for departure delay, 15 Minutes or More (1=Yes the flight is delayed by at least 15 minutes)
# MAGIC
# MAGIC ## Project Phases
# MAGIC The project consists of the below phases:
# MAGIC
# MAGIC 1. Data Cleaning: Dropped columns with >25% values and samples with missing essential information
# MAGIC
# MAGIC 2. a) Feature Selection: Selected flight and weather related features based on EDA and correlation analysis
# MAGIC
# MAGIC    b) Feature Engineering: (Additional features)
# MAGIC
# MAGIC     - Holiday Rush indicator by joining US Holidays information with the core dataset
# MAGIC     - Departure timeslot by grouping the flight departure time into Morning/Evening/Afternoon/Night
# MAGIC     - Sesonal Information by grouping deprature months into Seasons
# MAGIC     - Weather reading latecy flag by measuring the time lag between scheduled departure time and time of weather report
# MAGIC     - Recency/Frequency/Monetary using RFM Analysis
# MAGIC     - Airport Rank using Graph Analysis
# MAGIC
# MAGIC    c) Feature Transformation:
# MAGIC
# MAGIC     - PCA for dimentionality Reduction
# MAGIC     - Standard scaling of all numerical features
# MAGIC     - Binning of airport codes by flight volume into 4 buckets
# MAGIC     - One hot encoding of categorical features
# MAGIC     - Train/Test split: Split the data to use data from 2015-2018 for Train/validation and 2019 data for test
# MAGIC
# MAGIC 3. a) Train-Test Split: The 60 month data set was split using data from year 2015-2018 for traning and year 2019 data for testing
# MAGIC
# MAGIC    b) Model Training: Trained 5 different model categories Logistic Regression, Decision Tree, Random Forest, Gradient Boosted Tree and Neural Network
# MAGIC
# MAGIC 4. Hyperparameter Tuning: Used grid search for hyperparameter tuning using cross validation on data split by time blocks

# COMMAND ----------

# DBTITLE 1,Phase Diagram
import requests
from IPython.display import display, Image

# URL of the image
image_url = "http://13.66.213.192/w261/flow_diagram.png"

# Use requests to get the image
response = requests.get(image_url)

# Check if the request was successful (status code 200)
if response.status_code == 200:
    # Display the image in the notebook
    img = Image(response.content)
    display(img)
else:
    print(f"Image is not accessible. Status code: {response.status_code}")


# COMMAND ----------

# how many delays per total set
print(f"Number of Flights Delayed in 60M: {df_flights.filter('DEP_DEL15 = 1').count()}")
print(f"Total Number of Flights in 60M: {df_flights.count()}")
print(f"Percentage of Flights Delayed in 60M: {round(df_flights.filter('DEP_DEL15 = 1').count()/df_flights.count()*100,2)}%")

majority_class_proba = 1-round(df_flights.filter('DEP_DEL15 = 1').count()/df_flights.count())

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Engineering
# MAGIC
# MAGIC The occurrence of departure delays in aviation can be attributed to diverse factors, potentially cascading effects on subsequent flights. Weather conditions, holidays, time of day, and even minor operational changes can culminate in delays, as they impact cabin cleaning, baggage loading, and more. To comprehensively address these intricacies, we devised an array of features that encapsulate various dimensions of departure delay.
# MAGIC
# MAGIC The foundation of our analysis was the airline delay data sourced from the Bureau of Transportation Statistics. This dataset encompassed historical flight records, airport information, and weather measurements. Transforming this raw data into actionable insights necessitated rigorous data cleaning, feature extraction, and data enrichment processes.
# MAGIC
# MAGIC <b>Key Feature Engineering Highlights</b>
# MAGIC </br>
# MAGIC
# MAGIC <b>Airport and Carrier Groups</b></br>
# MAGIC In response to the left-skewed distribution of flight volumes observed during exploratory data analysis (EDA), we partitioned airports and carriers into four categories. These partitions were based on quantiles of flight volume distribution. This grouping strategy ensured that each category contained approximately 25% of the data, translating to around 33-34 airports and 3-4 carriers per group.
# MAGIC
# MAGIC <b>Weather Conditions</b></br>
# MAGIC Beyond using weather readings as numerical features, we engineered a feature to gauge the time interval preceding the flight's departure when the corresponding weather reading was recorded. This "time before departure" metric enhances the correlation of weather data with departure outcomes.
# MAGIC
# MAGIC <b>RFM Analysis</b></br>
# MAGIC Exploiting historical trends, we conducted Recency, Frequency, and Monetary (RFM) analysis to derive additional predictive features. RFM scores were calculated to quantify the recency of delays, frequency of delayed flights, and average delay time for a flight. This facilitated the segmentation of flights based on their historical behavior.
# MAGIC
# MAGIC <b>Graph Analysis</b></br>
# MAGIC Leveraging the interconnections within the flight network, we conducted graph analysis to compute the Airport Rank. This rank quantifies the significance of airports within the transportation network. Centrality measures like degree, betweenness, and closeness were harnessed to capture the connectivity and influence of each airport in the network.
# MAGIC
# MAGIC <b>Time Factors</b></br>
# MAGIC Supplementary information, such as the US Holiday calendar, was integrated with existing features to derive time-related features associated with flight departures. Columns indicating holidays and time blocks were added through joins between flight dates and holidays. Time blocks categorized departure times into distinct segments of the day, and the "Holiday Rush" period was defined as a two-day window encompassing each holiday. This period was flagged using a "HOLIDAY_RUSH" indicator to capture the potential impact of holiday travel on delays.
# MAGIC
# MAGIC <b>Seasonality and Day of the Month</b></br>
# MAGIC Departure months were grouped into "Summer", "Winter", "Fall", "Spring" to encapsulate seasonality effects, while "Day of the Month" was also considered as a potential feature for modeling.
# MAGIC
# MAGIC These tailored features collectively address various facets of departure delays, enhancing the predictive power of our models. Through systematic experimentation, we demonstrated the substantial value contributed by each feature family in refining our predictions and understanding the underlying dynamics of flight delays.
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Holidays Flag

# COMMAND ----------

#Create a list of years from 2015 to the current year
current_year = date.today().year
years = list(range(2015, current_year + 1))

#Initialize the UnitedStates class from the holidays library
us_holidays = holidays.UnitedStates(years=years)

# Extract the list of holiday dates as strings in 'yyyy-mm-dd' format
holiday_dates = [str(holiday_date) for holiday_date in us_holidays.keys()]

def is_holiday(date_str):
    return 1 if date_str in holiday_dates else 0

# Register the UDF with Spark
is_holiday_udf = F.udf(is_holiday, T.IntegerType())

# Add new holiday column to dataframe
df_flights = df_flights.withColumn('IS_HOLIDAY', is_holiday_udf(F.date_format('FL_DATE', 'yyyy-MM-dd')))

# COMMAND ----------

# MAGIC %md
# MAGIC We now add a `HOLIDAY_RUSH` indicator column to a window of 2 days on either side of identified holidays, plus the actual holiday date

# COMMAND ----------

# Add the 'HOLIDAY_RUSH' column to the DataFrame
df_flights = df_flights.withColumn('HOLIDAY_RUSH', F.when(
    (is_holiday_udf(F.date_format(F.date_sub(df_flights['FL_DATE'], 1), 'yyyy-MM-dd')) == 1) |
    (is_holiday_udf(F.date_format(F.date_sub(df_flights['FL_DATE'], 2), 'yyyy-MM-dd')) == 1) |
    (is_holiday_udf(F.date_format(F.date_add(df_flights['FL_DATE'], 1), 'yyyy-MM-dd')) == 1) |
    (is_holiday_udf(F.date_format(F.date_add(df_flights['FL_DATE'], 2), 'yyyy-MM-dd')) == 1) |
    (is_holiday_udf(F.date_format(df_flights['FL_DATE'], 'yyyy-MM-dd')) == 1),
    1
).otherwise(0))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Departure Timeslot Flag

# COMMAND ----------

## Most number of flights are during 6am - 7am timeslot
## Most delay is during 5pm - 6pm timeslot
## As day progresses, delay increases

## We can create another timeslot variable with DEP_TIME_BLK grouped as below:
## morning : 0600-0659, 0700-0759, 0800-0859, 0900-0959
## mid-morning: 0900-0959, 1000-1059, 1100-1159
## afternoon : 1200-1259, 1300-1359, 1400-1459
## evening : 1500-1559, 1600-1659, 1700-1759, 1800-1859
## night: 1900-1959, 2000-2059, 2100-2159, 2200-2259, 2300-2359, 0001-0559,

# Define a UDF to map 'DEPT_TIME_BLK' to 'TIME_SLOT'
def map_time_slot(dept_time_blk):
    if dept_time_blk in ['0600-0659', '0700-0759', '0800-0859', '0900-0959']:
        return 'morning'
    elif dept_time_blk in ['1000-1059', '1100-1159']:
        return 'mid-morning'
    elif dept_time_blk in ['1200-1259', '1300-1359', '1400-1459']:
        return 'afternoon'
    elif dept_time_blk in ['1500-1559', '1600-1659', '1700-1759', '1800-1859']:
        return 'evening'
    elif dept_time_blk in ['1900-1959', '2000-2059', '2100-2159', '2200-2259', '2300-2359', '0001-0559']:
        return 'night'
    else:
        return 'unknown'

# Convert the UDF to a Spark UDF
map_time_slot_udf = F.udf(map_time_slot)

# Add the new 'TIME_SLOT' column based on 'DEPT_TIME_BLK'
df_flights = df_flights.withColumn('TIME_SLOT', map_time_slot_udf('DEP_TIME_BLK'))


# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Seasonal Flag

# COMMAND ----------

# Define a UDF to map the month to the corresponding season
def get_season(month):
    if month in [12, 1, 2]:
        return 'winter'
    elif month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'fall'
    else:
        return 'unknown'

# Register the UDF with Spark
get_season_udf = F.udf(get_season, F.StringType())

# Add the 'season' column to the DataFrame
df_flights = df_flights.withColumn('SEASON', get_season_udf(F.col('MONTH')))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply Weather Measurement Flag
# MAGIC
# MAGIC Weather taken closer to depart would affect the delay. Here we're taking the differences between weather datetime column 'DATE' and the three scheduled depart columns.

# COMMAND ----------

# determine when weather condition was measured
df_flights = df_flights.withColumn('delta_depart_2h', 
                                    F.abs(F.col('DATE').cast(T.TimestampType())-F.col('two_hours_prior_depart_UTC').cast(T.TimestampType())))\
                            .withColumn('delta_depart_4h', 
                                    F.abs(F.col('DATE').cast(T.TimestampType())-F.col('four_hours_prior_depart_UTC').cast(T.TimestampType())))\
                            .withColumn('delta_depart', 
                                    F.abs(F.col('DATE').cast(T.TimestampType())-F.col('sched_depart_date_time_UTC').cast(T.TimestampType())))\
                            .withColumn('weather_condition_measured_time_from_depart', 
                                    F.least(F.col('delta_depart_2h'), F.col('delta_depart_4h'), F.col('delta_depart')))\
                            .withColumn('closest_depart_time_weather_measurement', 
                                    F.when(F.col('weather_condition_measured_time_from_depart') == F.col('delta_depart_2h'), '2hr_from_depart')\
                                    .when(F.col('weather_condition_measured_time_from_depart') == F.col('delta_depart_4h'), '4hr_from_depart')\
                                    .otherwise('on_depart'))\
                            .withColumn("weather_condition_measured_time_from_depart", 
                                    F.col('weather_condition_measured_time_from_depart').cast(T.IntegerType())).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## RFM (Recency, Frequency, Monetary) Analysis
# MAGIC Similar to the usage of RFM analysis to identify the best and most at risk customers of a business, we analyze the flight delay data to calculate recency, Frequency, Monetary value that contribute to the probablity of a flight being delayed. For flights(Carrier+FL Number), we calculate...
# MAGIC
# MAGIC - **Recency**: When was the last time this flight got delayed.  
# MAGIC   - More recent delay -> Higher probability of future delays
# MAGIC - **Frequency**: How frequently is a flight delayed. We calculate this using the no of delayed flights/the total number of flights.  
# MAGIC   - Higher ratio of delays -> Higher probability of future delays
# MAGIC - **Monetary**: The average delay time for the flight. We calculate this only for the delayed flights as Total delay time/number of flight.  
# MAGIC   - Higher average delay -> Higher probability of future delays

# COMMAND ----------

#Use the maximum flight date as the reference for calculating recency
max_flight_date = df_flights.select(F.max('FL_DATE')).collect()[0]['max(FL_DATE)'].strftime("%Y-%m-%d")
df_flights = df_flights.withColumn("RecencyDays", F.expr(f"datediff('{max_flight_date}', DATE)"))

# COMMAND ----------

#RFM Analysis
#Recency is only recent occurence of delayed flight
#Frequency should be calculated relative to total no of flights, not just occurences of delay
#Monetary should be average delay not just total delay since it will be sensitive to the frequency of flight
rfm_table = df_flights.filter("DEP_DEL15 == '1'") \
                        .groupBy("OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM") \
                        .agg(F.min("RecencyDays").alias("Recency"), \
                              F.avg("DEP_DELAY").alias("Monetary"))
                        
frequency_table = df_flights.groupBy("OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM") \
                        .agg(F.sum('DEP_DEL15').alias('no_delays'),
                            F.count('DEP_DEL15').alias('no_flights'))\
                            .withColumn('Frequency',F.col('no_delays')/F.col('no_flights'))

rfm_table = rfm_table.join(frequency_table, ["OP_UNIQUE_CARRIER", "OP_CARRIER_FL_NUM"]).drop("no_delays").drop("no_flights")
# display(rfm_table)

# COMMAND ----------

# binning the RFM tables into quantiles
r_quartile = rfm_table.approxQuantile("Recency", [0.25, 0.5, 0.75], 0)
f_quartile = rfm_table.approxQuantile("Frequency", [0.25, 0.5, 0.75], 0)
m_quartile = rfm_table.approxQuantile("Monetary", [0.25, 0.5, 0.75], 0)

rfm_table = rfm_table.withColumn("R_Quartile", \
                                 F.when(F.col("Recency") >= r_quartile[2] , 'A')\
                                 .when(F.col("Recency") >= r_quartile[1] , 'B')\
                                 .when(F.col("Recency") >= r_quartile[0] , 'C')\
                                 .otherwise('D'))

rfm_table = rfm_table.withColumn("F_Quartile", \
                                 F.when(F.col("Frequency") > f_quartile[2] , 'D')\
                                 .when(F.col("Frequency") > f_quartile[1] , 'C')\
                                 .when(F.col("Frequency") > f_quartile[0] , 'B')\
                                 .otherwise("A"))

rfm_table = rfm_table.withColumn("M_Quartile", \
                                 F.when(F.col("Monetary") >= m_quartile[2] , 'D')\
                                 .when(F.col("Monetary") >= m_quartile[1] , 'C')\
                                 .when(F.col("Monetary") >= m_quartile[0] , 'B')\
                                 .otherwise('A'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Airport Rank

# COMMAND ----------

from graphframes import GraphFrame
import networkx as nx

# Create a new DataFrame containing unique airport codes
airport_codes_df = df_flights.select('origin_iata_code').union(df_flights.select('dest_iata_code')).distinct()

# Create vertices DataFrame
vertices = airport_codes_df.withColumnRenamed('origin_iata_code', 'id')

# Create edges DataFrame
edges = df_flights.select('origin_iata_code', 'dest_iata_code', 'Distance') \
                  .withColumnRenamed('origin_iata_code', 'src') \
                  .withColumnRenamed('dest_iata_code', 'dst')

# Create a GraphFrame
#g = GraphFrame(vertices, edges)

# Create Graph from the GraphFrame
nx_graph = nx.from_pandas_edgelist(edges.toPandas(), 'src', 'dst', edge_attr='Distance', create_using=nx.DiGraph())

# Calculate degree centrality
degree_centrality = nx.degree_centrality(nx_graph)

# Calculate betweenness centrality
betweenness_centrality = nx.betweenness_centrality(nx_graph)

# Calculate closeness centrality
closeness_centrality = nx.closeness_centrality(nx_graph)

# Convert the results to DataFrames and alias the columns
degree_centrality_df = spark.createDataFrame([(k, v) for k, v in degree_centrality.items()], ['origin_iata_code', 'degree_centrality'])
betweenness_centrality_df = spark.createDataFrame([(k, v) for k, v in betweenness_centrality.items()], ['origin_iata_code', 'betweenness_centrality'])
closeness_centrality_df = spark.createDataFrame([(k, v) for k, v in closeness_centrality.items()], ['origin_iata_code', 'closeness_centrality'])

# Merge the graph-based features with the original DataFrame
df_flights = df_flights.join(degree_centrality_df, on='origin_iata_code', how='left')
df_flights = df_flights.join(betweenness_centrality_df, on='origin_iata_code', how='left')
df_flights = df_flights.join(closeness_centrality_df, on='origin_iata_code', how='left')

# COMMAND ----------

# Calculate the "airport rank" column based on the graph-based features
df_flights = df_flights.withColumn('airport_rank', 
                                   F.col('degree_centrality') +
                                   F.col('betweenness_centrality') +
                                   F.col('closeness_centrality'))

# COMMAND ----------

# Select a subset of airports to visualize
subset_airports = df_flights.select('origin_iata_code').distinct().limit(10).toPandas()['origin_iata_code'].tolist()

subgraph = nx_graph.subgraph(subset_airports)

pos = nx.spring_layout(subgraph, seed=42)

nx.draw(subgraph, pos, with_labels=True, node_size=1000, node_color='skyblue', font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(subgraph, 'Distance')
nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=8)

plt.title("Airport Graph")
plt.axis('off')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Numeric Features

# COMMAND ----------

## Weather features
numerics_features = ["HourlyPrecipitation", "HourlyDryBulbTemperature", 
                "HourlyPressureChange", "HourlyRelativeHumidity",
                "HourlySeaLevelPressure", "HourlyStationPressure",
                "HourlyVisibility", "HourlyWindGustSpeed","HourlyWindSpeed", "ELEVATION"]

## Cast Column to Float
for c in numerics_features:
    df_flights = df_flights.withColumn(c, F.col(c).cast("Float"))

# filter out Null values from our dataset (12M)
join_keys = ['FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM', 'OP_CARRIER_FL_NUM', 'sched_depart_date_time_UTC', 'DEP_DEL15']
df_flights_numeric_non_null = df_flights.select(join_keys + numerics_features).drop(*['HourlyPressureChange', 'HourlyWindGustSpeed']).na.drop().cache()
# print(f"Total Number of Flights (Non Null): {df_flights_numeric_non_null.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Merge Data

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge weather with non-nulls

# COMMAND ----------

# join data with main flights dataframe
# Merge data to obtain spacetime
df_flights_numeric_non_null = df_flights_numeric_non_null.alias('a').join(df_flights.alias('b').drop(*numerics_features+['DEP_DEL15']),
                                                                                (df_flights_numeric_non_null.FL_DATE == df_flights.FL_DATE) & 
                                                                                (df_flights_numeric_non_null.OP_UNIQUE_CARRIER == df_flights.OP_UNIQUE_CARRIER) &
                                                                                (df_flights_numeric_non_null.TAIL_NUM == df_flights.TAIL_NUM) &
                                                                                (df_flights_numeric_non_null.OP_CARRIER_FL_NUM == df_flights.OP_CARRIER_FL_NUM) &
                                                                                (df_flights_numeric_non_null.sched_depart_date_time_UTC == df_flights.sched_depart_date_time_UTC), how='left')\
                                                                    .drop(F.col("b.FL_DATE"), 
                                                                          F.col("b.OP_UNIQUE_CARRIER"), 
                                                                          F.col("b.TAIL_NUM"),
                                                                          F.col("b.OP_CARRIER_FL_NUM"), 
                                                                          F.col('b.sched_depart_date_time_UTC')).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge Non-Nulls with RMF Table

# COMMAND ----------

df_flights_numeric_non_null = df_flights_numeric_non_null.alias('a').join(rfm_table.alias('b'),
                                                                         (df_flights_numeric_non_null.OP_UNIQUE_CARRIER == rfm_table.OP_UNIQUE_CARRIER) &
                                                                         (df_flights_numeric_non_null.OP_CARRIER_FL_NUM == rfm_table.OP_CARRIER_FL_NUM),
                                                                          how='inner')\
                                                                    .drop(F.col("b.OP_UNIQUE_CARRIER"), 
                                                                          F.col("b.OP_CARRIER_FL_NUM")).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-numeric bucketized flights

# COMMAND ----------

# Filtering out to only large departure
df_flights_numeric_non_null = df_flights_numeric_non_null.filter("origin_type = 'large_airport'").cache()

# COMMAND ----------

# Parititon the Airport into quantiles
number_of_flights = df_flights_numeric_non_null.groupby('origin_iata_code').count()
# number_of_flights.summary().show()

# approxmiate the quantiles
iata_quartile = number_of_flights.approxQuantile("count", [0.00, 0.25, 0.5, 0.75, 1.00], 0)

# COMMAND ----------

airports_number_of_flights = number_of_flights.withColumn('airport_groups',
                                                          F.when(F.col('count') < iata_quartile[0], 'A')\
                                                            .when((F.col('count') >= iata_quartile[0]) & ((F.col('count') <= iata_quartile[1])) , 'B')\
                                                            .when((F.col('count') > iata_quartile[1]) & ((F.col('count') <= iata_quartile[2])) , 'C')\
                                                            .when((F.col('count') > iata_quartile[2]) & ((F.col('count') <= iata_quartile[3])) , 'D')\
                                                            .otherwise('E')).cache()

# COMMAND ----------

# Parititon the Carriers into quantiles
number_of_flights = df_flights_numeric_non_null.groupby('OP_UNIQUE_CARRIER').count()
# number_of_flights.summary().show()

# approxmiate the quantiles
carrier_quartile = number_of_flights.approxQuantile("count", [0.00, 0.25, 0.5, 0.75, 1.00], 0)

# COMMAND ----------

carriers_number_of_flights = number_of_flights.withColumn('carrier_groups', 
                                                          F.when(F.col('count') < carrier_quartile[0], "A")\
                                                            .when((F.col('count') >= carrier_quartile[0]) & ((F.col('count') < carrier_quartile[1])) , "B")\
                                                            .when((F.col('count') >= carrier_quartile[1]) & ((F.col('count') < carrier_quartile[2])) ,"C")\
                                                            .when((F.col('count') >= carrier_quartile[2]) & ((F.col('count') < carrier_quartile[3])) , "D")\
                                                            .otherwise('E')).cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Consolidation

# COMMAND ----------

df_flights_numeric_non_null = df_flights_numeric_non_null.join(airports_number_of_flights.drop(F.col('count')), ['origin_iata_code'], how='left')\
                                                            .join(carriers_number_of_flights.drop(F.col('count')), ['OP_UNIQUE_CARRIER'], how='left').cache()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dimensionality Reduction and One-Hot Encoding

# COMMAND ----------

def ApplyOHE(df, col):
    """
        helper function to apply OHE
    """  
    # StringIndexer Initialization
    indexer = StringIndexer(inputCol=col, outputCol=col+"_indexed")
    indexerModel = indexer.fit(df)

    # Transform the DataFrame using the fitted StringIndexer model
    indexed_df = indexerModel.transform(df)

    # apply OneHot Encoding to the indexed column
    encoder = OneHotEncoder(inputCol=col+"_indexed", outputCol=col+"_onehot")
    odf = encoder.fit(indexed_df).transform(indexed_df)
    return odf


df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'airport_groups')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'carrier_groups')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'TIME_SLOT')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'SEASON')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'R_Quartile')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'F_Quartile')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'M_Quartile')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'closest_depart_time_weather_measurement')
df_flights_numeric_non_null = ApplyOHE(df_flights_numeric_non_null, 'DAY_OF_MONTH')

# COMMAND ----------

# MAGIC %md 
# MAGIC ## PCA Explained Variance Search 
# MAGIC Start PCA Analysis with a plot of the correlation matrix to identify the most correlated features. PCA are extremely useful to dimensionally reduced highly correlated features to avoid autocorrelation.

# COMMAND ----------

# plot correlation matrix
# Serialize the weather features
numeric_features = ["HourlyPrecipitation", "HourlyDryBulbTemperature", 
                    "HourlyRelativeHumidity", "HourlySeaLevelPressure", 
                    "HourlyStationPressure", "HourlyVisibility",
                    "HourlyWindSpeed", "IS_HOLIDAY", "HOLIDAY_RUSH", 
                    "Recency", "Monetary", "Frequency",
                    "degree_centrality", "betweenness_centrality",
                    "closeness_centrality", "airport_rank", 'ELEVATION', "DEP_DEL15"]

# COMMAND ----------

if run_corr_matrix:
    # Generate the correlation matrix from the selected numeric features
    assembler = VectorAssembler(inputCols=numeric_features, 
                                outputCol="corr_features")
    df_flights_numeric_non_null_vector = assembler.transform(df_flights_numeric_non_null).select("corr_features")
    matrix = Correlation.corr(df_flights_numeric_non_null_vector, "corr_features")

    tMatrix = matrix.collect()[0][0]
    corrmatrix = tMatrix.toArray().tolist()

    def plot_corr_matrix(correlations,attr,fig_no):
        fig=plt.figure(fig_no, figsize=(20,20))
        ax=fig.add_subplot(111)
        ax = sns.heatmap(correlations, annot=True)
        ax.set_title("Correlation Matrix for Specified Attributes")
        ax.set_xticklabels(attr, rotation = 45)
        ax.set_yticklabels(attr, rotation = 45)
        plt.show()

    plot_corr_matrix(corrmatrix, numeric_features, 234)

# COMMAND ----------

# MAGIC %md
# MAGIC # Train/test split

# COMMAND ----------

# Filter the DataFrame based on the year
df_flights_numeric_non_null = df_flights_numeric_non_null.select("*").orderBy(F.rand())
if not new_data_flag:
    if random_split:
        ## random split
        train, test = df_flights_numeric_non_null.randomSplit([.8, .2], 512)
    else:
        ## time series split on year
        train = df_flights_numeric_non_null.filter(df_flights_numeric_non_null['YEAR'].isin(2015, 2016, 2017, 2018))
        test = df_flights_numeric_non_null.filter(df_flights_numeric_non_null['YEAR'] == 2019)

# COMMAND ----------

## Don't need to run this on large set
if run_PCA_analysis:
    # Run PCA Explain Variance Plot
    assembler = VectorAssembler(inputCols=numeric_features, outputCol="features")
    trainDF_VA = assembler.transform(train)

    scaler = StandardScaler(inputCol="features", outputCol="features_scaled")
    scaler_fit = scaler.fit(trainDF_VA)
    trainDF_VA = scaler_fit.transform(trainDF_VA)

    # begin PCA
    pca = PCA(k=len(numeric_features), inputCol="features_scaled", outputCol='pca_features')
    pca_applied = pca.fit(trainDF_VA)
    exp_var_pca = pca_applied.explainedVariance
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)

    # print("Begin Plotting Explaination of Variance")
    # Create the visualization plot
    plt.clf()
    plt.bar(range(0,len(exp_var_pca)), exp_var_pca, 
            alpha=0.5, align='center', 
            label='Individual explained variance')
    plt.step(range(0,len(cum_sum_eigenvalues)), 
            cum_sum_eigenvalues, 
            where='mid',
            label='Cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal component index')
    plt.legend(loc='best')
    plt.title(f"PCA Explained Variance Plot")
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Feature Transformation Pipeline
# MAGIC
# MAGIC To leverage Spark's machine learning pipeline, we vectorized both our numerical and one hot feautres. Specifically for the numerical features, we performed PCA on that set to reduce the total number of dimensions. 

# COMMAND ----------

# Create a pipeline to fully transform the data
def prepareDataSets(train, test, inputFeatures, onehotFeatures=[], doPCA=False, standardScale=False):
    '''
    Gets the train/test split, inputFeatures, and any one-hot encoded features and returns a train and test
    dataframe assembled ready to pipe into an Estimator (e.g. LogisticRegression) for training a model
    Optionally performs PCA and standard scaling on `inputFeatures`.
    '''

    outColva1 = "wFeatures"
    outCol = outColva1
    va1 = VectorAssembler(inputCols=inputFeatures, outputCol=outColva1)

    pipeline_stages = [va1]

    if standardScale:
        ## Standard Scaler
        outColScaler = "wFeatures_scaled"
        outCol = outColScaler
        standard_scaler = StandardScaler(inputCol=outColva1, outputCol=outColScaler)
        pipeline_stages.append(standard_scaler)

    if doPCA:
        ## PCA dimensionally reduction piece
        outColPCA = 'pca_wFeatures'
        outCol = outColPCA
        pca = PCA(k=10, inputCol=outColScaler, outputCol=outColPCA)
        pipeline_stages.append(pca)

    ## Serialize all remaining features
    complete_features = [outCol] + onehotFeatures
    final_assembler = VectorAssembler(inputCols=complete_features, outputCol=STANDARD_FEATURE_COLNAME)
    pipeline_stages.append(final_assembler)


    ## Setup the feature engineering pipeline and apply it onto the training data
    feature_engineered_pipeline = Pipeline(stages=pipeline_stages)

    ## apply the pipeline to the train and test set
    fitted_pipeline = feature_engineered_pipeline.fit(train)
    trainDF_VA = fitted_pipeline.transform(train)
    testDF_VA = fitted_pipeline.transform(test)

    return (trainDF_VA, testDF_VA)

# COMMAND ----------

numeric_features = ["HourlyPrecipitation", "HourlyDryBulbTemperature", 
                    "HourlyRelativeHumidity", "HourlySeaLevelPressure", 
                    "HourlyStationPressure", "HourlyVisibility",
                    "HourlyWindSpeed", "IS_HOLIDAY", "HOLIDAY_RUSH", 
                    "Recency", "Monetary", "Frequency", "ELEVATION",
                    'weather_condition_measured_time_from_depart',
                    "degree_centrality", "betweenness_centrality",
                    "closeness_centrality", "airport_rank"]

onehot_features = ['R_Quartile_onehot', 'F_Quartile_onehot', 'M_Quartile_onehot',
                   'closest_depart_time_weather_measurement_onehot',
                   'SEASON_onehot', 'TIME_SLOT_onehot', 
                   "airport_groups_onehot", "carrier_groups_onehot", 
                   "DAY_OF_MONTH_onehot"]

STANDARD_FEATURE_COLNAME = "input_features"
STANDARD_LABEL_COLNAME = "DEP_DEL15"

# COMMAND ----------

# Prepare data sets for model
if not new_data_flag:
    trainDF_VA, testDF_VA = prepareDataSets(train, test, numeric_features, onehot_features, True, True)
else:
    new_data, new_data_cloned = prepareDataSets(df_flights_numeric_non_null, df_flights_numeric_non_null, numeric_features, onehot_features, True, True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Resampling
# MAGIC
# MAGIC Since we noticed that a majority of class are not delayed (83%), we implementatted a resampling technique to upsample the low frequency class, delayed flight such that it will contain similar amount of flights as the high frequency class. For our study, we decided to pursue upsampling since we did not want to remove any data. The upsampled data are measured against the non-resampled data in term of performance metrics (test AUC) and training time.

# COMMAND ----------

def bootstrap_resampling(df, target_col, random_state, mode="upsampling"):
    """
        Resample the input dataframe based on the target column
        inputs:
            - df <pyspark.sql.DataFrame()>: dataframe input
            - target_col <string>  
                
        Parameters
        ----------
        df : pyspark.sql.DataFrame()
            DataFrame for the given data

        target_col : string
            Column label for the target  

        random_state : int
            numerical seed for random state of resampling

        mode : { string, choices=(upsampling, downsampling) },
            Mode of resampling

        Returns
        ----------
        augmented_df : pd.DataFrame()
            Resampled DataFrame such that the target are balanced
    """
    from functools import reduce
    from pyspark.sql import DataFrame

    temp_df = df
    
    # split the main dataframe into individual target dataframes
    label_splitted_dfs = []
    for label in temp_df.select(F.col(target_col)).distinct().collect():
        l = label[target_col]
        label_splitted_dfs.append(temp_df.filter(f"{target_col} = {l}"))

    # determine the label with the least/most sample
    df_lengths = [df.count() for df in label_splitted_dfs]
    min_length_index = df_lengths.index(min(df_lengths))
    max_length_index = df_lengths.index(max(df_lengths))


    # performing bootstrap replicant according to input mode
    if mode == 'upsampling':

        # # remove the df with most sample from list
        # # upsample all remaining df to have the same length as the df with most sample 
        most_sample_df = label_splitted_dfs[max_length_index]
        label_splitted_dfs.pop(max_length_index) 

        # creating ratio detailing max/min classes
        ratio = int(df_lengths[max_length_index]/df_lengths[min_length_index])
        temp_df_list = [most_sample_df]
        for df in label_splitted_dfs:
            a = range(ratio)
            sub_sample = df.withColumn("dummy", F.explode(F.array([F.lit(x) for x in a]))).drop('dummy')
            temp_df_list.append(sub_sample)

        augmented_df = reduce(DataFrame.unionByName, temp_df_list)
        augmented_df = augmented_df.select("*").orderBy(F.rand())
        augmented_df = augmented_df.repartition(252).cache()  # evenly distributed the shuffle data
        return augmented_df
    
    elif mode == 'downsampling':
        # remove the df with least sample from list
        # downsample all remaining df to have the same length as the df with least sample 
        least_sample_df = label_splitted_dfs[min_length_index]
        label_splitted_dfs.pop(min_length_index)

        temp_df_list = [least_sample_df]
        for df in label_splitted_dfs:
            sub_sample = df.sample(withReplacement=False, # sample without replacement
                                    n=min(df_lengths)/sum(df_lengths), # resample until dataframe reach
                                    seed=random_state) # reproducible result
            temp_df_list.append(sub_sample)


        # concat everything together and return the resampled dataframe
        # remember to shuffle the data!
        augmented_df = reduce(DataFrame.unionByName, temp_df_list)
        augmented_df = augmented_df.select("*").orderBy(F.rand())
        augmented_df = augmented_df.repartition(252).cache()  # evenly distributed the shuffle data
        return augmented_df
    else:
        print("Mode input incorrect, exiting...")
        return

# COMMAND ----------

# MAGIC %md
# MAGIC ## SMOTE Implementation
# MAGIC
# MAGIC The idea behind SMOTE is to introduce some level of randomness when creating synthetic data for the minority class. The machniery of SMOTE is performing many batches of k-nearest neighbours on the miniory class of data and by that, generate a new set of datapoint that follows the outputs neighbors. Typically SMOTE is a more robust way to upsample the data compared to traditional resampling technique. For our experiment, we modified the implementation described in [SMOTE implementation in PySpark](https://medium.com/@hwangdb/smote-implementation-in-pyspark-76ec4ffa2f1d) to upsample our data using Spark's Vectorized UDF. However, we're still facing challenges with long runtime (> 10 hours) when applying SMOTE upsampling on a 3M dataset.

# COMMAND ----------


import random
import numpy as np
from functools import reduce
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.functions import rand,col,when,concat,substring,lit,udf,lower,sum as ps_sum,count as ps_count,row_number
from pyspark.sql.window import *
from pyspark.sql import DataFrame
from pyspark.ml.feature import VectorAssembler,BucketedRandomProjectionLSH,VectorSlicer
from pyspark.sql.window import Window
from pyspark.ml.linalg import Vectors,VectorUDT
from pyspark.sql.functions import array, create_map, struct
from pyspark.sql.functions import pandas_udf

############################## spark smote oversampling ##########################
#for categorical columns, must take its stringIndexed form (smote should be after string indexing, default by frequency)

def smote(vectorized_sdf, features, label, smote_config):
    '''
    contains logic to perform smote oversampling, given a spark df with 2 classes
    inputs:
    * vectorized_sdf: cat cols are already stringindexed, num cols are assembled into 'features' vector
      df target col should be 'label'
    * smote_config: config obj containing smote parameters
    output:
    * oversampled_df: spark df after smote oversampling
    '''
    cols = [features, label]
    vectorized_sdf = vectorized_sdf.select(*cols)
    dataInput_min = vectorized_sdf.filter(f"{label} == '1'")
    dataInput_maj = vectorized_sdf.filter(f"{label} == '0'")
    
    # LSH, bucketed random projection
    brp = BucketedRandomProjectionLSH(inputCol="input_features", outputCol="hashes",seed=smote_config['seed'], bucketLength=smote_config['bucketLength'])

    # smote only applies on existing minority instances    
    model = brp.fit(dataInput_min)
    model.transform(dataInput_min)

    # here distance is calculated from brp's param inputCol
    self_join_w_distance = model.approxSimilarityJoin(dataInput_min, dataInput_min, float("inf"), distCol="EuclideanDistance")

    # remove self-comparison (distance 0)
    self_join_w_distance = self_join_w_distance.filter(self_join_w_distance.EuclideanDistance > 0)

    over_original_rows = Window.partitionBy("datasetA").orderBy("EuclideanDistance")

    self_similarity_df = self_join_w_distance.withColumn("r_num", F.row_number().over(over_original_rows))

    self_similarity_df_selected = self_similarity_df.filter(self_similarity_df.r_num <= smote_config["k"])

    over_original_rows_no_order = Window.partitionBy('datasetA')

    # list to store batches of synthetic data
    res = [dataInput_min, dataInput_maj]
    
    # two udf for vector add and subtract, subtraction include a random factor [0,1]
    subtract_vector_udf = F.pandas_udf(lambda arr: random.uniform(0, 1)*(arr[0]-arr[1]), VectorUDT())
    add_vector_udf = F.pandas_udf(lambda arr: arr[0]+arr[1], VectorUDT())


    # apply SMOTE iterations
    for i in range(smote_config['multiplier']):
        print("generating batch %s of synthetic instances"%i)
        # logic to randomly select neighbour: pick the largest random number generated row as the neighbour
        df_random_sel = self_similarity_df_selected.withColumn("rand", F.rand()).withColumn('max_rand', F.max('rand').over(over_original_rows_no_order))\
                            .where(F.col('rand') == F.col('max_rand')).drop(*['max_rand','rand','r_num'])
        # create synthetic feature numerical part
        df_vec_diff = df_random_sel.select('*', subtract_vector_udf(F.array('datasetA.input_features', 'datasetB.input_features')).alias('vec_diff'))
        df_vec_modified = df_vec_diff.select('*', add_vector_udf(F.array('datasetA.input_features', 'vec_diff')).alias('input_features'))
        # df_vec_diff = df_random_sel.withColumn('vec_diff', subtract_vector_udf(F.col('')))
        
        # this df_vec_modified is the synthetic minority instances,
        # df_vec_modified = df_vec_modified.drop(*['datasetA','datasetB','vec_diff','EuclideanDistance'])
        df_vec_modified = df_vec_modified.select(features)\
                                        .withColumn('DEP_DEL15', F.lit(1))
        res.append(df_vec_modified)
    
    oversampled_df = reduce(DataFrame.unionByName, res)
    oversampled_df = oversampled_df.select("*").orderBy(F.rand())
    oversampled_df = oversampled_df.repartition(252).cache()  # evenly distributed the shuffle data
    # # union synthetic instances with original full (both minority and majority) df
    # # oversampled_df = dfunion.union(vectorized_sdf.select(dfunion.columns))
    
    return oversampled_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Data to Blob Storage

# COMMAND ----------

# standardized the features and label name
STANDARD_FEATURE_COLNAME = "input_features"
STANDARD_LABEL_COLNAME = "DEP_DEL15"

# COMMAND ----------

# DATA Loading flags
perform_resampling = False
resample_by_year = False
apply_smote = False
new_data = False

# COMMAND ----------

smote_config = {
        "k": 3,              # number of neighbors
        "multiplier": 3,     # number of synthetic batches
        "seed": 234,         # random seeds
        "bucketLength": 3   # bucket length to control the probability of features being hashed into the same bucket
}

if perform_resampling:
    # resample on pyspark does not currently support exact number of samples... Need to rethink this strategy
    # apply resample by year strategy
    if resample_by_year:

        # get the available years in the main dataframe
        years = trainDF_VA.select('YEAR').distinct().collect()

        # for each year, perform upsampling and append to a list
        # aggregate the list of DFs together
        subsample_list = []
        for year in sorted(years):
            y = year['YEAR']
            year_sliced_df = trainDF_VA.filter(f'YEAR = {y}')

            if apply_smote:
                year_sliced_df_resample = smote(year_sliced_df, 
                                                STANDARD_FEATURE_COLNAME,
                                                STANDARD_LABEL_COLNAME, 
                                                smote_config)
                subsample_list.append(year_sliced_df_resample)
            else:
                year_sliced_df_resample = bootstrap_resampling(year_sliced_df, 
                                                               STANDARD_LABEL_COLNAME, 
                                                               random_state=541, 
                                                               mode="upsampling")
                subsample_list.append(year_sliced_df_resample)
            
        trainDF_VA_resample = reduce(DataFrame.unionByName, subsample_list)
        trainDF_VA_resample = trainDF_VA_resample.select("*").orderBy(F.rand())

        if apply_smote:
            trainDF_VA_resample.repartition(512)\
                                .write.mode("overwrite")\
                                .parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_SMOTE_yearly_resample")
        else:
            trainDF_VA_resample.repartition(512)\
                                .write.mode("overwrite")\
                                .parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_yearly_resample")

    else:
        if apply_smote:
            trainDF_VA_resample = bootstrap_resampling(trainDF_VA, 
                                                       STANDARD_LABEL_COLNAME, 
                                                       random_state=541, 
                                                       mode="upsampling")
            trainDF_VA_resample.repartition(512)\
                                .write.mode("overwrite")\
                                .parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_resample")
        else:
            trainDF_VA_resample = smote(trainDF_VA, 
                                        STANDARD_FEATURE_COLNAME, 
                                        STANDARD_LABEL_COLNAME, 
                                        smote_config)
            trainDF_VA_resample.repartition(512)\
                                .write.mode("overwrite")\
                                .parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_SMOTE_resample")
else:
    # write to ADLS
    trainDF_VA.repartition(512).write.mode("overwrite").parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA")
testDF_VA.repartition(512).write.mode("overwrite").parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_test_VA")

# COMMAND ----------

if new_data_flag:
    new_data.repartition(512).write.mode("overwrite").parquet(f"{team_blob_url}/OTPW_Flights/{dataset_name}_new_data_VA")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Read Data From Blob Storage

# COMMAND ----------

# DATA Loading flags
perform_resampling = True
resample_by_year = True
apply_smote = False
new_data = False

# COMMAND ----------


# read data from ADLS
if perform_resampling:
    if resample_by_year:
        trainDF_VA = spark.read.format('parquet').load(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_yearly_resample")
    else:
        trainDF_VA = spark.read.format('parquet').load(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA_resample")
else:
    # write to ADLS
    trainDF_VA = spark.read.format('parquet').load(f"{team_blob_url}/OTPW_Flights/{dataset_name}_train_VA")
testDF_VA = spark.read.format('parquet').load(f"{team_blob_url}/OTPW_Flights/{dataset_name}_test_VA")

# COMMAND ----------

# read_new data to blob storage
if new_data_flag:
    new_data = spark.read.format('parquet').load(f"{team_blob_url}/OTPW_Flights/{dataset_name}_new_data_VA")

# COMMAND ----------

# MAGIC %md
# MAGIC # Modelings & Pipeline
# MAGIC
# MAGIC ## Pipeline Overview
# MAGIC
# MAGIC After processing the data through our feature engineere pipeline and split our data into train and test set, we fitted four different models onto the trained data set. There are two types of features in our model, numerics features, which takes on any decimal values, and one-hot encoded features, which represents categorical features. While the list of input features changes across iterations, across all experiment performed onto our dataset, we detrmine that the follows are the best numeric features and one-hot encoded features across all model:
# MAGIC
# MAGIC |Feature Type| Faetures|
# MAGIC |------------|---------|
# MAGIC |numeric_features | ["HourlyPrecipitation", "HourlyDryBulbTemperature", "HourlyRelativeHumidity", "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyVisibility", "HourlyWindSpeed", "IS_HOLIDAY", "HOLIDAY_RUSH", "Recency", "Monetary", "Frequency", "ELEVATION", 'weather_condition_measured_time_from_depart', "degree_centrality", "betweenness_centrality", "closeness_centrality", "airport_rank"]|
# MAGIC |onehot_features | ['R_Quartile_onehot', 'F_Quartile_onehot', 'M_Quartile_onehot','closest_depart_time_weather_measurement_onehot','SEASON_onehot', 'TIME_SLOT_onehot', "airport_groups_onehot", "carrier_groups_onehot", "DAY_OF_MONTH_onehot"] |
# MAGIC
# MAGIC
# MAGIC For determination of flight delay indicator (DEP_DEL15), we decided to use AUC because it is a calibrated trade-off between sensitivy and specifity.One way of interpreting AUC is as the probability that the model ranks a random positive example more highly than a random negative example ([Google, Classication: ROC curve and AUC](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)).
# MAGIC

# COMMAND ----------

# import requests
# from IPython.display import display, Image
# # ![model_pipeline](http://13.66.213.192/w261/generic_model_pipeline-2.png)
# # URL of the image
# image_url = "http://13.66.213.192/w261/generic_model_pipeline-2.png"

# # Use requests to get the image
# response = requests.get(image_url)

# # Check if the request was successful (status code 200)
# if response.status_code == 200:
#     # Display the image in the notebook
#     img = Image(response.content)
#     display(img)
# else:
#     print(f"Image is not accessible. Status code: {response.status_code}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## HyperParam Tuning: Grid search k-fold cross-validation
# MAGIC
# MAGIC Since logistic regression shows promising results, we decided to apply hyper-parameter tuning on the logistic regression using the 3M dataset. Here, we're interested in fine-tuning the model using the parameters listed in table 1. As stated earlier, we decided to use AUC as our performance metric at each k-fold cross-validation since it is a calibrated trade-off between sensitivy and specifity. This means that the AUC metric would have accounted for the predicted class imablance that we noted in above.
# MAGIC
# MAGIC The flights data is essentially a time series, given that flight delays are autocorrelated. Past delays impact future delays in a given time window, therefore whether a flight is delayed or not is not independent of a subsequent flight being delayed. Given the nature of the data, we have to adjust the typical k-fold cross-validation method to account for this autocorrelation. This lead us to roll our own **time-series blocked cross-validation method**, where we're essentially selecting folds in a serialized fashion from the beginning of the time horizon, forwards until we reach the end of the training subset. 
# MAGIC
# MAGIC In this fashion, we can have both a better estimate of the models' performance on unseen data, as well as serving for hyperparameter tuning. For this latter goal, we are generating the power set from a provided search space for the hyper-parameters that we can tune on each model. 
# MAGIC
# MAGIC
# MAGIC #### Table 1: Logistic Regression Model Parameters
# MAGIC |Parameter | ranges    | description|
# MAGIC |----------|-----------|------------|
# MAGIC |'maxIter' | [100, 200, 300] | max nubmer of iteration for the iteration to converges |
# MAGIC |'regParam'| [0.01, 0.001, 0.00] | regularization paramter |
# MAGIC |'elasticNetParam'|  [0.0, 0.25, 0.5, 0.75, 1.0] | elastic mixing paramters for L1 and L2 penalty |
# MAGIC
# MAGIC
# MAGIC We performed several experiments oo evaluate the performance of different cross-validation methods, Random CV and Time Series CV. From our experiment, random CV took the most amount of time to complete. Additionally, we notice that while Tthe time series CV yield a slightly higher test AUC score, the parameters were the same as the random CV.
# MAGIC The differences in AUC are then attributed to the sampling variance associated with the different k-fold partitioning scheme.
# MAGIC
# MAGIC #### Table 2: Logistic Regression Best search parameters
# MAGIC | Runs | Max Iteration | ElasticNet Params | reg Params | test AUC | Time |
# MAGIC | ---- | ------------- | ----------------- | ---------- | -------- | ---- |
# MAGIC | Random CV       | 100 | 0.00             | 0.00       | 0.612    | 6.2h    |
# MAGIC | Time Series CV  | 100 | 0.00             | 0.00       | 0.626    | 1.2h    |
# MAGIC
# MAGIC
# MAGIC ## Experimental Performed
# MAGIC Following the gridsearch results, we performed a total of 63 experiments. Of which, about 30 experiments were performed on an upsampled dataset performed on the minority class (delayed by 15 minutes “DEP_DEL15”). With the exception of decision tree performed on the 3M dataset, which we suspected to have a data leakage problem, our results showed that Logistic Regression yield the best performance AUC metric on a held out testing set follows by Neural network. Furthermore, the test AUC scores are higher in the 3M than 60M dataset, but otherwise are consistent across models. Table 4 reports the performance metrics on held out test set on the baseline majority class across each dataset (3 months, 12 months, and 60 months).
# MAGIC
# MAGIC Across all fitted models, Logistic regression performed the best of the held out test set, reported an AUC score of 0.6976 for the 3M dataset, 0.6355 for the 12M dataset, and 0.6562 for 2019 held out test set. Compared to each baseline experiment, where we predicted the majority class (no delay), this is an improvement of 39.5% for the 3M dataset, 27.0% for 12M dataset, and 31.2% for the held out 2019 dataset.
# MAGIC
# MAGIC
# MAGIC #### Table 4: Baseline Experiments
# MAGIC | Duration | Name                              | dataset_name         | test_AUC | test_accuracy | test_precision | test_recall |
# MAGIC | -------- | --------------------------------- | -------------------- | -------- | ------------- | -------------- | ----------- |
# MAGIC | 4.7min   | majority_class_basline_ascending  | OTPW_3M (resampled)  | 0.5      | 0.712532      | 0.712403       | 0.712514    |
# MAGIC | 7.0min   | majority_class_basline_ascending  | OTPW_12M (original)  | 0.5      | 0.710948      | 0.713924       | 0.711815    |
# MAGIC | 2.6min   | majority_class_basline_ascending  | OTPW_3M (original)   | 0.5      | 0.711731      | 0.711731       | 0.711731    |
# MAGIC | 29.1s    | majority_class_basline_ascending  | OTPW_60M (original)  | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC | 1.2min   | majority_class_basline_ascending  | OTPW_60M (resampled) | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC | 10.8min  | majority_class_basline_descending | OTPW_3M (resampled)  | 0.5      | 0.713294      | 0.712194       | 0.711574    |
# MAGIC | 2.7min   | majority_class_basline_descending | OTPW_3M (original)   | 0.5      | 0.711731      | 0.711731       | 0.711731    |
# MAGIC | 30.9s    | majority_class_basline_descending | OTPW_60M (original)  | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC | 1.7min   | majority_class_basline_descending | OTPW_60M (resampled) | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC | 7.3min   | majority_class_basline_random     | OTPW_3M (resampled)  | 0.5      | 0.712909      | 0.713183       | 0.711072    |
# MAGIC | 21.3min  | majority_class_basline_random     | OTPW_12M (original)  | 0.5      | 0.711731      | 0.711731       | 0.711731    |
# MAGIC | 29.3s    | majority_class_basline_random     | OTPW_60M (resampled) | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC | 1.8min   | majority_class_basline_random     | OTPW_60M (reampled)  | 0.5      | 0.734695      | 0.734695       | 0.734695    |
# MAGIC
# MAGIC ### Key Experiments: Original - Resampled - CV
# MAGIC
# MAGIC Of the 63 experiments, some were performed on original dataset and some were perfomed on the resampled dataset. The two tables below consolidates only the best experiment results based on testAUC metric grouped by each dataset. Since we're interested in utilizing the model to predict delay on future cases, the experiments were not performed on the upsampled dataset. 
# MAGIC
# MAGIC #### Table 5: 3 Months Dataset Experiments
# MAGIC
# MAGIC | Duration | dataset_name       | family                                                                                           | test_AUC | test_accuracy | test_precision | test_recall |
# MAGIC | -------- | ------------------ | ------------------------------------------------------------------------------------------------ | -------- | ------------- | -------------- | ----------- |
# MAGIC | 36.7s    | OTPW_3M (resample) | binary decision tree                                                                             | 0.763204 | 0.765326      | 0.765326       | 0.765326    |
# MAGIC | 15.0min  | OTPW_3M (original) | binomial logistic regression                                                                     | 0.697641 | 0.802631      | 0.760873       | 0.802631    |
# MAGIC | 1.9min   | OTPW_3M (resample) | binomial logistic regression                                                                     | 0.697477 | 0.635851      | 0.765814       | 0.635851    |
# MAGIC | 51.1s    | OTPW_3M (resample) | Gradient Boosting Classifier                                                                     | 0.660159 | 0.683178      | 0.683178       | 0.683178    |
# MAGIC | 4.1min   | OTPW_3M (resample) | Ranfom Forest Classifier                                                                         | 0.650896 | 0.673618      | 0.673618       | 0.673618    |
# MAGIC | 10.1h    | OTPW_3M (resample) | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - MLP-2 Softmax | 0.641032 | 0.679022      | 0.679022       | 0.679022    |
# MAGIC | 11.5min  | OTPW_3M (resample) | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - MLP-2 - Softmax                                        | 0.630484 | 0.655642      | 0.655642       | 0.655642    |
# MAGIC | 8.2min   | OTPW_3M (original) | Gradient Boosting Classifier                                                                     | 0.537543 | 0.745222      | 0.745276       | 0.744245    |
# MAGIC | 9.0min   | OTPW_3M (original) | binary decision tree                                                                             | 0.513612 | 0.724226      | 0.725219       | 0.724181    |
# MAGIC | 20.4min  | OTPW_3M (original) | Ranfom Forest Classifier                                                                         | 0.513486 | 0.725809      | 0.72416        | 0.724419    |
# MAGIC
# MAGIC #### Table 6: 60 Months Dataset Experiments
# MAGIC | Duration | dataset_name         | family                                                                                             | test_AUC  | test_accuracy | test_precision | test_recall |
# MAGIC | -------- | -------------------- | -------------------------------------------------------------------------------------------------- | --------- | ------------- | -------------- | ----------- |
# MAGIC | 11.1min  | OTPW_60M (resampled) | binomial logistic regression                                                                       | 0.656224  | 0.659482822   | 0.760371753    | 0.659482822 |
# MAGIC | 3.1min   | OTPW_60M (original)  | binomial logistic regression                                                                       | 0.6562238 | 0.817303234 | 0.768425354 | 0.817303234 |
# MAGIC | 6.2h     | OTPW_60M (resampled ) | binomial logistic regression + random cross-validation                                            | 0.6119155 | 0.693256355   | 0.693256355    | 0.693256355 |
# MAGIC | 5.6min   | OTPW_60M (resampled) | Gradient Boosting Classifier                                                                       | 0.6105613 | 0.695115082   | 0.695115082    | 0.695115082 |
# MAGIC | 4.7min   | OTPW_60M (resampled) | Gradient Boosting Classifier                                                                       | 0.6103969 | 0.696020997   | 0.696020997    | 0.696020997 |
# MAGIC | 27.8min  | OTPW_60M (resampled) | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - MLP-2 - Softmax                                          | 0.6091038 | 0.697003518   | 0.697003518    | 0.697003518 |
# MAGIC | 7.2h     | OTPW_60M (resampled) | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - MLP-2 - Softmax | 0.6088221 | 0.694548099   | 0.694548099    | 0.694548099 |
# MAGIC | 2.4min   | OTPW_60M (resampled) | Ranfom Forest Classifier                                                                           | 0.6065553 | 0.69363982    | 0.69363982     | 0.69363982  |
# MAGIC | 2.3min   | OTPW_60M (resampled) | Ranfom Forest Classifier                                                                           | 0.6009704 | 0.704577975   | 0.704577975    | 0.704577975 |
# MAGIC | 7.1min   | OTPW_60M (resampled) | binary decision tree                                                                               | 0.5978069 | 0.692065533   | 0.692065533    | 0.692065533 |
# MAGIC | 25.5min  | OTPW_60M (resampled) | binary decision tree                                                                               | 0.5924744 | 0.694255876   | 0.694255876    | 0.694255876 |
# MAGIC | 25.6min  | OTPW_60M (resampled) | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - MLP-2 - Softmax                                          | 0.5908096 | 0.692781659   | 0.692781659    | 0.692781659 |
# MAGIC | 5.1min   | OTPW_60M (original)  | binary decision tree                                                                               | 0.515814  | 0.746065317   | 0.746065317    | 0.746065317 |
# MAGIC | 3.8min   | OTPW_60M (original)  | Gradient Boosting Classifier                                                                       | 0.5052957 | 0.739331342   | 0.739331342    | 0.739331342 |
# MAGIC | 3.7h     | OTPW_60M (resampled) | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - MLP 2 - Softmax | 0.501165  | 0.735809378   | 0.735809378    | 0.735809378 |
# MAGIC | 53.6min  | OTPW_60M (original)  | Ranfom Forest Classifier                                                                           | 0.5       | 0.734695239   | 0.734695239    | 0.734695239 |
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Time-series k-fold cross-validation

# COMMAND ----------

def parameter_sets(search_space):
  ''' Create a dictionary combination of the parameter grid '''
  parameter_names = list(search_space.keys())
  v = search_space.values()
  combinations = list(itertools.product(*v))
  return parameter_names, combinations

def time_series_CV(dataset, search_space, fixed_params, model, k=3, metric='auc'):
    '''
    k-fold cross validation for time-series
    '''

    # Initiate trackers
    best_score = 0
    best_param_vals = None

    df = dataset
    n = df.count()

    # Assign sequential row IDs based on `FL_DATE` # TODO is this the right col?
    df = df.withColumn('row_id', F.row_number().over(Window.partitionBy().orderBy('FL_DATE')))
    chunk_size = int(n/(k+1))

    parameter_names, parameters = parameter_sets(search_space)

    # assert len(parameters.keys()) >= 1, 'Insufficient parameters entered'

    for p in parameters:
        # Print parameter set
        param_print = {x[0]:x[1] for x in zip(parameter_names,p)}
        print(f'Parameters: {param_print}')

        estimator = model(**param_print, **fixed_params)

        # Track score
        scores = []

        # k-folds
        for i in range(k):

            # Do the split
            train_df = df.filter((F.col('row_id') > chunk_size * i)&(F.col('row_id') <= chunk_size * (i+1))).cache()
            test_df  = df.filter((F.col('row_id') > chunk_size * (i+1))&(F.col('row_id') <= chunk_size * (i+2))).cache()

            # Fit the model
            fit_model = estimator.fit(train_df)

            # Predict on held out test set
            test_predictions = fit_model.transform(test_df)

            # Get the score
            evaluator = BinaryClassificationEvaluator(labelCol=STANDARD_LABEL_COLNAME, rawPredictionCol="prediction", metricName='areaUnderROC')
            score = evaluator.evaluate(test_predictions)
            scores.append(score)

            # Set best parameter set to current one for first fold
            if best_param_vals is None:
                best_param_vals = p

        # Take average of all scores
        avg_score = np.average(scores)

        # Update best score and parameter set to reflect optimal test performance
        if avg_score > best_score:
            previous_best = best_score
            best_score = avg_score
            best_parameters = param_print
            best_param_vals = p
            print(f'New best score of {best_score:.2f}')
        else:
            print(f'Result was no better, score was {avg_score:.2f} with best {metric} score {best_score:.2f}')
        print('\n')

    return best_parameters, best_score


# COMMAND ----------

# MAGIC %md
# MAGIC ### Spark Random K-Fold Cross Validation

# COMMAND ----------

if grisearch:
    with mlflow.start_run(run_name='logreg_random_cv') as run:

        # TODO: add these code under each of the model pipeline so that we Gridsearch each model type!
        log_reg = LogisticRegression(family="binomial", 
                                    featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15")


        # elastic mixing paramters for L1 and L2 penalty
        # max nubmer of iteration for the iteration to converges
        # regularization paramter ??? What is this paramter?
        paramGrid = ParamGridBuilder() \
                    .addGrid(log_reg.elasticNetParam, [0.0, 0.25, 0.5, 0.75, 1.0]) \
                    .addGrid(log_reg.maxIter, np.arange(100, 200, 10)) \
                    .addGrid(log_reg.regParam, [0.01, 0.001, 0.00]) \
                    .build()

        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')


        crossval = CrossValidator(estimator=log_reg, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)
        
        # cache the data to avoid memory issues
        trainDF_VA = trainDF_VA.cache()

        # Run the cross validation
        log_reg_cvModel = crossval.fit(trainDF_VA)

        predictions = log_reg_cvModel.transform(testDF_VA)

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)
        mlflow.spark.log_model(log_reg_cvModel, "flights_log_reg_CV")

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")
        print("best_param model", log_reg_cvModel.getEstimatorParamMaps()[ np.argmax(log_reg_cvModel.avgMetrics) ] )

# COMMAND ----------

if grisearch:
    with mlflow.start_run(run_name='dt_random_cv') as run:
        # Decision Tree search
        ## Run Simple DT and get training metrics
        ## Fit Decision Tree model
        dt = DecisionTreeClassifier(featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15", 
                                    seed=456)
        
        
        # number of trees to train
        # the maximum depth of the tree; how extensive should the tree grow
        # regularization paramter ??? What is this paramter?
        # Max number of bins for discretizing continuous features
        paramGrid = ParamGridBuilder() \
                    .addGrid(dt.maxDepth, np.arange(2, 10, 1)) \
                    .addGrid(dt.maxBins, np.arange(2, 10, 1))\
                    .build()


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')


        crossval = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)

        # cache the data to avoid memory issues
        trainDF_VA = trainDF_VA.cache()

        # Run the cross validation
        dt_cvModel = crossval.fit(trainDF_VA)

        ## Obtain the classification metrics for training set
        predictions = dt_cvModel.transform(testDF_VA)


        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)


        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)
        mlflow.spark.log_model(dt_cvModel, "flights_DT_CV")


        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")
        print("best_param model", dt_cvModel.getEstimatorParamMaps()[ np.argmax(dt_cvModel.avgMetrics) ] )

# COMMAND ----------

if grisearch:
    with mlflow.start_run(run_name='rf_random_cv') as run:

        # Random Forest search
        
        ## Run Simple RF and get training metrics
        ## Fit Ranfom Forest model
        rf = RandomForestClassifier(featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15", 
                                    seed=456)
        
        
        # number of trees to train
        # the maximum depth of the tree; how extensive should the tree grow
        # regularization paramter ??? What is this paramter?
        # Max number of bins for discretizing continuous features
        paramGrid = ParamGridBuilder() \
                    .addGrid(rf.numTrees, np.arange(100, 200, 50)) \
                    .addGrid(rf.maxDepth, np.arange(2, 10, 2)) \
                    .addGrid(rf.maxBins, np.arange(2, 10, 2))\
                    .build()


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')


        crossval = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)


        # cache the data to avoid memory issues
        trainDF_VA = trainDF_VA.cache()

        # Run the cross validation
        rf_cvModel = crossval.fit(trainDF_VA)

        ## Obtain the classification metrics for training set
        predictions = rf_cvModel.transform(testDF_VA)


        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)
        mlflow.spark.log_model(rf_cvModel, "flights_RF_CV")

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")
        print("best_param model", rf_cvModel.getEstimatorParamMaps()[ np.argmax(rf_cvModel.avgMetrics) ] )

# COMMAND ----------

if grisearch:
    with mlflow.start_run(run_name='gbt_random_cv') as run:

        # GBT  search
        ## Run Simple GBT and get training metrics
        ## Fit Ranfom Forest model
        gbt = GBTClassifier(featuresCol=STANDARD_FEATURE_COLNAME, 
                            labelCol="DEP_DEL15", 
                            seed=456)
        
        # Number of iteration to apply boosting of the residuals
        # the maximum depth of the tree; how extensive should the tree grow
        # regularization paramter ??? What is this paramter?
        # Max number of bins for discretizing continuous features
        paramGrid = ParamGridBuilder() \
                    .addGrid(gbt.maxIter, np.arange(50, 100, 50)) \
                    .addGrid(gbt.maxDepth, np.arange(2, 10, 2)) \
                    .addGrid(gbt.maxBins, np.arange(2, 10, 1))\
                    .build()


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')


        crossval = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=3)


        # cache the data to avoid memory issues
        trainDF_VA = trainDF_VA.cache()

        # Run the cross validation
        gbt_cvModel = crossval.fit(trainDF_VA)

        ## Obtain the classification metrics for training set
        predictions = gbt_cvModel.transform(testDF_VA)

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)
        mlflow.spark.log_model(gbt_cvModel, "flights_GBT_CV")


        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")
        print("best_param model", gbt_cvModel.getEstimatorParamMaps()[ np.argmax(gbt_cvModel.avgMetrics) ] )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Hyperopt tuning

# COMMAND ----------

from hyperopt import hp, fmin, tpe, Trials

def lr_objective_func(params):
    #maxIter = params['maxIter']
    regParam = params['regParam']
    elasticNetParam = params['elasticNetParam']
    with mlflow.start_run():
        estimator = LogisticRegression(family="binomial",
                                featuresCol=STANDARD_FEATURE_COLNAME, 
                                labelCol=STANDARD_LABEL_COLNAME,
                                maxIter=100,
                                regParam=regParam,
                                elasticNetParam=elasticNetParam
                                )
        model = estimator.fit(trainDF_VA)
        
        # use the balanced accuracy for our metric
        #evaluator = BinaryClassificationEvaluator(labelCol=STANDARD_LABEL_COLNAME, rawPredictionCol="prediction", metricName='areaUnderROC')

        #predictions = model.evaluate(testDF_VA)

        # Evaluation
        test_results = model.evaluate(testDF_VA)
        mlflow.log_metric("AUC", test_results.areaUnderROC)
        areaUnderROC = test_results.areaUnderROC

        # # Calc balanced accuracy
        # tp = predictions.filter("DEP_DEL15 = 1 AND prediction = 1").count()
        # fp = predictions.filter("DEP_DEL15 = 0 AND prediction = 1").count()
        # tn = predictions.filter("DEP_DEL15 = 0 AND prediction = 0").count()
        # fn = predictions.filter("DEP_DEL15 = 1 AND prediction = 0").count()
        # tnr = tn / (tn + fp)
        # tpr = tp / (tp + fn)

        # # For binary class, this is just the mean
        # balanced_accuracy = (tnr + tpr) / 2.0

        # mlflow.log_metric("balanced_accuracy_lr", balanced_accuracy)

    return areaUnderROC

search_space = {
    #'maxIter': hp.quniform('maxIter', 70, 160, 10),
    'regParam': hp.loguniform('regParam', -5, 2),
    'elasticNetParam': hp.uniform('elasticNetParam', 0, 1)
}

# Disable mlflow logging for the optimization phase
mlflow.pyspark.ml.autolog(log_models=False)
trials = Trials()

# COMMAND ----------

run_hyperopt = True
from hyperopt.early_stop import no_progress_loss
# [HECTOR] Something is not working here, this took forever, and still would go on forever...
if run_hyperopt:
    lg_best_hyperparam = fmin(fn=lr_objective_func,
                            space=search_space,
                            algo=tpe.suggest,
                            trials=trials,
                            rstate=np.random.default_rng(42),
                            max_evals=40,
                            early_stop_fn=no_progress_loss(20)
                            )
    lg_best_hyperparam

# COMMAND ----------

lg_best_hyperparam

# COMMAND ----------

# creating a MLFlow Logistic Regression Baseline Model
if run_model_flag:
    with mlflow.start_run(run_name='flights_logistic_regression_hyperopt') as run:
        
        ## Run a simple logistic regression model and get training accuracy
        ## Fit Logistic Regression model
        log_reg = LogisticRegression(family="binomial",
                                    featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15",
                                    elasticNetParam=0.620118997676641,
                                    regParam=0.39387449470050384)

        fitted_log_reg = log_reg.fit(trainDF_VA)



        ## Obtain the classification metrics for training set
        train_results = fitted_log_reg.evaluate(trainDF_VA)

        # Metrics
        accuracy = train_results.accuracy
        precision = train_results.weightedPrecision
        recall =  train_results.weightedRecall 
        AUC =  train_results.areaUnderROC

        ## Log train metrics to MLFlow
        mlflow.log_metric("trained_accuracy", accuracy)
        mlflow.log_metric("trained_precision", precision)
        mlflow.log_metric("trained_recall", recall)
        mlflow.log_metric("trained_AUC", AUC)
        mlflow.spark.log_model(fitted_log_reg, "flights_logistic_regression_model")

        print("###########################Training Metric###########################")
        print(f"Model Training Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Training Precision: {np.round(precision*100, 3)}")
        print(f"Model Training Recall: {np.round(recall*100, 3)}")
        print(f"Model Training AUC: {np.round(AUC, 3)}")
        print("\n")

        ## Obtain the classification metrics for test set
        test_results = fitted_log_reg.evaluate(testDF_VA)

        # Metrics
        accuracy = test_results.accuracy
        precision = test_results.weightedPrecision
        recall =  test_results.weightedRecall
        AUC =  test_results.areaUnderROC

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Non-Grid Search: Model Pipeline Using MLFlow 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Majority Class Baseline Models Experimentation

# COMMAND ----------

# MAGIC %md
# MAGIC #### Baseline Descending Order

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Majority Class Baseline Model
    with mlflow.start_run(run_name='majority_class_basline_descending') as run:


        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "majority class")

        predictions = testDF_VA.withColumn('prediction', 
                                        F.when(F.rand(seed=451) < majority_class_proba, 0).otherwise(1).cast(T.DoubleType()))\
                                .orderBy('prediction', ascending = False)

        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Baseline Ascending Order

# COMMAND ----------

if run_model_flag:
    # creating a MLFlow Majority Class Baseline Model
    with mlflow.start_run(run_name='majority_class_basline_ascending') as run:
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "majority class")

        predictions = testDF_VA.withColumn('prediction', 
                                        F.when(F.rand(seed=451) < majority_class_proba, 0).otherwise(1).cast(T.DoubleType()))\
                                .orderBy('prediction', ascending = True)

        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")


# COMMAND ----------

# MAGIC %md
# MAGIC #### Baseline Random Order

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Majority Class Baseline Model
    with mlflow.start_run(run_name='majority_class_basline_random') as run:
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "majority class")


        predictions = testDF_VA.withColumn('prediction', 
                                        F.when(F.rand(seed=451) < majority_class_proba, 0).otherwise(1).cast(T.DoubleType()))

        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Logistic Regression

# COMMAND ----------

# # shuffling the dataset before training/testing
# trainDF_VA = trainDF_VA.select("*").orderBy(F.rand()).cache()
# testDF_VA = testDF_VA.select("*").orderBy(F.rand()).cache()

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Logistic Regression Baseline Model
    with mlflow.start_run(run_name='flights_logistic_regression') as run:
        
        ## Run a simple logistic regression model and get training accuracy
        ## Fit Logistic Regression model
        log_reg = LogisticRegression(family="binomial", 
                                    featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15")

        fitted_log_reg = log_reg.fit(trainDF_VA)

        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "binomial logistic regression")


        ## Obtain the classification metrics for training set
        train_results = fitted_log_reg.evaluate(trainDF_VA)

        # Metrics
        accuracy = train_results.accuracy
        precision = train_results.weightedPrecision
        recall =  train_results.weightedRecall 
        AUC =  train_results.areaUnderROC

        ## Log train metrics to MLFlow
        mlflow.log_metric("trained_accuracy", accuracy)
        mlflow.log_metric("trained_precision", precision)
        mlflow.log_metric("trained_recall", recall)
        mlflow.log_metric("trained_AUC", AUC)
        mlflow.spark.log_model(fitted_log_reg, "flights_logistic_regression_model")

        print("###########################Training Metric###########################")
        print(f"Model Training Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Training Precision: {np.round(precision*100, 3)}")
        print(f"Model Training Recall: {np.round(recall*100, 3)}")
        print(f"Model Training AUC: {np.round(AUC, 3)}")
        print("\n")

        ## Obtain the classification metrics for test set
        test_results = fitted_log_reg.evaluate(testDF_VA)

        # Metrics
        accuracy = test_results.accuracy
        precision = test_results.weightedPrecision
        recall =  test_results.weightedRecall
        AUC =  test_results.areaUnderROC

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision Tree

# COMMAND ----------

# Create datasets for Tree based without PCA and scaling
# trainDF_VA, testDF_VA = prepareDataSets(train, test, numeric_features, onehot_features, False, False)

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Logistic Regression Baseline Model
    with mlflow.start_run(run_name='flights_decision_tree') as run:
        
        ## Tun a simple Decision Tree model and get training metrics
        ## Fit Decision Tree model
        decision_tree = DecisionTreeClassifier(maxDepth=20,
                                            maxBins = 10,
                                            featuresCol=STANDARD_FEATURE_COLNAME, 
                                            labelCol="DEP_DEL15")
        fitted_decision_tree = decision_tree.fit(trainDF_VA)
        
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "binary decision tree")


        ## Obtain the classification metrics for training set
        predictions = fitted_decision_tree.transform(testDF_VA)


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Random Forest

# COMMAND ----------

# Test for Tree based with PCA and scaling (potential faster training)
# trainDF_VA, testDF_VA = prepareDataSets(train, test, numeric_features, onehot_features, True, True)

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Logistic Regression Baseline Model
    with mlflow.start_run(run_name='flights_random_foreast') as run:
        
        ## Run Simple RF and get training metrics
        ## Fit Ranfom Forest model
        # rf = RandomForestClassifier(numTrees=20,   # number of trees to train
        #                             maxDepth=10,    # The maximum depth of the tree; how extensive should the tree grow
        #                             featuresCol=STANDARD_FEATURE_COLNAME, 
        #                             labelCol="DEP_DEL15", 
        #                             seed=456)
        rf = RandomForestClassifier(featuresCol=STANDARD_FEATURE_COLNAME, 
                                    labelCol="DEP_DEL15", 
                                    seed=456)
        fitted_rf = rf.fit(trainDF_VA)
        
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "Ranfom Forest Classifier")


        ## Obtain the classification metrics for training set
        predictions = fitted_rf.transform(testDF_VA)


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gradient Boosting

# COMMAND ----------

if run_model_flag:

    # creating a MLFlow Logistic Regression Baseline Model
    with mlflow.start_run(run_name='flights_GBT') as run:
        
        ## Run Simple RF and get training metrics
        ## Fit Ranfom Forest model
        # gbt = GBTClassifier(maxDepth=20,      # The maximum depth of the tree; how extensive should the tree grow
        #                     maxIter=100,      # Number of iteration to apply boosting of the residuals
        #                     featuresCol=STANDARD_FEATURE_COLNAME, 
        #                     labelCol="DEP_DEL15", 
        #                     seed=456)
        gbt = GBTClassifier(featuresCol=STANDARD_FEATURE_COLNAME, 
                            labelCol="DEP_DEL15", 
                            seed=456)
        fitted_gbt = gbt.fit(trainDF_VA)
        
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "Gradient Boosting Classifier")


        ## Obtain the classification metrics for training set
        predictions = fitted_gbt.transform(testDF_VA)


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        ## Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Neural Networks
# MAGIC
# MAGIC Throughout the years, advancement in neural networks has been eponential. To rounded off the delay detection in our project, we're leveraging the availability of neural networks on spark and implemented two models, a shallow neural network with a single hidden layer and a deeper network with 4 hidden layers. Both neural networks has an input size of 32 features and an output of 2 with a softmax activation function. Due to the nature of our problem, where we're simply performing a binary classifcaiton, the sofmax activation function reduced to a sigmoidal function. According to Multilayer-Perceptron model detailed in the [spark documentation](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.classification.MultilayerPerceptronClassifier.html), each of the hidden layers also contains a sigmoid activation function and is represented in the following equation:
# MAGIC
# MAGIC $$
# MAGIC f(z_i) = \frac{1}{1+e^{-z_i}}
# MAGIC $$
# MAGIC
# MAGIC To minimize training time and reproducibility, a seed is set on each network and each network is trained for 100 iterations with a blocksize of 512. For the first neural network, we choose 64 as the single hidden layer size and can be represented as follow:
# MAGIC
# MAGIC $$
# MAGIC MLP 32 -> Sigmoid -> MLP 64 -> Sigmoid -> MLP 2 -> Softmax 
# MAGIC $$
# MAGIC
# MAGIC For the second neural network, we choose 128 as the first and third hidden layer size, 252 as the seond hidden layzer size, and 64 as the final hidden layer size to simulate a gradual increase in weigts that follows by a gradual decrease in weights as the network proceed toward making a predictions. This second network can be represent as follow:
# MAGIC
# MAGIC $$
# MAGIC MLP 32 -> Sigmoid -> MLP 128 -> Sigmoid -> MLP 252 -> Sigmoid -> MLP 128 -> Sigmoid -> MLP 64 -> Sigmoid -> MLP 2 -> Softmax 
# MAGIC $$
# MAGIC
# MAGIC Each neural networks were fitted on both 3 months and 60 months where the minority class are upsample to match proportionally with the majority class. Two additional runs were performed on 60M dataset where no upsampling were performed. Table 5 reports the experiment results. Interestingly, resampling of the 60M dataset shows a dramatically increase in the AUC score on the heldout 2019 set for the deep neural network (20 % increase) while only shows a minial increase in the AUC score for the shallow neural network (3% increase). Compared to the upsampled 3M dataset, the upsampled 60M dataset shows a decrease in training AUC for both shallow and deep neural networks, 3% and 5% decrease, respectively. This observation could be a case for leakage during training the training of the 3M dataset due to the random 80/20 split performed earlier. Furthermore, while the dense neural network shows a minimal improvement in AUC (less than 1 % improvement across experiment), the shallow neural network actually outperformed the dense neural network in term of training time with a greater than 90% reduction in training time.
# MAGIC
# MAGIC ### Table 5: Neural Network Experimental Results
# MAGIC | Model                                                                              | Dataset | Training Time | Test AUC | Test Accuracy | Test Precision | Test Recall |
# MAGIC | --------------------------------------------------------------------------------------------- | ------- | ------------- | -------- | ------------- | -------------- | ----------- |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - 2 Softmax                                           | 3 Months | 11.5 mim     | 0.6305   | 0.6556        | 0.6556         | 0.6556      |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - 2 Softmax  | 3 Months | 10.1 hr      | 0.6410   | 0.6790        | 0.6790	        | 0.6790      |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - 2 Softmax                                           | 60 Months | 27.8 mim    | 0.6091   | 0.6970        | 0.6970         | 0.6970      |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - 2 Softmax  | 60 Months | 7.2 hr      | 0.6088   | 0.6945        | 0.6945	        | 0.6945      |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-64 - Sigmoid - 2 Softmax (no upsample)                             | 60 Months | 25.6 mim    | 0.5908   | 0.6928        | 0.6927         | 0.6928      |
# MAGIC | NN: MLP-32 - Sigmoid - MLP-128 - Sigmoid - MLP-252 - Sigmoid - MLP-128 - Sigmoid - 2 Softmax (no upsample)    | 60 Months | 3.7 hr | 0.5012   |	0.7358   | 0.7358	  | 0.7358      |
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Shallow Neural Network

# COMMAND ----------

if run_model_flag:

    with mlflow.start_run(run_name='flights_NN') as run:

        # specify layers for the neural network:
        # input layer of size 32 (features), 1 intermediate of size 64
        # and output of size 2 (classes) 

        # MLP-32 - ReLu - MLP-64 - Relu - 2 Softmax 
        num_input_features = len(trainDF_VA
                                .select(STANDARD_FEATURE_COLNAME)
                                .take(1)[0][STANDARD_FEATURE_COLNAME]
                                .toArray())
        layers = [num_input_features, 64, 2]

        # create the trainer and set its parameters
        trainer = MultilayerPerceptronClassifier(maxIter=100, 
                                                layers=layers, 
                                                blockSize=512, 
                                                seed=523,
                                                featuresCol=STANDARD_FEATURE_COLNAME, 
                                                labelCol="DEP_DEL15")

        # train the model
        model = trainer.fit(trainDF_VA)

            
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "NN: MLP-32 - ReLu - MLP-64 - Relu - 2 Softmax")

        # compute accuracy on the test set
        predictions = model.transform(testDF_VA)


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        # Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Deep Neural Network

# COMMAND ----------

if run_model_flag:
    with mlflow.start_run(run_name='flights_NN') as run:

        # specify layers for the neural network:
        # input layer of size 32 (features), four intermediate of size 128, 252, 128, and 64
        # and output of size 2 (classes) 

        # MLP-32 - MLP-128 - Relu - MLP-252 - Relu - MLP-128 - Relu - 2 Softmax 
        num_input_features = len(trainDF_VA.select(STANDARD_FEATURE_COLNAME).take(1)[0][STANDARD_FEATURE_COLNAME].toArray())
        layers = [num_input_features, 128, 252, 128, 64, 2]

        # create the trainer and set its parameters
        trainer = MultilayerPerceptronClassifier(maxIter=100, 
                                                layers=layers, 
                                                blockSize=512, 
                                                seed=523,
                                                featuresCol=STANDARD_FEATURE_COLNAME, 
                                                labelCol="DEP_DEL15")

        # train the model
        model = trainer.fit(trainDF_VA)

            
        # log model input parameters to MLFlow
        mlflow.log_param("dataset_name", dataset_name)
        mlflow.log_param("featuresCol_1", numeric_features)
        mlflow.log_param("featuresCol_2", onehot_features)
        mlflow.log_param("labelCol", "DEP_DEL15")
        mlflow.log_param("family", "NN: MLP-32 - Relu - MLP-128 - Relu - MLP-252 - Relu - MLP-128 - Relu - 2 Softmax")

        # compute accuracy on the test set
        predictions = model.transform(testDF_VA)


        # Select (prediction, true label) and compute test error
        evaluatorMulti = MulticlassClassificationEvaluator(labelCol="DEP_DEL15", 
                                                        predictionCol="prediction")
        evaluator = BinaryClassificationEvaluator(labelCol="DEP_DEL15", 
                                                rawPredictionCol="prediction", 
                                                metricName='areaUnderROC')

        # Metrics
        accuracy = evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "accuracy"})
        precision = evaluatorMulti.evaluate(predictions, 
                                            {evaluator.metricName: "weightedPrecision"})
        recall =  evaluatorMulti.evaluate(predictions, 
                                        {evaluator.metricName: "weightedRecall"})
        AUC =  evaluator.evaluate(predictions)

        # Log test metrics to MLFlow
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_AUC", AUC)

        print("###########################Testing Metric###########################")
        print(f"Model Test Accuracy: {np.round(accuracy*100, 3)}")
        print(f"Model Test Precision: {np.round(precision*100, 3)}")
        print(f"Model Test Recall: {np.round(recall*100, 3)}")
        print(f"Model Test AUC: {np.round(AUC, 3)}")


# COMMAND ----------

# MAGIC %md 
# MAGIC #Leakage
# MAGIC
# MAGIC In machine learning, data leakage is an error where information about a target variable is inadvertently included in a model's training. This can lead to artificially inflated performance metrics during training, potentially causing the model to perform poorly on new, unseen data. For example, when training a model to predict flight delays, if both “scheduled departure time” and “actual departure time” were included, the model would likely give high importance to actual departure time as a predictor of delays. However, it would not perform well in predicting current or future flights, where “actual departure time” is not available. It learned patterns specific to training that were not generalizable. To avoid data leakage, it's important to only include features available at the time of prediction and avoid incorporating any information that leaks the predicted outcome.
# MAGIC
# MAGIC In our experiments, we initially tried to implement a feature which would flag if there were existing or recent delays in the airport. While this feature would greatly assist in predicting if a flight was delayed, we decided against including it for data leakage concerns because that information would clearly not be available for future flights.
# MAGIC
# MAGIC For our workflow, we have implemented the following measures to avoid the “cardinal sins” of machine learning:
# MAGIC
# MAGIC -	Careful feature selection and feature engineering to avoid leaks.
# MAGIC
# MAGIC -	Screening our model results for performance outliers that might indicate leakage. 
# MAGIC
# MAGIC -	Our dataset contains millions of samples, which easily meet any statistical requirements for significance. 
# MAGIC
# MAGIC -	Model training used properly divided training and validation datasets. 
# MAGIC
# MAGIC -	Final model evaluation based on previously unseen data from an unseen year.
# MAGIC
# MAGIC -	Using AUC as the primary performance metric
# MAGIC
# MAGIC   - More robust compared to accurancy against class imbalances, which is the case here where most flights are not delayed versus delayed.
# MAGIC
# MAGIC   - Suitability for ranking predictions, where the severity of the flight delay is also a factor worth considering.
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC # Final Prediction Pipeline

# COMMAND ----------

# Final Prediction Pipeline
run_pred_pipeline = False
if new_data_flag and run_pred_pipeline:

    # From ML Flow Logged, fetch the logged model on the 60M dataset
    import mlflow
    logged_model = 'runs:/bbb0776845fb48d5a59a77a9b13a329a/flights_logistic_regression_model'

    # Load model
    loaded_model = mlflow.spark.load_model(logged_model)

    # Perform inference via model.transform()
    predictions = loaded_model.transform(new_data)

    # show 10 predictions
    predictions.limit(10).display()


# COMMAND ----------

# MAGIC %md
# MAGIC # Results
# MAGIC Our best performing model for each of the data sets was binomial logistic regression model. Here are the results for each of the different datasets, as gathered from MLFlow. As our dataset is highly imbalanced we decided that accuracy was not the best metric to use. Rather to optimize the model using Test AUC, as it is an aggregated measure of performance across all classifications. 
# MAGIC
# MAGIC #### Table 3: Best Performance Metric
# MAGIC | Model | Dataset | Test AUC | Test Accuracy | Test Precision | Test Recall |
# MAGIC | ----- | ------- | -------- | ------------- | -------------- | ----------- |
# MAGIC | majority class baseline      | 3 Months  | 0.5         |	0.712909   |	0.713183	 |0.711072     |
# MAGIC | binomial logistic regression | 3 Months (original) | 0.697641218 | 0.802630851 | 0.760872969 | 0.802630851 |
# MAGIC | binomial logistic regression | 3 Months (resampled)    | 0.697641 | 0.802631      | 0.760873       | 0.802631    |
# MAGIC | binomial logistic regression | 12 Months (original) | 0.635524129 | 0.802888248 | 0.737436417 | 0.802888248 |
# MAGIC | binomial logistic regression | 60 Months (original) | 0.6562238 | 0.817303234 | 0.768425354 | 0.817303234 |
# MAGIC | binomial logistic regression | 60 Months (resampled) | 0.656224 | 0.659482822 | 0.760371753 | 0.659482822 |
# MAGIC
# MAGIC

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Experiments Outputs

# COMMAND ----------

experiments = spark.read.format('csv')\
                    .option('inferSchema', True)\
                    .option('header', True)\
                    .load(f"{team_blob_url}/experiments/Phase4_experiments.csv")
experiments.select('Name', 'Training Time (s)', 'dataset_name', 'family', 'featuresCol_1', 'featuresCol_2', 'test_AUC', 'test_accuracy', 'test_precision', 'test_recall').display()

# COMMAND ----------

#neural network time vs logistic regression
experiments_df['test_AUC_to_training_time_ratio'] = experiments_df['test_AUC']/experiments_df['Training Time (s)']
# experiments_sliced = experiments_df[experiments_df['Name'].isin(['flights_logistic_regression', 'flights_NN'])]
experiments_df = experiments_df.sort_values(['dataset_name','test_AUC_to_training_time_ratio'])
experiments_df = experiments_df[experiments_df['dataset_name'] == 'OTPW_60M']
experiments_df = experiments_df.groupby(['Name']).mean().sort_values('Training Time (s)')
experiments_df

# COMMAND ----------

experiments_df = experiments.toPandas()

# training time vs AUC
plt.clf()
# plt.figure(figsize=(12,6))
fig, ax = plt.subplots(1, 2, figsize=(16,6))

# ax2 = ax1.twinx()

ax[0].bar(x=experiments_df.index, height=experiments_df['test_AUC'], color='tab:blue')
ax[1].bar(x=experiments_df.index, height=experiments_df['Training Time (s)'], color='tab:red')

# plt.legend(loc="upper right")
# plt.title("Training Time vs Test AUC")
ax[0].set_title('Test AUC')
ax[1].set_title('Training Time (s)')
ax[0].set_ylabel('Test AUC')
ax[1].set_ylabel('Training Time (s)')
ax[0].tick_params(axis='x', labelrotation =90)
ax[1].tick_params(axis='x', labelrotation =90)
plt.show()


# COMMAND ----------

# training time vs AUC
plt.clf()
plt.figure(figsize=(12,6))
sns.scatterplot(x=experiments_df['Training Time (s)'], y=experiments_df['test_AUC'], hue=experiments_df['Name'])
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.title('Training Time vs Test AUC')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC # Discussion of Results
# MAGIC
# MAGIC ### Gap Analysis
# MAGIC
# MAGIC #### Expectation of Performance
# MAGIC
# MAGIC The majority class of our dataset is approximately 80% of the data. Therefore, we would expect that a model that always choses the majority class would have an accuracy of approximately 80%. However, this wouldn't provide much value to our use case of helping passengers predict if their flights are going to be delayed. Inaccurate predictions of each class have similar consequences to the passengers. A customer that thinks their flight is going to be delayed when it isn't, is in jeopardy of missing their flight. In contrast a passenger that doesn't know their flight is going to be delayed may be left waiting for hours unnecessarily.
# MAGIC
# MAGIC #### Comparision of Results
# MAGIC
# MAGIC According to Science Direct, "In general, an AUC of 0.5 suggests no discrimination, 0.7 to 0.8 is considered acceptable, 0.8 to 0.9 is considered excellent, and more than 0.9 is considered outstanding." Using their thresholds, the pipeline that we have developed is slightly below the 0.7 threshold for acceptability. Meaning, that our modeling pipeline fails to give satisfy the our use case.
# MAGIC
# MAGIC #### Possible Reasons
# MAGIC
# MAGIC The reason for our model failing to achieve higher predictability can be attributed to many factors. One such reason for this is that not all of the underlying factors for delays not represented in the data, such as
# MAGIC   - Natural disasters other than weather
# MAGIC   - Aircraft maintance problems
# MAGIC   - Employee Absenses 
# MAGIC   - and many others
# MAGIC
# MAGIC While additional data could help to boost model performance, there are also further experimentation work that we believe could help be used to improve our model too:
# MAGIC   - Further hyperparameter tuning
# MAGIC   - Addition of airport historic delay rate
# MAGIC   - Feature importance analysis 
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC # Conclusion
# MAGIC
# MAGIC In conclusion, our highest performing model based on AUC is binomial logistic regression. The performance of the model approaches the acceptable AUC threshold of 0.7, falling just short at 0.66 AUC. Throughout our experiments, comparing the results across larger and larger datasets helped us find instances of overfitting and data leakage. For example, our initial binomial logistic regression model met the 0.7 AUC threshold for the 3 month dataset, but fell short in the larger 12 month and 60 month datasets, forcing us to review, rethink, and refine the features that went into the model. We will focus on exploring new features and models to uncover additional insights and potential improvements. The process will involve continuous hyperparameter tuning to optimize model performance and ensure its accuracy and reliability. Ultimately, we aim to consolidate all our efforts into the final deliverables, presenting a robust and effective solution to the problem at hand. By adhering to these steps, we are confident in creating a high-quality and impactful model for our project.
