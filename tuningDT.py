#!/usr/bin/env python
# coding: utf-8

# # Project: Fraud Detection 

# ## 1. Overview

# ### PaySim simulates mobile money transactions based on a sample of real transacions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world. The objective of the project is to predict if a transaction is fraudulent or not.

# ## 2. Preprocess the data

#### Libraries

# libraries: mathematical computing
import numpy as np
import pandas as pd

# libraries: sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc

# libraries: pyspark SparkContext
from pyspark import SparkContext, SparkConf

# libraries: pyspark sql
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from  pyspark.sql.functions import monotonically_increasing_id, desc, row_number

# libraries: pyspark machine learning
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.mllib.stat import Statistics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel, TrainValidationSplit
from pyspark.ml.feature import HashingTF, Tokenizer, RFormula
from pyspark.ml.regression import RandomForestRegressor


# libraries: visualization
import seaborn as sb
import matplotlib.pyplot as mpt
import functools
from collections import Counter


# create the SparkSession 

spark = SparkSession.builder.appName("TuningLR1").getOrCreate()

# Creating a Spark-Context

sc = SparkContext.getOrCreate()

# read the parquet file

df_bank_par = spark.read.parquet('df.parquet')

# convert string columns into integer columns

df_bank_par = df_bank_par.withColumn("isFraud",df_bank_par["isFraud"].cast(IntegerType()))


#### Construction of models

### train/test split

train,test = df_bank_par.randomSplit([0.7,0.3])


### Let´s assemble these datasets "train" and "test" into a single feature vector using VectorAssembler class per each one.
### Let´s assemble the train dataset as a single feature vector using VectorAssembler class

columns = ['step','amount','type_CASH_OUT','type_PAYMENT','type_CASH_IN','type_TRANSFER','type_DEBIT','type2_CC','type2_CM','isFraud']

assembler = VectorAssembler(inputCols=columns, outputCol='features')

train = assembler.transform(train).withColumnRenamed("features", "my_features")

print("The train dataset is:")

train.show(10)


# let´s assemble the test dataset as a single feature vector using VectorAssembler class

columns = ['step','amount','type_CASH_OUT','type_PAYMENT','type_CASH_IN','type_TRANSFER','type_DEBIT','type2_CC','type2_CM','isFraud']

assembler = VectorAssembler(inputCols=columns, outputCol='features')

test = assembler.transform(test).withColumnRenamed("features", "my_features")

print("The test dataset is:")

test.show(10)


##### Model: Random Forest

# Hyperparameters Tunning: Train Validation Split

dt = DecisionTreeClassifier(labelCol='isFraud',featuresCol='my_features')

stages = [assembler,dt]

pipeline = Pipeline().setStages(stages) 

params = ParamGridBuilder()\
            .addGrid(dt.impurity,["gini","entropy"])\
            .build()
            #.addGrid(dt.maxDepth, [int(x) for x in np.linspace(start=0, stop=30, num=3)])\
            
evaluator = BinaryClassificationEvaluator()\
            .setMetricName("areaUnderROC")\
            .setRawPredictionCol("prediction")\
            .setLabelCol("isFraud")

crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=params,
                          evaluator=evaluator,
                          numFolds=3
                         )

print("The tuning of the hyperparameters has began!")

cvModel = crossval.fit(train)

print("The tuning of the hyperparameters has finished!")

bestPipeline = cvModel.bestModel
bestModel = bestPipeline.stages[1] # type: ignore

# save the model
cvModel.save("tuningDTProof")
print("The process of tuning has finished!")