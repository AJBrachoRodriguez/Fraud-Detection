#!/usr/bin/env python
# coding: utf-8

# # Project: Fraud Detection 

# ## 1. Overview

# ### PaySim simulates mobile money transactions based on a sample of real transacions extracted from one month of financial logs from a mobile money service implemented in an African country. The original logs were provided by a multinational company, who is the provider of the mobile financial service which is currently running in more than 14 countries all around the world. The objective of the project is to predict if a transaction is fraudulent or not.

# ## 2. Preprocess the data

# ### Libraries

# In[ ]:


# libraries: mathematical computing
import numpy as np
import pandas as pd

# libraries: sklearn
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

# libraries: pyspark SparkContext
from pyspark import SparkContext, SparkConf

# libraries: pyspark sql
from pyspark.sql.types import IntegerType, FloatType
from pyspark.sql.window import Window
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from  pyspark.sql.functions import monotonically_increasing_id, desc, row_number
from pyspark.sql import SQLContext

# libraries: pyspark machine learning
from pyspark.ml import Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import RandomForestRegressor, LinearRegression
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression, DecisionTreeClassifier, NaiveBayes, RandomForestClassificationModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.evaluation import MulticlassMetrics, BinaryClassificationMetrics
from pyspark.mllib.stat import Statistics
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel, TrainValidationSplit,TrainValidationSplitModel
from pyspark.ml.feature import HashingTF, Tokenizer, RFormula
from pyspark.ml.regression import RandomForestRegressor

# libraries: visualization
import seaborn as sb
import matplotlib.pyplot as mpt
import functools
from collections import Counter
from ydata_profiling import ProfileReport


# In[ ]:


# Creating a Spark-Context

sc = SparkContext.getOrCreate()


# In[ ]:


# Spark Builder

spark = SparkSession.builder.appName("fraudDetection").getOrCreate()


# In[ ]:


spark


# In[ ]:


# global variables

global df_bank, results 


# #### We´ll use PySpark to preprocess the data.

# In[ ]:


# spark dataframe 

df = spark.read.csv('fraudDetection.csv', header=True)


# In[ ]:


spark.conf.set("spark.sql.execution.arrow.enabled","true")


# #### Now, we´ll convert this "df" dataframe into a parquet file using the following method of pyspark. The file will be named "fraudDetection.parquet"

# In[ ]:


df.write.parquet("fraudDetection.parquet")


# #### Now, we´ll read the file as a parquet file. The calculation will be faster.

# In[ ]:


df_bank_par = spark.read.parquet("fraudDetection.parquet")


# #### Let´s take a look to the data with the first 10 rows.

# In[ ]:


df_bank_par.show(10)


# In[ ]:


df_bank_par.printSchema()


# #### There are 11 columns, none of them is numerical (they are categorical). Let´s count the number of registers.

# In[ ]:


print(f"The total number of registers is:",df_bank_par.count())


# #### We have more than six miliions of transactions in the dataset.

# ### 2.1 Feature Engineering

# #### Firstly, we´ll create a function to create a new variable.

# In[ ]:


### 2.1.1.- creation of a new variable: type2

df_type2 = df_bank_par.withColumn("type2",f.concat(f.substring("nameOrig",1,1),f.substring("nameDest",1,1)))


# In[ ]:


df_type2.show(5)


# #### We´ve created a new column named "type2" which is composed by the first character of the column "nameOrig" and the first character of the column "nameDest"

# In[ ]:


### 2.1.2.1.- One Hot Encoding: column "type"

df_type2.show(3)


# #### We´ll use some libraries of Spark for Machine Learning (SparkML).

# In[ ]:


### StringIndexer Initialization
### column: type

indexer_type = StringIndexer(inputCol="type",outputCol="types_indexed")
indexerModel_type = indexer_type.fit(df_type2)


# In[ ]:


### Transform the DataFrame using the fitted StringIndexer model

indexed_df_type2 = indexerModel_type.transform(df_type2)
indexed_df_type2.show(10)


# #### Here, we´ve set each of the elements of the "type" column into indexes.

# In[ ]:


### apply One-Hot-Encoding to the indexed column, that is, 
### "types_indexed"

encoder_type = OneHotEncoder(dropLast=False, inputCol="types_indexed", outputCol="types_onehot")
encoder_type_df = encoder_type.fit(indexed_df_type2).transform(indexed_df_type2)
encoder_type_df.show(truncate=False)


# In[ ]:


encoder_type_df.printSchema()


# In[ ]:


encoder_type_df_split = encoder_type_df.select('*',vector_to_array('types_onehot').alias('types_onehot_split'))
encoder_type_df_split.show(5)


# In[ ]:


### now, we´ll split the "types_onehot_split" into five columns, one per category

num_categories = len(encoder_type_df_split.first()['types_onehot_split'])
cols_expanded = [(f.col('types_onehot_split')[i].alias(f"{indexerModel_type.labels[i]}")) for i in range(num_categories)]
type_df = encoder_type_df_split.select('*',*cols_expanded)


# In[ ]:


type_df.show(100)


# #### We´ve applied One-Hot-Encoding to the column "type" resulting in five new columns:
# + CASH_OUT
# + CASH_IN
# + PAYMENT
# + TRANSFER 
# + DEBIT

# #### Now, we´ll apply this procedure to the column "type2".

# In[ ]:


### 2.1.2.2.- One Hot Encoding: column "type2"

type_df.show(5)


# In[ ]:


### StringIndexer Initialization
### column: type2

indexer_type = StringIndexer(inputCol="type2",outputCol="types_indexed2")
indexerModel_type = indexer_type.fit(type_df)


# In[ ]:


### Transform the DataFrame using the fitted StringIndexer model

indexed_df_type = indexerModel_type.transform(type_df)
indexed_df_type.show(10)


# In[ ]:


### apply One-Hot-Encoding to the indexed column, that is, 
### "types_indexed2"

encoder_type2 = OneHotEncoder(dropLast=False, inputCol="types_indexed2", outputCol="types_onehot2")
encoder_type2_df = encoder_type2.fit(indexed_df_type).transform(indexed_df_type)
encoder_type2_df.show(truncate=False)


# In[ ]:


encoder_type2_df.printSchema()


# In[ ]:


encoder_type2_df_split = encoder_type2_df.select('*',vector_to_array('types_onehot2').alias('types_onehot_split2'))
encoder_type2_df_split.show(5)


# In[ ]:


### now, we´ll split the "types_onehot_split2" into two columns, one per category

num_categories = len(encoder_type2_df_split.first()['types_onehot_split2'])
cols_expanded = [(f.col('types_onehot_split2')[i].alias(f"{indexerModel_type.labels[i]}")) for i in range(num_categories)]
encoder_type2_df_split = encoder_type2_df_split.select('*',*cols_expanded)


# In[ ]:


encoder_type2_df_split.show(5)


# #### We´ve split the "type2" column into two columns based on One-Hot-Encoding. Now, we´ll eliminate some unnecessaruy columns. Let´s check out all the columns.

# In[ ]:


encoder_type2_df_split.printSchema()


# #### Now, we´ll eliminate the unnecessary columns:
# + nameOrig
# + nameDest
# + isFlaggedFraud
# + newbalanceDest
# + oldbalanceDest
# + oldbalanceOrg
# + newbalanceOrig 
# + types_indexed
# + types_onehot
# + types_onehot_split
# + types_indexed2
# + types_onehot2
# + types_onehot_split2
# + type
# + type2

# In[ ]:


df_bank_par = encoder_type2_df_split.drop("nameOrig","nameDest","isFlaggedFraud","newbalanceDest","oldbalanceDest",
                       "oldbalanceOrg","newbalanceOrig","type","types_indexed","types_onehot",
                       "types_onehot_split","type2","types_indexed2","types_onehot2","types_onehot_split2","types_indexed3","types_onehot3","types_onehot_split3" )
df_bank_par.show(5)


# In[ ]:


df_bank_par.columns


# In[ ]:


df_bank_par.count()


# In[ ]:


type(df_bank_par)


# #### We can see that there are the same quantity of registers.

# ### 2.2 Data Cleaning

# In[ ]:


### 2.2.1.- Eliminate duplicated

num_all_rows = df_bank_par.count()
num_all_rows


# In[ ]:


num_duplicated_rows = df_bank_par.distinct().count() 


# In[ ]:


print(f"The total number of duplicated rows is:",num_all_rows - num_duplicated_rows)


# #### We can see that there are 7597 duplicated rows. Let´s remove the null values and duplicated values from the df_bank_par dataframe.

# In[ ]:


df_bank_par = df_bank_par.dropna()

df_bank_par = df_bank_par.dropDuplicates()


# In[ ]:


df_bank_par.count()


# #### We can see the duplicated registers have been removed because there are fewer registers than before. Let´s take a look at the "clean" dataset.

# In[ ]:


df_bank_par.show(10)


# ## 3. Exploratory Data Analysis (EDA)

# ### 3.1 Visualization

# #### The visualization will be done using a functions which leverages the method histogram() of pyspark. 

# In[ ]:


# definition of the "histogram" function

def histogram(df, col, bins=10, xname=None, yname=None):
    
    '''
    This function makes a histogram from spark dataframe named 
    df for column name col. 
    '''
    
    # Calculating histogram in Spark 
    vals = df.select(col).rdd.flatMap(lambda x: x).histogram(bins)
    
    # Preprocessing histogram points and locations 
    width = vals[0][1] - vals[0][0]
    loc = [vals[0][0] + (i+1) * width for i in range(len(vals[1]))]
    
    # Making a bar plot 
    mpt.bar(loc, vals[1], width=width)
    mpt.xlabel(col)
    mpt.ylabel(yname)
    mpt.show()


# #### There are some features that need to be converted to integers such as "step","amount" and "isFraud".

# In[ ]:


# convert string columns into integer columns

df_bank_par = df_bank_par.withColumn("step",df_bank_par["step"].cast(IntegerType()))


# In[ ]:


df_bank_par = df_bank_par.withColumn("amount",df_bank_par["amount"].cast(IntegerType()))


# In[ ]:


df_bank_par = df_bank_par.withColumn("isFraud",df_bank_par["isFraud"].cast(IntegerType()))


# In[ ]:


df_bank_par.printSchema()


# In[ ]:


df_bank_par.show(5)


# #### We´ve seen that all the features are "integer" types now. Therefore, we´re able to perform various visualizations with the histogram method. That´s what we´ll do next.

# In[ ]:


# histogram: "step"

histogram(df_bank_par, 'step', bins=15, yname='frequency')


# In[ ]:


# histogram: "amount"

histogram(df_bank_par, 'amount', bins=15, yname='frequency')


# In[ ]:


# histogram: "Debit"

histogram(df_bank_par, 'Debit', bins=15, yname='frequency')


# In[ ]:


# histogram: "Payment"

histogram(df_bank_par, 'Payment', bins=15, yname='frequency')


# In[ ]:


# histogram: "CASH_OUT"

histogram(df_bank_par, 'CASH_OUT', bins=15, yname='frequency')


# In[ ]:


# histogram: "CASH_IN"

histogram(df_bank_par, 'CASH_IN', bins=15, yname='frequency')


# In[ ]:


# histogram: "TRANSFER"

histogram(df_bank_par, 'TRANSFER', bins=15, yname='frequency')


# In[ ]:


# histogram: "CC"

histogram(df_bank_par, 'CC', bins=15, yname='frequency')


# In[ ]:


# histogram: "CM"

histogram(df_bank_par, 'CM', bins=15, yname='frequency')


# In[ ]:


# histogram: "isFraud"

histogram(df_bank_par, 'isFraud', bins=15, yname='frequency')


# #### Remember that our label is "isFraud", therefore, we can see that this class is unbalanced as we can see from the previous graphic. We need to perform an **Oversampling** through ***Data Balancing***.

# ### 3.2 Data Balancing

# #### In this case, there are two options to do the ***Data Balancing***: with **PySpark** and with **Sklearn**. You can uncomment the pyspark section, otherwise it will be done with Sklearn. 

# In[ ]:


################################################################ Oversampling with PySpark ##############################################################

# Create undersampling function
#def oversample_minority(df, ratio=1):
#    '''
#    ratio is the ratio of majority to minority
#    Eg. ratio 1 is equivalent to majority:minority = 1:1
#    ratio 5 is equivalent to majority:minority = 5:1
#    '''
#    minority_count = df.filter(f.col('isFraud')==1).count()
#    majority_count = df.filter(f.col('isFraud')==0).count()
#    
#    balance_ratio = majority_count / minority_count
#    
#    print(f"Initial Majority:Minority ratio is {balance_ratio:.2f}:1")
#    if ratio >= balance_ratio:
#        print("No oversampling of minority was done as the input ratio was more than or equal to the initial ratio.")
#    else:
#        print(f"Oversampling of minority done such that Majority:Minority ratio is {ratio}:1")
#    
#    oversampled_minority = df.filter(f.col('isFraud')==1).sample(withReplacement=True, fraction=(balance_ratio/ratio),seed=88)
#    oversampled_df = df.filter(f.col('isFraud')==0).union(oversampled_minority)
#    
#    return oversampled_df

#oversampled_df = oversample_minority(df_bank_par,ratio=1)

#minority_count = oversampled_df.filter(f.col('isFraud')==1).count()
#majority_count = oversampled_df.filter(f.col('isFraud')==0).count()
#minority_count, majority_count
#oversampled_df = oversampled_df.dropna()
#oversampled_df = oversampled_df.dropDuplicates()
#df_bank_par = oversampled_df


# In[ ]:


################################################################# Oversampling with Sklearn  #############################################################

df_banco = pd.read_csv("fraudDetection.csv")


# In[ ]:


#@title
def procesar_datos():
  global df_banco, resultados
  df_banco=df_banco.copy()
  # Crea la nueva variable type2 con la combinación de la primera letra de las columnas nameOrig y nameDest
  df_banco['type2'] = df_banco['nameOrig'].str[0] + df_banco['nameDest'].str[0]


# In[ ]:


procesar_datos()
df_banco.head()


# In[ ]:


# one-hot encoding: "type" and "type2"
df_encoded = pd.get_dummies(df_banco, columns=['type', 'type2'], dtype=int)
df_encoded.sample(4)


# In[ ]:


df_encoded.info()


# In[ ]:


# columns to eliminate
columns_to_drop = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# eliminate those columns
df_encoded.drop(columns=columns_to_drop, inplace=True)

# reset the index
df_encoded.reset_index(drop=True, inplace=True)
df_encoded


# In[ ]:


# eliminate the duplicated registers and save the result in df_banco

df_banco = df_encoded.drop_duplicates()


# In[ ]:


# eliminate the duplicated registers and reset the index in df_banco
df_banco.dropna(inplace=True)
df_banco.reset_index(drop=True, inplace=True)


# In[ ]:


df_banco.head()


# In[ ]:


df_banco.info()


# In[ ]:


# count the values in the column "isFraud"
conteo_isfraud = df_banco['isFraud'].value_counts()

# create the "bar" graphic 
mpt.figure(figsize=(8, 6))
conteo_isfraud.plot(kind='bar', color=['skyblue', 'salmon'])
mpt.title('Distribution of the column "isFraud" ')
mpt.xlabel('isFraud')
mpt.ylabel('quantity')
mpt.xticks([0, 1], ['NoFraud', 'Fraud'], rotation=0)

# aggregate the values over the bar
for i, valor in enumerate(conteo_isfraud):
    mpt.text(i, valor + 0.01 * max(conteo_isfraud), str(valor), ha='center', va='bottom')

mpt.show()


# In[ ]:


def boxplot_histogramas():

  # define the attributes to plot
  attributes = ['isFraud', 'amount', 'step']  # Puedes agregar más attributes aquí si lo deseas

  # generate the boxplot graphic
  mpt.figure(figsize=(12, 6))
  for i, attribute in enumerate(attributes, 1):
      mpt.subplot(1, len(attributes), i)
      sb.boxplot(x=df_banco[attribute])
      mpt.title(f'Boxplot de {attribute}')
  mpt.tight_layout()
  mpt.show()

  # generate histograms
  mpt.figure(figsize=(12, 6))
  for i, attribute in enumerate(attributes, 1):
      mpt.subplot(1, len(attributes), i)
      sb.histplot(df_banco[attribute], kde=True)
      mpt.title(f'Histograma de {attribute}')
  mpt.tight_layout()
  mpt.show()
boxplot_histogramas()


# In[ ]:


# create the histogram of the column 'amount'
mpt.figure(figsize=(10, 6))
mpt.hist(df_banco['amount'], bins=140, color='skyblue', edgecolor='black')
mpt.title('Histogram of the column "amount" ')
mpt.xlabel('Amount')
mpt.ylabel('Frequency')
mpt.grid(True)
mpt.show()


# In[ ]:


ProfileReport(df_banco)


# In[ ]:


#@title

def balancing_class():
    global df_banco, resultados

    # Instancia SMOTE
    smote = SMOTE(random_state=42)

    # Balanceo de clases
    X_res, y_res = smote.fit_resample(df_banco.drop(columns=['isFraud']), df_banco['isFraud'])

    # Reconstrucción del DataFrame balanceado
    df_banco = pd.DataFrame(X_res, columns=df_banco.drop(columns=['isFraud']).columns)
    df_banco['isFraud'] = y_res

    # Elimina registros duplicados
    df_banco.drop_duplicates(inplace=True)
    df_banco.reset_index(drop=True, inplace=True)

# Llama a la función balanceo_clases
balancing_class()

# Imprime el resultado final
df_banco


# In[ ]:


# count the values of the column "isFraud"
count_isfraud = df_banco['isFraud'].value_counts()

# create the graphic
mpt.figure(figsize=(8, 6))
count_isfraud.plot(kind='bar', color=['skyblue', 'salmon'])
mpt.title('Distribution of the column isFraud')
mpt.xlabel('isFraud')
mpt.ylabel('Quantity')
mpt.xticks([0, 1], ['NoFraud', 'Fraud'], rotation=0)
mpt.show()


# In[ ]:


balancing_class()
ProfileReport(df_banco)


# #### Now, we´ll convert this pandas dataframe into a PySpark dataframe to leverage, but first to a parquet file.

# In[ ]:


df_banco.to_parquet('df.parquet')


# #### Now, we´ll convert read the parquet file into a PySpark dataframe to make the calculations faster. 

# In[ ]:


df_bank_par = spark.read.parquet('df.parquet')


# In[ ]:


df_bank_par.count()


# In[ ]:


df_bank_par.printSchema()


# In[ ]:


# convert string columns into integer columns

df_bank_par = df_bank_par.withColumn("isFraud",df_bank_par["isFraud"].cast(IntegerType()))


# In[ ]:


df_bank_par.printSchema()


# In[ ]:


class_0 = df_bank_par.filter(f.col("isFraud")==0)
class_1 = df_bank_par.filter(f.col("isFraud")==1)


# In[ ]:


class_0.count()


# In[ ]:


class_1.count()


# In[ ]:


######################################################## Convert parquet file into Pandas ###########################################
#############################################################  (optional section)   #################################################

##df_bank_par_pandas = df_bank_par.to_pandas_on_spark()
##df_bank_par_pandas.head(10)
##df_bank_par_pandas.describe()
##type(df_bank_par_pandas)


# #### Let´s create a function to find a correlation between the target variable "isFraud" and the features. 

# In[ ]:


# definition of the function "correlation_df"

def correlation_df(df,target_var,feature_cols, method):
    # assemble features into a vector
    target_var = [target_var]
    feature_cols = feature_cols
    df_cor = df.select(target_var + feature_cols)
    assembler = VectorAssembler(inputCols=target_var + feature_cols, outputCol="features")
    df_cor = assembler.transform(df_cor)

    # calculate correlation matrix
    correlation_matrix = Correlation.corr(df_cor, "features", method =method).head()[0]

    # extract the correlation coefficient between target and each feature
    target_corr_list = [correlation_matrix[i,0] for i in range(len(feature_cols)+1)][1:]

    # create a Dataframe with target variable, feature names and correlation coefficients
    correlation_data = [(feature_cols[i],float(target_corr_list[i])) for i in range(len(feature_cols))]

    correlation_df = spark.createDataFrame(correlation_data, ["feature","correlation"] )

    correlation_df = correlation_df.withColumn("abs_correlation",f.abs("correlation"))

    # print the result
    return correlation_df


# #### Now, let´s calculate the correlation among them.

# In[ ]:


target = "isFraud"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "amount"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "step"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type_CASH_IN"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type_CASH_OUT"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type_DEBIT"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type_PAYMENT"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type_TRANSFER"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type2_CC"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# In[ ]:


target = "type2_CM"

indep_cols = [x for x in df_bank_par.columns if x not in [target] ]

corr_values_df = correlation_df(df=df_bank_par, target_var= target, feature_cols= indep_cols, method='pearson')

print(f"The corelation between {target} and the other features is: ")

corr_values_df.show()


# ## 4. Construction of models

# ## 4.1 train/test split

# In[ ]:


train,test = df_bank_par.randomSplit([0.7,0.3])


# #### Let´s assemble these datasets "train" and "test" into a single feature vector using VectorAssembler class per each one.

# In[ ]:


# let´s assemble the train dataset as a single feature vector using VectorAssembler class

columns = ['step','amount','type_CASH_OUT','type_PAYMENT','type_CASH_IN','type_TRANSFER','type_DEBIT','type2_CC','type2_CM','isFraud']

assembler = VectorAssembler(inputCols=columns, outputCol='features')

train = assembler.transform(train).withColumnRenamed("features", "my_features")

train.show(10)


# In[ ]:


train.count()


# In[ ]:


# let´s assemble the test dataset as a single feature vector using VectorAssembler class

columns = ['step','amount','type_CASH_OUT','type_PAYMENT','type_CASH_IN','type_TRANSFER','type_DEBIT','type2_CC','type2_CM','isFraud']

assembler = VectorAssembler(inputCols=columns, outputCol='features')

test = assembler.transform(test).withColumnRenamed("features", "my_features")

test.show(10)


# In[ ]:


test.count()


# ## 4.2 Models
# 
# We´ll use several machine learning algorithms to evaluate all of them and to select the best one. We´ll start with Random Forest. However, it´s important to create some lists where to store the results of the models:

# In[ ]:


name_model = []

accuracy = []

precision = []

recall = []

auc_roc = []


# ### 4.2.1 Random Forest

# #### Training

# In[ ]:


####################################################### model from the cross validation ##############################################
################################################### (uncomment if you want to use this model ) #######################################

#modelRF = CrossValidatorModel.load("tuningRF")
#print(modelRF.explainParams())
#type(modelRF)
#modelRF.extractParamMap()


# In[ ]:


rf = RandomForestClassifier(labelCol="isFraud", featuresCol="my_features")


# In[ ]:


print(rf.explainParams())


# In[ ]:


modelRF = rf.fit(train)


# In[ ]:


predictions = modelRF.transform(test)


# In[ ]:


# evaluation

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)


# #### Hyperparameters tuning

# ##### In the "values list" you can change the values range of the hyperparameters to test with. Besides, you need to set the hyperparameter name in the argument of the "RandomForestClassifier" model. 

# In[ ]:


# visualization of the overfitting

# define the tree depths to evaluate
values = [i for i in range(0, 11)]
train_scores = []
test_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 rf = RandomForestClassifier(labelCol="isFraud", featuresCol="my_features", maxDepth=i)
 # configure the evaluators and metrics
 evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")
 metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")
 # fit model on the training dataset
 modelRF = rf.fit(train)
 # predictions on the training dataset
 predictions = modelRF.transform(train)
 # evaluate on the train dataset
 train_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 train_scores.append(train_accuracy)
 # evaluate on the test dataset
 predictions = modelRF.transform(test)
 test_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 test_scores.append(test_accuracy)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_accuracy, test_accuracy))


# In[ ]:


# plot of train and test scores vs "<hyperparameter>"

mpt.plot(values, train_scores, '-o', label='Train')
mpt.plot(values, test_scores, '-o', label='Test')
mpt.legend()
mpt.show()


# In[ ]:


# evaluate in the training dataset

predictions = modelRF.transform(train)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)

print(f"Precsion: ", precision_rf)

print(f"Recall: ", recall_rf)


# ROC curve : visualization

print("The ROC curve is:")
trainingSummary = modelRF.summary
lrROC = trainingSummary.roc.toPandas()

mpt.plot(lrROC['FPR'],lrROC['TPR'])
mpt.ylabel('False Positive Rate')
mpt.xlabel('True Positive Rate')
mpt.title('ROC Curve - Random Forest')
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
mpt.show()


# Precision Recall curve : visualization

print("The Precision Recall curve is:")
pr = trainingSummary.pr.toPandas()
mpt.plot(pr['recall'],pr['precision'])
mpt.ylabel('Precision')
mpt.xlabel('Recall')
mpt.title('Precision Recall curve - Random Forest')
mpt.show()


# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Random Forest')
#mpt.show()


# In[ ]:


# evaluate in the test dataset

predictions = modelRF.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)

print(f"Precsion: ", precision_rf)

print(f"Recall: ", recall_rf)

# ROC curve : visualization

trainingSummary = modelRF.summary
lrROC = trainingSummary.roc.toPandas()

mpt.plot(lrROC['FPR'],lrROC['TPR'])
mpt.ylabel('False Positive Rate')
mpt.xlabel('True Positive Rate')
mpt.title('ROC Curve - Random Forest')
mpt.show()

print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))

# fill the results list

auc_roc.append(auc_rf)

accuracy.append(accuracy_rf)

recall.append(recall_rf)

precision.append(precision_rf)

name_model.append("Random Forest")


# In[ ]:


#####################################################  Visualization ROC curve using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Random Forest')
#mpt.show()


# #### Now, we see the feature importance graphic.

# In[ ]:


# Cross Validation Model: feature importance 

modelRF = CrossValidatorModel.load("tuningRFProof")

bestPipeline = modelRF.bestModel
bestModel = bestPipeline.stages[1] # type: ignore

importances = bestModel.featureImportances

x_values = list(range(len(importances)))

mpt.bar(x_values,importances, orientation='vertical')
mpt.xticks(x_values, columns, rotation = 40)
mpt.ylabel('Importance')
mpt.xlabel('Feature')
mpt.title('Feature Importances')


# ### Predictions

# In[ ]:


# predictions of the model

predictions = modelRF.transform(test)

print('The predictions of the model are:')

predictions.show(10)


# #### We can see that there are three more columns: rawPrediction, probability and prediction.

# ### Evaluation

# #### Let´s check out the Consufion Matrix.

# In[ ]:


preds_and_labels = predictions.select(["prediction","isFraud"])
preds_and_labels = preds_and_labels.withColumn("isFraud", f.col("isFraud").cast(FloatType())).orderBy("prediction")

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print("The Confusion Matrix is:")

metrics.confusionMatrix().toArray()


# ### 4.2.2 Logistic Regression

# #### Training

# In[ ]:


####################################################### modelLRrom the cross validation ##############################################
################################################### (uncomment LR you want to use this model ) #######################################

#modelLR = CrossValidatorModel.load("tuningLR")
#print(modelLR.explainParams())
#type(modelLR)
#modelLR.extractParamMap()


# In[ ]:


lr = LogisticRegression(featuresCol="my_features", labelCol="isFraud", elasticNetParam=0.5, regParam=0.1)


# In[ ]:


print(lr.explainParams())


# In[ ]:


modelLR = lr.fit(train)

predictions = modelLR.transform(test)


# In[ ]:


# evaluation

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)


# #### Hyperparameters tuning

# ##### In the "values list" you can change the values range of the hyperparameters to test with. Besides, you need to set the hyperparameter name in the argument of the "Logistic Regression" model. 

# In[ ]:


# visualization of the overfitting

# define the tree depths to evaluate
values = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
train_scores = []
test_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 lr = LogisticRegression(labelCol="isFraud", featuresCol="my_features", elasticNetParam=i)
 # configure the evaluators and metrics
 evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")
 metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")
 # fit model on the training dataset
 modelLR = lr.fit(train)
 # predictions on the training dataset
 predictions = modelLR.transform(train)
 # evaluate on the train dataset
 train_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 train_scores.append(train_accuracy)
 # evaluate on the test dataset
 predictions = modelLR.transform(test)
 test_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 test_scores.append(test_accuracy)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_accuracy, test_accuracy))


# In[ ]:


# plot of train and test scores vs tree depth

mpt.plot(values, train_scores, '-o', label='Train')
mpt.plot(values, test_scores, '-o', label='Test')
mpt.legend()
mpt.show()


# In[ ]:


# evaluate in the training dataset

predictions = modelLR.transform(train)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)

print(f"Precsion: ", precision_rf)

print(f"Recall: ", recall_rf)


# ROC curve : visualization


print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
print("The ROC curve is:")
trainingSummary = modelLR.summary
lrROC = trainingSummary.roc.toPandas()

mpt.plot(lrROC['FPR'],lrROC['TPR'])
mpt.ylabel('False Positive Rate')
mpt.xlabel('True Positive Rate')
mpt.title('ROC Curve - Logistic Regression')
mpt.show()


# Precision Recall curve : visualization

print("The Precision Recall curve is:")
pr = trainingSummary.pr.toPandas()
mpt.plot(pr['recall'],pr['precision'])
mpt.ylabel('Precision')
mpt.xlabel('Recall')
mpt.title('Precision Recall curve - Logistic Regression')
mpt.show()


# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Logistic Regression')
#mpt.show()


# In[ ]:


# evaluate in the test dataset

predictions = modelLR.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_rf = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)

print(f"Precsion: ", precision_rf)

print(f"Recall: ", recall_rf)


# fill the results list

auc_roc.append(auc_rf)

accuracy.append(accuracy_rf)

recall.append(recall_rf)

precision.append(precision_rf)

name_model.append("Logistic Regression")


# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Logistic Regression')
#mpt.show()


# In[ ]:


# confusion matrix

predictions = modelLR.transform(test)

preds_and_labels = predictions.select(["prediction","isFraud"])
preds_and_labels = preds_and_labels.withColumn("isFraud", f.col("isFraud").cast(FloatType())).orderBy("prediction")

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print("The Confusion Matrix is:")

metrics.confusionMatrix().toArray()


# In[ ]:


# coefficients and intercept

##print('The coefficients are:', modelLR.coefficients)
##print('The independent term is:', modelLR.intercept)


# In[ ]:


# model summary

##model_LR = CrossValidatorModel.load("tuningLRProof")
##summary_lr = model_LR.bestModel

##print('The area under ROC:',summary_lr)
##print('The ROC is:',summary_lr.roc.show())
##print('pr is',summary_lr.pr.show())


# In[ ]:


##summary_lr.objectiveHistory


# #### Predictions

# In[ ]:


# make predictions of the logistic regression model using the test dataset

predictions.select('isfraud','prediction','probability').show(10)


# ### Evaluation

# #### Let´s check out the Confusion Matrix.

# In[ ]:


preds_and_labels = predictions.select(["prediction","isFraud"])
preds_and_labels = preds_and_labels.withColumn("isFraud", f.col("isFraud").cast(FloatType())).orderBy("prediction")

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print("The Confusion Matrix is:")

metrics.confusionMatrix().toArray()


# ### 4.2.3 Decision Tree

# #### Training

# In[ ]:


####################################################### model from the cross validation ##############################################
################################################### (uncomment if you want to use this model ) #######################################

#modelDT = CrossValidatorModel.load("tuningDT")
#print(modelDT.explainParams())
#type(modelDT)
#modelDT.extractParamMap()


# In[ ]:


dt = DecisionTreeClassifier(featuresCol="my_features", labelCol="isFraud")

modelDT = dt.fit(train)

predictions = modelDT.transform(test)


# In[ ]:


# preliminary evaluation

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})


print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)

print(f"Precsion: ", precision_dt)

print(f"Recall: ", recall_dt)


# #### Hyperparameters tuning

# ##### In the "values list" you can change the values range of the hyperparameters to test with. Besides, you need to set the hyperparameter name in the argument of the "Decision Tree" model. 

# In[ ]:


# visualization of the overfitting


# define the tree depths to evaluate
values = [i for i in range(1,16)]
train_scores = []
test_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 dt = DecisionTreeClassifier(featuresCol="my_features",labelCol="isFraud",maxDepth=i)
 # configure the evaluators and metrics
 evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")
 metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")
 # fit model on the training dataset
 modelDT = dt.fit(train)
 # predictions on the training dataset
 predictions = modelDT.transform(train)
 # evaluate on the train dataset
 train_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 train_scores.append(train_accuracy)
 # evaluate on the test dataset
 predictions = modelDT.transform(test)
 test_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 test_scores.append(test_accuracy)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_accuracy, test_accuracy))


# plot of train and test scores vs tree depth
mpt.plot(values, train_scores, '-o', label='Train')
mpt.plot(values, test_scores, '-o', label='Test')
mpt.legend()
mpt.show()


# In[ ]:


# evaluate in the training dataset

predictions = modelDT.transform(train)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_dt = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_dt = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_dt)

print(f"Accuracy: ", accuracy_dt)

print(f"Precsion: ", precision_dt)

print(f"Recall: ", recall_dt)

# ROC curve : visualization

print("The ROC curve is:")
trainingSummary = modelDT.summary
lrROC = trainingSummary.roc.toPandas()

mpt.plot(lrROC['FPR'],lrROC['TPR'])
mpt.ylabel('False Positive Rate')
mpt.xlabel('True Positive Rate')
mpt.title('ROC Curve - Decision Tree')
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
mpt.show()


# Precision Recall curve : visualization

print("The Precision Recall curve is:")
pr = trainingSummary.pr.toPandas()
mpt.plot(pr['recall'],pr['precision'])
mpt.ylabel('Precision')
mpt.xlabel('Recall')
mpt.title('Precision Recall curve - Decision Tree')
mpt.show()


# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Decision Tree')
#mpt.show()


# In[ ]:


# evaluate in the test dataset

predictions = modelDT.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_dt = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_dt = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_dt)

print(f"Accuracy: ", accuracy_dt)

print(f"Precision: ", precision_dt)

print(f"Recall: ", recall_dt)

# full the results list

auc_roc.append(auc_dt)

accuracy.append(accuracy_dt)

recall.append(recall_dt)

precision.append(precision_dt)

name_model.append("Decision Tree")



# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Logistic Regression')
#mpt.show()


# #### Predictions

# In[ ]:


# make predictions of the decision tree model using the test dataset

predictions = modelDT.transform(test)

print("The predictions are:")

predictions.show(50)


# ### Evaluation

# #### Let´s check out the Confusion Matrix.

# In[ ]:


preds_and_labels = predictions.select(["prediction","isFraud"])
preds_and_labels = preds_and_labels.withColumn("isFraud", f.col("isFraud").cast(FloatType())).orderBy("prediction")

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print("The Confusion Matrix is:")

metrics.confusionMatrix().toArray()


# ### 4.2.4 Naive Bayes

# #### Training

# In[ ]:


####################################################### model from the cross validation ##############################################
################################################### (uncomment if you want to use this model ) #######################################

#modelNB = CrossValidatorModel.load("tuningNB")
#print(modelNB.explainParams())
#type(modelNB)
#modelNB.extractParamMap()


# In[ ]:


nb = NaiveBayes(featuresCol="my_features", labelCol="isFraud")

modelNB = nb.fit(train)

predictions = modelNB.transform(test)


# In[ ]:


print(nb.explainParams())


# In[ ]:


# preliminary evaluation

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_rf = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_rf = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

print(f"AUC-ROC: ", auc_rf)

print(f"Accuracy: ", accuracy_rf)


# #### Hyperparameters tuning

# ##### In the "values list" you can change the values range of the hyperparameters to test with. Besides, you need to set the hyperparameter name in the argument of the "Naive Bayes" model. 

# In[ ]:


# visualization of the overfitting

# define the tree depths to evaluate
values = [i for i in range(0, 11)]
train_scores = []
test_scores = []
# evaluate a decision tree for each depth
for i in values:
 # configure the model
 lr = NaiveBayes(labelCol="isFraud", featuresCol="my_features", smoothing=i)
 # configure the evaluators and metrics
 evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")
 metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")
 # fit model on the training dataset
 modelLR = lr.fit(train)
 # predictions on the training dataset
 predictions = modelLR.transform(train)
 # evaluate on the train dataset
 train_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 train_scores.append(train_accuracy)
 # evaluate on the test dataset
 predictions = modelLR.transform(test)
 test_accuracy = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})
 test_scores.append(test_accuracy)
 # summarize progress
 print('>%d, train: %.3f, test: %.3f' % (i, train_accuracy, test_accuracy))

 # plot of train and test scores vs tree depth

mpt.plot(values, train_scores, '-o', label='Train')
mpt.plot(values, test_scores, '-o', label='Test')
mpt.legend()
mpt.show()


# In[ ]:


# evaluate in the training dataset

predictions = modelNB.transform(train)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_dt = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_dt = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_dt)

print(f"Accuracy: ", accuracy_dt)

print(f"Precsion: ", precision_dt)

print(f"Recall: ", recall_dt)


# ROC curve : visualization

print("The ROC curve is:")
trainingSummary = modelNB.summary
lrROC = trainingSummary.roc.toPandas()

mpt.plot(lrROC['FPR'],lrROC['TPR'])
mpt.ylabel('False Positive Rate')
mpt.xlabel('True Positive Rate')
mpt.title('ROC Curve - Naive Bayes')
print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
mpt.show()


# Precision Recall curve : visualization

print("The Precision Recall curve is:")
pr = trainingSummary.pr.toPandas()
mpt.plot(pr['recall'],pr['precision'])
mpt.ylabel('Precision')
mpt.xlabel('Recall')
mpt.title('Precision Recall curve - Naive Bayes')
mpt.show()


# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Naive Bayes')
#mpt.show()


# In[ ]:


# evaluate in the test dataset

predictions = modelNB.transform(test)

evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="isFraud")

auc_dt = evaluator.evaluate(predictions)

# Accuracy, Precision and Recall

metrics = MulticlassClassificationEvaluator(predictionCol="prediction",labelCol="isFraud")

accuracy_dt = metrics.evaluate(predictions, {metrics.metricName:"accuracy"})

precision_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedPrecision"})

recall_dt = metrics.evaluate(predictions, {metrics.metricName:"weightedRecall"})

print(f"AUC-ROC: ", auc_dt)

print(f"Accuracy: ", accuracy_dt)

print(f"Precsion: ", precision_dt)

print(f"Recall: ", recall_dt)

# full the results list

auc_roc.append(auc_dt)

accuracy.append(accuracy_dt)

recall.append(recall_dt)

precision.append(precision_dt)

name_model.append("Naive Bayes")



# In[ ]:


#####################################################  Visualization ROC CURVE using Sklearn ############################################################

########################### ROC graphic ###########################

#preds_and_labels = predictions.select("prediction", "isFraud")
#preds_and_labels_collect = preds_and_labels.collect()

#preds_and_labels_list = [ (float(i[0][0]), 1.0 - float(i[1])) for i in preds_and_labels_collect  ]
#preds_and_labels = sc.parallelize(preds_and_labels_list)

#metrics = BinaryClassificationMetrics(preds_and_labels)


######################### Visualization ###########################

#fpr = dict()
#tpr = dict()
#roc_auc = dict()

#y_test = [i[1] for i in preds_and_labels_collect]
#y_score = [i[0] for i in preds_and_labels_collect]

#fpr, tpr, _ = roc_curve(y_test, y_score)

#roc_auc = auc(fpr, tpr)

#print("The graphic ROC curve is:")

#mpt.figure(figsize=(5,4))
#mpt.plot(fpr, tpr, label='ROC curve' % roc_auc)
#mpt.plot([0,1],[0,1],'k--')
#mpt.xlim([0.0,1.0])
#mpt.ylim([0.0,1.05])
#mpt.xlabel('False Positive Rate')
#mpt.ylabel('True Positive Rate')
#mpt.title('ROC Curve - Naive Bayes')
#mpt.show()


# #### Predictions

# In[ ]:


# make predictions of the naive bayes model using the test dataset

predictions = modelNB.transform(test)

predictions.show(50)


# #### Evaluation

# #### Let´s check out the Confusion Matrix.

# In[ ]:


preds_and_labels = predictions.select(["prediction","isFraud"])
preds_and_labels = preds_and_labels.withColumn("isFraud", f.col("isFraud").cast(FloatType())).orderBy("prediction")

metrics = MulticlassMetrics(preds_and_labels.rdd.map(tuple))

print("The Confusion Matrix is:")

metrics.confusionMatrix().toArray()


# ## 4.3 Evaluation and Selection of the model
# 
# We´ll evaluate the models using the metrics used in the previous step and we´ll select the model with the best performance. As first step, let´s create a dictionary with the results of every model.

# In[ ]:


results = {
    'Name_Model': name_model,
    'Accuracy':accuracy,
    'Precision':precision,
    'Recall':recall,
    'AUC_ROC':auc_roc
}


# In[ ]:


results


# #### Now, let´s create a pandas dataframe with the results dictionary.

# In[ ]:


results_df = pd.DataFrame(results)
results_df.set_index('Name_Model', inplace=True)
#results_df.set_index("Name_Model", inplace=True)


# In[ ]:


results_df.head(5)


# #### Let´s visualize these results.

# In[ ]:


# "results_df" dataframe

colors = ['#0077b6','#CDDBF3','#9370DB','#DDA0DD']
results_df.plot(kind='bar', figsize=(12,6), colormap='viridis', rot=0)
mpt.title('Comparison of metrics per model')
mpt.xlabel('Models')
mpt.ylabel('Score')
mpt.legend(title = 'Metrics')
mpt.tight_layout
mpt.show()

pd.DataFrame()


# In[ ]:


# transpose of the "results_df" dataframe

results_df = results_df.T
colors = ['#0077b6','#CDDBF3','#9370DB','#DDA0DD']
results_df.plot(kind='bar', figsize=(12,6), colormap='viridis', rot=0)
mpt.title('Comparison of metrics per model')
mpt.xlabel('Metrics')
mpt.ylabel('Score')
mpt.legend(title = 'Models')
mpt.tight_layout
mpt.show()


# ## 5. Conclusions

# ### + The fraud is related with the variables "Payment", "CC" and "CM".
# ### + Random Forest represents the machine learning algorithm with the highest reliability without "overfitting".

# ## 6. Storage

# ### 6.1 Model

# In[ ]:


# model: Random Forest

modelRF.save("modelRF")

# model: Logistic Regression

modelLR.save("modelLR")

# model: Decision Tree

modelDT.save("modelDT")

# model: Naive Bayes

modelNB.save("modelNB")


# ### 6.2 Load

# In[ ]:


# model: Random Forest

loaded_model_RF =  CrossValidatorModel.load("modelRF")

# model: Logistic Regression

loaded_model_LR = CrossValidatorModel.load("modelLR")

# model: Decision Tree

loaded_model_LR = CrossValidatorModel.load("modelDT")

# model: Naive Bayes

loaded_model_LR = CrossValidatorModel.load("modelNB")

