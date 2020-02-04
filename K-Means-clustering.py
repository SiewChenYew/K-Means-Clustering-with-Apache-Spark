import numpy as np
import matplotlib.pyplot as plt
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import MinMaxScaler

# Loads data.
df = spark.read.format("csv").option("header","true").load("/home/student/Downloads/train.csv")
dataset = df.drop('session_id','DateTime','user_id','product_category_2','is_click')
dataset = dataset.dropna()
# No of rows
dataset.count()
dataset.show()

indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(dataset) for column in list(dataset.columns)]

pipeline = Pipeline(stages=indexers)
dataset_r = pipeline.fit(dataset).transform(dataset)
columnList = [item[0] for item in dataset_r.dtypes if item[1].startswith('double')]
dataset_numeric = dataset_r.select(columnList)

vecAssembler = VectorAssembler(inputCols=list(dataset_numeric.columns), outputCol="features")
transformed = vecAssembler.transform(dataset_numeric)
scaler = MinMaxScaler(inputCol="features",\
         outputCol="scaledFeatures")
scalerModel =  scaler.fit(transformed.select("features"))
df_kmeans = scalerModel.transform(transformed)
df_kmeans.show()
df_kmeans = df_kmeans.select('scaledFeatures')

cost = np.zeros(10)
for k in range(2,10):
    kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
    model = kmeans.fit(df_kmeans.sample(False,0.1, seed=42))
    cost[k] = model.computeCost(df_kmeans) 

fig, ax = plt.subplots(1,1, figsize =(8,6))
ax.plot(range(2,10),cost[2:10])
ax.set_xlabel('k')
ax.set_ylabel('cost')
plt.show()

k = 7
kmeans = KMeans().setK(k).setSeed(1).setFeaturesCol("scaledFeatures")
model = kmeans.fit(df_kmeans)
centers = model.clusterCenters()

print("Cluster Centers: ")
for center in centers:
    print(center)

transformed = model.transform(df_kmeans).select('scaledFeatures','prediction')
rows = transformed.collect()
print(rows[:3])

df_pred = sqlContext.createDataFrame(rows)
df_pred.show()


