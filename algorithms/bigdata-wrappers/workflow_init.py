from pyspark.sql.functions import col
import json
from algorithmic_library import *

def run(spark, dataset, features, target, schema):
    features = features.split(",")
    dataset = dataset.drop("source_id", "dataset_id","alias")
    schemaJson = json.loads(schema)
    for column in schemaJson.keys():
        dataset = dataset.withColumn(column, col(column).cast(schemaJson[column]))
    pandasDataset = dataset.toPandas()
    workflowObject = DatasourceAlgorithmic(pandasDataset, "supervised", features, target)
    return workflowObject
