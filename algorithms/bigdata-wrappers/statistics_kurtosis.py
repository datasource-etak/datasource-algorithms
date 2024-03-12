import json
from collections import OrderedDict

## workflow_all_all_3_statistics_kurtosis.py
def run(spark, workflowObject):
    data = spark.createDataFrame(workflowObject.kurtosis().reset_index())
    data = data.withColumnRenamed("index", "feature")
    data = data.withColumnRenamed("0", "kurtosis")
    pdata = data.toPandas()
    labels = list(pdata["feature"])
    values = list(pdata["kurtosis"])
    print(json.dumps([{"type": "barchart", "name" : "Kurtosis of Training Features and Target", "data" : {"labels": labels, "values" : values}}]))

    ## TO-DO add store train_test_data
    return workflowObject
