import json
from collections import OrderedDict

## workflow_all_all_3_statistics_skew.py
def run(spark, workflowObject):
    data = spark.createDataFrame(workflowObject.skew().reset_index())
    data = data.withColumnRenamed("index", "feature")
    data = data.withColumnRenamed("0", "skew")
    pdata = data.toPandas()
    labels = list(pdata["feature"])
    values = list(pdata["skew"])
    print(json.dumps([{"type": "barchart", "name" : "Skew of Training Features and Target", "data" : {"labels": labels, "values" : values}}]))

    #data.show()

    ## TO-DO add store train_test_data
    return workflowObject
