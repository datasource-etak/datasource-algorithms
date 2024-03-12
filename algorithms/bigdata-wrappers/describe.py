import json
from collections import OrderedDict

def run(spark, workflowObject):
    data = spark.createDataFrame(workflowObject.getStats().reset_index())
    data = data.withColumnRenamed("index", "metric")
    jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.toJSON().collect()))

    print(json.dumps([{"type": "table", "name" : "Features and Target Statistics", "data" : jdata}]))

    ## TO-DO add store train_test_data
    return workflowObject
