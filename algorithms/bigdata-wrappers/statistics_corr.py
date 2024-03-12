import json
from collections import OrderedDict

## workflow_all_all_3_statistics_corr.py
def run(spark, with_target, workflowObject):
    with_target = True if with_target == "true" else False
    data = spark.createDataFrame(workflowObject.corr(with_target).reset_index())
    data = data.withColumnRenamed("index", "versus")
    jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.toJSON().collect()))
    suffix = "All vs Target" if with_target else "All vs All"
    print(json.dumps([{"type": "table", "name" : "Correlation of Features and Target - " + suffix, "data" : jdata}]))
    ## TO-DO add store train_test_data
    return workflowObject
