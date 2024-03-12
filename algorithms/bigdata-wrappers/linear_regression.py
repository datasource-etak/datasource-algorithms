
import json
from collections import OrderedDict

## workflow_all_all_4_models_linear_regression.py
def run(spark, workflowObject):
    output = workflowObject.linearRegression()

    data = spark.createDataFrame(output.reset_index())
    jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.toJSON().collect()))
    print(json.dumps([{"type": "table", "name" : "Linear Regression Estimated Parameters", "data" : jdata}]))
    ## TO-DO add store train_test_data
    return workflowObject
