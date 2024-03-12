
import json
from collections import OrderedDict
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


## workflow_all_all_4_models_panel_OLS.py
def run(spark, timeColumn, entity_effects, time_effects, workflowObject):
    entity_effects = True if entity_effects == "true" else False
    time_effects = True if time_effects == "true" else False
    output = workflowObject.panelOLS(timeColumn, entity_effects, time_effects)
    data = spark.createDataFrame(output)
    jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.toJSON().collect()))
    name = "PanelOLS(entity_effects=" + str(entity_effects) + ",time_effects=" + str(time_effects) + ")"
    print(json.dumps([{"type": "table", "name" : name, "data" : jdata}]))
    ## TO-DO add store train_test_data
    return workflowObject
