import json, uuid
from pyspark.sql.functions import col, monotonically_increasing_id, lit, concat
from collections import OrderedDict


def run(spark, slug, workflow_alias, workflowObject):
    def parse_output(o, name):
      data = spark.createDataFrame(o.reset_index())
      jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.drop("index").toJSON().collect()))
      return {"type": "table", "name" : name, "data" : jdata}

    def save_dataset(pdata, slug, use_test):
      if True:
        suffix = "test" if use_test else "train"
        id = str(uuid.uuid4())
        dataset = spark.createDataFrame(pdata)
        dataset = dataset.withColumn("message_type", lit(id))
        dataset = dataset.withColumn("slug", lit(slug))
        dataset = dataset.withColumn("dataset_id", lit(id))
        dataset = dataset.withColumn("message_id", concat(col("message_type"), monotonically_increasing_id().cast("string")))
        dataset = dataset.withColumn("alias", lit(workflow_alias + "_target_predictions_" + suffix))
        #dataset.show()
        catalog = """
          {
              "table": {
                  "namespace": \""""+slug + "_db_el"+"""\",
                  "name": "Events"
              },
              "rowkey": "key",
              "columns": {
                  "message_id":{"cf":"rowkey", "col":"key", "type":"string"},"""        
        result_dataset_schema = {}
        for column in dataset.schema.fields:
          if column.name != "message_id":
            catalog+="""
                      \""""+column.name+"""\":{"cf":"messages", "col":\""""+column.name+"""\", "type":"string"},"""
            result_dataset_schema[column.name] = column.dataType.simpleString()
            dataset = dataset.withColumn(column.name, col(column.name).cast("string"))
        catalog=catalog[0:-1]
        catalog+="""
              }
          }"""
        dataset.write.options(catalog=catalog, newtable=5).format("org.apache.spark.sql.execution.datasources.hbase").save()
        dataset_descr = {"type": "multiple_scatterplots_with_line", "uuid": id, "name" : suffix.capitalize() + " Targets vs. Predictions",  "alias" : workflow_alias + "_target_predictions_" + suffix, "dataset_description": "Dataset " + suffix + " targets vs predictions of workflow " + workflow_alias, "schema" : result_dataset_schema}
        return dataset_descr
      return None
    result = workflowObject.getSignificantsFeaturesForRegressionModels()
    data = spark.createDataFrame(result.reset_index())
    jdata = list(map(lambda x: json.loads(x, object_pairs_hook=OrderedDict), data.drop("index").toJSON().collect()))
    
    ret_res = [{"type": "table", "name" : "Feature Significance per Model", "data" : jdata}]
    predictions = workflowObject.getTargetsPredictionsForRegressionModels()
    
    output = save_dataset(predictions[0], slug, False)
    ret_res.append(output)
    if workflowObject.isSplit:
        output_test = save_dataset(predictions[1], slug, True)
        ret_res.append(output_test)
        
    output = workflowObject.compareRegressionModelOutputs()
    ret_res.append(parse_output(output[0], "Evaluation metrics on train data"))
    if len(output) > 1:
        ret_res.append(parse_output(output[1], "Evaluation metrics on test data"))
        
    print(json.dumps(ret_res))
    return ""