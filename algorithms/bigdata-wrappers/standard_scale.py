import json, uuid
from pyspark.sql.functions import col, monotonically_increasing_id, lit, concat

def run(spark, workflow_alias, slug, workflowObject):
    def save_dataset(workflowObject, use_test):
      if use_test == False or workflowObject.isSplit == True:
        pdata = workflowObject.getTestData() if use_test else workflowObject.getTrainData()
        suffix = "test" if use_test else "train"
        id = str(uuid.uuid4())
        dataset = spark.createDataFrame(pdata)
        dataset = dataset.withColumn("message_type", lit(id))
        dataset = dataset.withColumn("slug", lit(slug))
        dataset = dataset.withColumn("dataset_id", lit(id))
        dataset = dataset.withColumn("message_id", concat(col("message_type"), monotonically_increasing_id().cast("string")))
        dataset = dataset.withColumn("alias", lit(workflow_alias + "_scale_" + suffix))
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
        dataset_descr = {"type": "dataset", "uuid": id, "name" : suffix.capitalize() + " Data After Scale", "alias" : workflow_alias + "_scale_" + suffix, "dataset_description": "Dataset " + suffix + " part after scale in workflow " + workflow_alias, "schema" : result_dataset_schema}
        return dataset_descr
      return None

    workflowObject.scale("standard")

    train = save_dataset(workflowObject, False)
    test = save_dataset(workflowObject, True)
    final = [train] if test is None else [train, test]
    print(json.dumps(final))

    ## TO-DO add store train_test_data
    return workflowObject
