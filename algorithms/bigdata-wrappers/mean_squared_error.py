
def run(spark, workflowObject):
    workflowObject.registerMSE()
    return workflowObject