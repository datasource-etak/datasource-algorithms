
def run(spark, workflowObject):
    workflowObject.registerRMSE()
    return workflowObject