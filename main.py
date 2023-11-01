from mri_workflow import MriWorkflow

def main():
    workflow = MriWorkflow("/Users/berkearda/Desktop/pai_project_1_1/aml_task_1/task1/X_train.csv", "/Users/berkearda/Desktop/pai_project_1_1/aml_task_1/task1/y_train.csv")
    workflow.execute()
   
if __name__ == "__main__":
    main()