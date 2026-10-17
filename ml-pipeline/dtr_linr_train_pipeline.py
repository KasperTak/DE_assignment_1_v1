#!/usr/bin/env python
# coding: utf-8

# # Installation
# ###### Install the packages required for executing this notebook

# # In[ ]:


# get_ipython().system(' pip3 install --user --no-cache-dir --upgrade "kfp>2" "google-cloud-pipeline-components>2"                                         google-cloud-aiplatform')


# # In[ ]:


# get_ipython().system(' pip3 install --upgrade --quiet google-cloud-aiplatform                                    google-cloud-storage                                   kfp                                   google-cloud-pipeline-components')


# # #### Check the versions of the packages you installed. The KFD SDK version shoudl be > 2.

# # In[ ]:


# get_ipython().system(' python3 -c "import kfp; print(\'KFP SDK version: {}\'.format(kfp.__version__))"')
# get_ipython().system(' pip3 freeze | grep aiplatform')
# get_ipython().system(' python3 -c "import google_cloud_pipeline_components; print(\'google_cloud_pipeline_components version: {}\'.format(google_cloud_pipeline_components.__version__))"')


# # #### Project and Pipeline Configurations

# In[ ]:


#The Google Cloud project that this pipeline runs in.
PROJECT_ID = "deassignment1-511413"
# The region that this pipeline runs in
REGION = "us-central1"
# Specify a Cloud Storage URI that your pipelines service account can access. The artifacts of your pipeline runs are stored within the pipeline root.
PIPELINE_ROOT = "gs://deassignment1_temp"


# In[ ]:


import kfp
import typing
from typing import Dict
from typing import NamedTuple
from kfp import dsl
from kfp.dsl import (Artifact,
                        Dataset,
                        Input,
                        Model,
                        Output,
                        Metrics,
                        ClassificationMetrics,
                        component, 
                        OutputPath, 
                        InputPath)
import google.cloud.aiplatform as aip
from google_cloud_pipeline_components.types import artifact_types


# #### Pipeline Component: Data Ingestion

# In[ ]:


@dsl.component(
    packages_to_install=["pandas","google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def download_data(project_id: str, bucket: str, file_name: str, dataset: Output[Dataset]):
    '''download data'''
    from google.cloud import storage
    import pandas as pd
    import logging 
    import sys
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Downloading the file from a Google bucket 
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket)
    blob = bucket.blob(file_name)
    blob.download_to_filename(dataset.path + ".csv")
    logging.info('Downloaded Data!')


# #### Pipeline Component: Dataset splitting
# One initial dataset (housing.csv) will be split into train- en test-set

# In[ ]:


@dsl.component(
    packages_to_install=["pandas","scikit-learn==1.3.2"],
    base_image="python:3.12.7-slim"
)
def split_data(dataset: Input[Dataset], train_set: Output[Dataset], test_set: Output[Dataset], test_size: float=0.2):
    # split housing.csv into a train set and a test set.
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset.path + ".csv")
    train, test = train_test_split(df, test_size=test_size, random_state=42)

    # NaN-waarden in kolom "total_bedrooms" veranderen naar mediaan. 
    median_bedrooms = train["total_bedrooms"].median()
    train["total_bedrooms"] = train["total_bedrooms"].fillna(median_bedrooms)
    test["total_bedrooms"] = test["total_bedrooms"].fillna(median_bedrooms)



    train.to_csv(train_set.path + ".csv", index=False)
    test.to_csv(test_set.path + ".csv", index=False)


# #### Pipeline Component: Training Linear Regression model

# In[ ]:


@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_linr(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a LinR with default parameters'''
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import json
    import logging 
    import sys
    import os
    import pickle  

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    logging.info(df.columns)

    X = df.drop(columns=["longitude", "latitude", "median_house_value"])
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    linr_model = LinearRegression()
    # fit the model
    linr_model.fit(X_train, y_train)
    # Evaluate the model
    val_predictions = linr_model.predict(X_test)
    MAE = mean_absolute_error(y_test, val_predictions)
    MSE = mean_squared_error(y_test, val_predictions)
    RMSE = MSE ** 0.5

    metrics_dict = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "linr"
    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(linr_model, f)

    outputs = NamedTuple('output', metrics=dict)
    return outputs(metrics_dict)



# #### Pipeline Component: Training Decision Tree regression

# In[ ]:


@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_dtr(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a DTR with default parameters'''
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn import metrics
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import json
    import logging 
    import sys
    import os
    import pickle  

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    logging.info(df.columns)

    X = df.drop(columns=["longitude", "latitude", "median_house_value"])
    y = df["median_house_value"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    dtr_model = DecisionTreeRegressor(criterion='absolute_error', random_state=42)
    # fit the model
    dtr_model.fit(X_train, y_train)
    # Evaluate the model
    val_predictions = dtr_model.predict(X_test)
    MAE = mean_absolute_error(y_test, val_predictions)
    MSE = mean_squared_error(y_test, val_predictions)
    RMSE = MSE ** 0.5

    metrics_dict = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "dtr"
    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(dtr_model, f)

    outputs = NamedTuple('output', metrics=dict)
    return outputs(metrics_dict)



# #### Pipeline Component: Prediction with Scikit

# In[ ]:


@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def predict(model: Input[Model], features: Input[Dataset], results: Output[Dataset]):
    import pandas as pd
    import pickle  
    import json
    import logging
    import sys
    import os

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    filename = model.path + ".pkl"

    # Loading the saved model
    model_sk = pickle.load(open(filename, 'rb'))

    xNew = df[model_sk.feature_names_in_]
    dfcp = df.copy()
    y_mhv = model_sk.predict(xNew) # mhv = Median House Value (aka Y)
    logging.info(y_mhv)
    dfcp['pmhv'] = y_mhv.tolist()
    dfcp.to_csv(results.path + ".csv", index=False, encoding='utf-8-sig')


# #### Pipeline Component: Algorithm Selection

# In[ ]:


@dsl.component(
    base_image="python:3.12.7-slim"
)
def compare_model(dtr_metrics: dict, linr_metrics: dict) -> str:
    import logging
    import json
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(dtr_metrics)
    logging.info(linr_metrics)
    if dtr_metrics.get("MAE") > linr_metrics.get("MAE"):
        return "DTR"
    else :
        return "LINR"


# #### Upload Model and Metrics to Google Bucket

# In[ ]:


@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def upload_model_to_gcs(project_id: str, model_repo: str, model: Input[Model]):
    '''upload model to gsc'''
    from google.cloud import storage   
    import logging 
    import sys
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)    

    # upload model to Google Cloud Storage
    client = storage.Client(project = project_id)
    bucket = client.bucket(model_repo)
    mb_path = str(model.metadata["algo"]) + '/' + str(model.metadata["algo"]) + '_model' + str(model.metadata["file_type"])
    blob = bucket.blob(mb_path)
    blob.upload_from_filename(model.path + str(model.metadata["file_type"]))
    logging.info("Saved the model to the folder : " + mb_path + "in the bucket" + model_repo)


# #### Define the Pipeline

# In[ ]:


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="houseprice-predictor-training-pipeline ")
def pipeline(project_id: str, data_bucket: str, dataset_filename: str, model_repo: str):

    # 1.    housing.csv downloaden
    di_op = download_data(
        project_id=project_id,
        bucket=data_bucket,
        file_name=dataset_filename
    )

    # 2.    opsplitsen in train_set en test_set
    split_op = split_data(dataset=di_op.outputs["dataset"])

    # 3.    beide modellen trainen op de train_set
    training_dtr_job_run_op = train_dtr(
        features=split_op.outputs["train_set"]
    )

    training_linr_job_run_op = train_linr(
        features=split_op.outputs["train_set"]
    )

    # 4.    Metrics vergelijken
    comp_model_op = compare_model(dtr_metrics=training_dtr_job_run_op.outputs["metrics"],
                                  linr_metrics=training_linr_job_run_op.outputs["metrics"]).after(training_dtr_job_run_op, training_linr_job_run_op)

    # 5.    alleen het beste model gebruiken
    with dsl.If(comp_model_op.output == "DTR"):
        predict_dtr_job_run_op = predict(model=training_dtr_job_run_op.outputs["out_model"], features=split_op.outputs["test_set"])

        upload_model_dtr_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_dtr_job_run_op.outputs["out_model"]
        ).after(predict_dtr_job_run_op)

    with dsl.If(comp_model_op.output == "LINR"):
        predict_linr_job_run_op = predict(model=training_linr_job_run_op.outputs["out_model"], features=split_op.outputs["test_set"])

        upload_model_linr_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_linr_job_run_op.outputs["out_model"]
        ).after(training_linr_job_run_op)
    
    


# ### Compile the pipeline into a JSON file

# In[ ]:


from kfp import compiler
compiler.Compiler().compile(pipeline_func=pipeline,
                            package_path="houseprice_predictor_training_pipeline.yaml")


# ### Submit the pipeline run

# In[ ]:


import google.cloud.aiplatform as aip

# Before initializing, make sure to set the GOOGLE_APPLICATION_CREDENTIALS
# environment variable to the path of your service account.
aip.init(
    project=PROJECT_ID,
    staging_bucket=PIPELINE_ROOT,
)

# Prepare the pipeline job
job = aip.PipelineJob(
    display_name="houseprice-predictor-ct-pipeline",
    enable_caching=False,
    template_path="houseprice_predictor_training_pipeline.yaml",
    pipeline_root=PIPELINE_ROOT,
    location=REGION,
    parameter_values={
        'project_id': PROJECT_ID, # makesure to use your project id (deassignment1-511413)
        'data_bucket': 'deassignment1_data',  # makesure to use your data bucket name 
        'dataset_filename': 'housing.csv',     # makesure to upload these to your data bucket from DE2026/lab4/data
        'model_repo':'deassignment1_models' # make sure to use your model bucket name 
    }
)

job.run()

