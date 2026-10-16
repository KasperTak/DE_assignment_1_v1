import kfp
import argparse
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
from kfp import compiler


# Pipeline Component : Data Ingestion
@dsl.component(
    packages_to_install=["pandas", "google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def download_data(project_id: str, bucket: str, file_name: str, dataset: Output[Dataset]):
    '''download data and align columns with prediction-ui'''
    from google.cloud import storage
    import pandas as pd
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # Downloading the file from a Google bucket
    client = storage.Client(project=project_id)
    bucket_obj = client.bucket(bucket)
    blob = bucket_obj.blob(file_name)
    raw_path = dataset.path + "_raw.csv"
    blob.download_to_filename(raw_path)
    logging.info('Downloaded Data!')

    # Load and clean dataset to match the 6 features used in prediction-ui
    df = pd.read_csv(raw_path)

    # Rename standard Kaggle California Housing columns to match prediction-ui keys
    column_mapping = {
        'housing_median_age': 'mah',
        'total_rooms': 'trb',
        'total_bedrooms': 'tbb',
        'population': 'tprb',
        'households': 'thb',
        'median_income': 'mihb',
        'median_house_value': 'price'
    }
    df = df.rename(columns=column_mapping)

    # Impute missing values in total_bedrooms ('tbb') with median
    if 'tbb' in df.columns:
        df['tbb'] = df['tbb'].fillna(df['tbb'].median())

    # Keep only the 6 UI features (+ target column 'price' if present in the file)
    feature_cols = ['mah', 'trb', 'tbb', 'tprb', 'thb', 'mihb']
    if 'price' in df.columns:
        df = df[feature_cols + ['price']]
    else:
        df = df[feature_cols]

    df = df.dropna()
    df.to_csv(dataset.path + ".csv", index=False)
    logging.info(f"Saved cleaned dataset with columns: {df.columns.tolist()}")


# Pipeline Component : Training Decision Tree Regressor
@dsl.component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_dt(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a DecisionTreeRegressor with default parameters'''
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import logging
    import sys
    import pickle

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    logging.info(df.columns)

    feature_cols = ['mah', 'trb', 'tbb', 'tprb', 'thb', 'mihb']
    train_X, val_X, train_y, val_y = train_test_split(
        df[feature_cols],
        df['price'],
        test_size=0.30,
        random_state=42
    )

    model_dt = DecisionTreeRegressor(random_state=42)
    model_dt.fit(train_X, train_y)

    # Evaluate the regression model
    val_predictions = model_dt.predict(val_X)
    r2 = float(r2_score(val_y, val_predictions))
    rmse = float(np.sqrt(mean_squared_error(val_y, val_predictions)))
    mae = float(mean_absolute_error(val_y, val_predictions))

    metrics_dict = {
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "dt"

    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(model_dt, f)

    outputs = NamedTuple('outputs', metrics=dict)
    return outputs(metrics_dict)


# Pipeline Component : Training Linear Regression
@dsl.component(
    packages_to_install=['pandas', 'numpy', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def train_lr(features: Input[Dataset], out_model: Output[Model]) -> NamedTuple('outputs', metrics=dict):
    '''train a LinearRegression with default parameters'''
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import logging
    import sys
    import pickle

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    logging.info(df.columns)

    feature_cols = ['mah', 'trb', 'tbb', 'tprb', 'thb', 'mihb']
    x_train, x_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df['price'],
        test_size=0.30,
        random_state=101
    )

    model_lr = LinearRegression()
    model_lr.fit(x_train, y_train)

    # Evaluate the regression model
    predictions = model_lr.predict(x_test)
    r2 = float(r2_score(y_test, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))

    metrics_dict = {
        "r2_score": r2,
        "rmse": rmse,
        "mae": mae,
    }
    logging.info(metrics_dict)

    out_model.metadata["file_type"] = ".pkl"
    out_model.metadata["algo"] = "lr"

    # Save the model
    m_file = out_model.path + ".pkl"
    with open(m_file, 'wb') as f:
        pickle.dump(model_lr, f)

    outputs = NamedTuple('outputs', metrics=dict)
    return outputs(metrics_dict)


# Pipeline Component : Prediction with Scikit
@dsl.component(
    packages_to_install=['pandas', 'scikit-learn==1.3.2'],
    base_image="python:3.12.7-slim"
)
def predict(model: Input[Model], features: Input[Dataset], results: Output[Dataset]):
    import pandas as pd
    import pickle
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    df = pd.read_csv(features.path + ".csv")
    filename = model.path + ".pkl"

    # Loading the saved model
    model_sk = pickle.load(open(filename, 'rb'))

    # Select the exact 6 features used in prediction-ui
    xNew = df[['mah', 'trb', 'tbb', 'tprb', 'thb', 'mihb']]

    dfcp = df.copy()
    y_preds = model_sk.predict(xNew)
    logging.info(y_preds)
    dfcp['predicted_price'] = y_preds.tolist()
    dfcp.to_csv(results.path + ".csv", index=False, encoding='utf-8-sig')


# Pipeline Component : Algorithm Selection
@dsl.component(
    base_image="python:3.12.7-slim"
)
def compare_model(dt_metrics: dict, lr_metrics: dict) -> str:
    import logging
    import sys
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.info(dt_metrics)
    logging.info(lr_metrics)
    if dt_metrics.get("r2_score") > lr_metrics.get("r2_score"):
        return "DT"
    else:
        return "LR"


# Upload Model to Google Bucket
@dsl.component(
    packages_to_install=["google-cloud-storage"],
    base_image="python:3.12.7-slim"
)
def upload_model_to_gcs(project_id: str, model_repo: str, model: Input[Model]):
    '''upload model to gcs'''
    from google.cloud import storage
    import logging
    import sys

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    client = storage.Client(project=project_id)
    bucket = client.bucket(model_repo)

    # Save under algorithm subfolder (e.g., dt/dt_model.pkl or lr/lr_model.pkl)
    mb_path = str(model.metadata["algo"]) + '/' + str(model.metadata["algo"]) + '_model' + str(
        model.metadata["file_type"])
    blob = bucket.blob(mb_path)
    blob.upload_from_filename(model.path + str(model.metadata["file_type"]))
    logging.info("Saved the model to the folder : " + mb_path + " in the bucket " + model_repo)

    # Also save as model.pkl in bucket root for prediction-api
    root_blob = bucket.blob("model" + str(model.metadata["file_type"]))
    root_blob.upload_from_filename(model.path + str(model.metadata["file_type"]))


# Define the workflow of the pipeline.
@kfp.dsl.pipeline(
    name="california-housing-training-pipeline")
def ml_pipeline(project_id: str, data_bucket: str, trainset_filename: str, model_repo: str, testset_filename: str):
    di_op = download_data(
        project_id=project_id,
        bucket=data_bucket,
        file_name=trainset_filename
    )

    training_dt_job_run_op = train_dt(
        features=di_op.outputs["dataset"]
    )

    training_lr_job_run_op = train_lr(
        features=di_op.outputs["dataset"]
    )

    pre_di_op = download_data(
        project_id=project_id,
        bucket=data_bucket,
        file_name=testset_filename
    ).after(training_dt_job_run_op, training_lr_job_run_op)

    comp_model__op = compare_model(
        dt_metrics=training_dt_job_run_op.outputs["metrics"],
        lr_metrics=training_lr_job_run_op.outputs["metrics"]
    ).after(training_dt_job_run_op, training_lr_job_run_op)

    # defining the branching condition
    with dsl.If(comp_model__op.output == "DT"):
        predict_dt_job_run_op = predict(
            model=training_dt_job_run_op.outputs["out_model"],
            features=pre_di_op.outputs["dataset"]
        )
        upload_model_dt_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_dt_job_run_op.outputs['out_model']
        ).after(predict_dt_job_run_op)

    with dsl.If(comp_model__op.output == "LR"):
        predict_lr_job_run_op = predict(
            model=training_lr_job_run_op.outputs["out_model"],
            features=pre_di_op.outputs["dataset"]
        )
        upload_model_lr_to_gc_op = upload_model_to_gcs(
            project_id=project_id,
            model_repo=model_repo,
            model=training_lr_job_run_op.outputs['out_model']
        ).after(predict_lr_job_run_op)


def compile_pipeline(target_file):
    compiler.Compiler().compile(pipeline_func=ml_pipeline,
                                package_path=target_file)


def _run_pipeline():
    PROJECT_ID = "deassignment1-511413"
    REGION = "us-central1"
    PIPELINE_ROOT = "gs://deassignment1_temp"

    job = aip.PipelineJob(
        display_name="california-housing-ct-pipeline",
        enable_caching=False,
        template_path="training_pipeline.yaml",
        pipeline_root=PIPELINE_ROOT,
        location=REGION,
        parameter_values={
            'project_id': PROJECT_ID,
            'data_bucket': 'deassignment1_data',
            'trainset_filename': 'housing.csv',
            'testset_filename': 'housing.csv',
            'model_repo': 'deassignment1_models'
        }
    )

    job.run()


def parse_command_line_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_file', type=str, default="training_pipeline.yaml", help="Pipeline Name")
    args = parser.parse_args()
    return vars(args)


if __name__ == '__main__':
    compile_pipeline(**parse_command_line_arguments())
    # _run_pipeline()