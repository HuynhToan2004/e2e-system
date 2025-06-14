from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from datetime import datetime, timedelta
import mlflow
import os
import glob
import shutil
from itertools import product
import sys
import yaml

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from app.src.train import train
from app.src.test import evaluate_model

# MLflow tracking
MLFLOW_URI = "http://mlflow_server:5000"

# default_args = {
#     "start_date": datetime.now(),
#     "retries": 2,
#     "catchup": False
# }

def log_dataset_metadata(yaml_path: str) -> dict:
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    metadata = {}
    for key, value in config["dataset"].items():
        metadata[f"dataset_{key}"] = value

    return metadata

# Define the function to run DVC pull
def run_dvc_pull():
    original_dir = os.getcwd()
    print("project root:", project_root)
    print("original_dir: ",original_dir)
    # Change working directory to /app where .git and .dvc are located
    os.chdir('app')
    # Verify current directory for debugging
    print(f"Current working directory: {os.getcwd()}")
    # Run DVC pull command
    os.system('dvc pull')
    # Optional: Check if DVC pull was successful
    if os.path.exists('./src/dataset'):
        print("DVC pull successful, dataset directory exists")
    else:
        raise Exception("DVC pull failed, dataset directory not found")
    os.chdir(original_dir)
    print(f"Restored working directory: {os.getcwd()}")

def init_param_grid():
    return {
        "freeze_layers": [10, 15],
        "epochs": [5],
        "lr0": [0.01, 0.1]
    }

def prepare_pretrained_model(pretrained_model_path: str) -> str:
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("YOLOv8n_Finetune")

    if not os.path.exists(pretrained_model_path):
        with mlflow.start_run(run_name="initial_training"):
            model_name, config, metrics = train(
                pretrained_model_path=None,
                freeze_layers=0,
                epochs=1,
                lr0=0.01
            )
            mlflow.log_params(config)
            mlflow.log_metrics(metrics)

            latest_dir = max(glob.glob("runs/detect/train*"), key=os.path.getctime)
            model_path = os.path.join(latest_dir, "weights", f"{model_name}.pt")
            mlflow.log_artifact(model_path)
            return model_path
    return pretrained_model_path

def train_and_validate(param_grid: dict, pretrained_model_path: str):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("YOLOv8n_Finetune")

    metadata = log_dataset_metadata("app/src/data.yaml")
    combinations = list(product(*param_grid.values()))
    keys = list(param_grid.keys())
    model_info = []

    with mlflow.start_run(run_name="hyperparameter_tuning"):
        for idx, values in enumerate(combinations, 1):
            params = dict(zip(keys, values))

            with mlflow.start_run(run_name=f"train_{idx}", nested=True):
                mlflow.log_params(metadata)
                mlflow.log_params({**params, "pretrained_model": pretrained_model_path})

                model_name, config, val_metrics = train(pretrained_model_path, **params)
                mlflow.log_params(config)
                mlflow.log_metrics(val_metrics)

                train_dirs = glob.glob("runs/detect/train*")
                train_dir = max(train_dirs, key=os.path.getctime)
                model_path = os.path.join(train_dir, "weights", f"{model_name}.pt")

                if os.path.exists(model_path):
                    mlflow.log_artifact(model_path)

                model_info.append({
                    "run_name": f"train_{idx}",
                    "params": config,
                    "model_name": model_name,
                    "model_path": model_path
                })

    return model_info

def evaluate_models(model_info: list, metric_to_optimize="mAP50-95"):
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("YOLOv8n_Finetune")

    metadata = log_dataset_metadata("app/src/data.yaml")
    best_model, best_score, best_params = None, -float("inf"), None

    with mlflow.start_run(run_name="hyperparameter_tuning_eval"):
        for idx, info in enumerate(model_info, 1):
            model_path = info["model_path"]
            if not os.path.exists(model_path):
                continue

            with mlflow.start_run(run_name=f"eval_{idx}", nested=True):
                mlflow.log_params(metadata)
                mlflow.log_params(info["params"])
                mlflow.log_param("model_name", info["model_name"])

                metrics = evaluate_model(model_path)
                mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})

                score = metrics.get(metric_to_optimize, -1)
                if score > best_score:
                    best_model, best_score, best_params = model_path, score, info["params"]

    return {
        "best_model": best_model,
        "best_score": best_score,
        "best_params": best_params
    }

def save_best_model(result: dict, metric_to_optimize="mAP50-95"):
    if not result["best_model"]:
        return

    metadata = log_dataset_metadata("app/src/data.yaml")
    final_path = os.path.join("runs/detect", "best_model.pt")
    shutil.copy(result["best_model"], final_path)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("YOLOv8n_Finetune")

    with mlflow.start_run(run_name="best_model"):
        mlflow.log_params(metadata)
        mlflow.log_params(result["best_params"])
        mlflow.log_metrics({metric_to_optimize: result["best_score"]})
        mlflow.log_param("model_name", os.path.basename(result["best_model"]))
        mlflow.log_artifact(final_path)

# Default arguments cho DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


with DAG(
    "lp_yolov8_pipeline",
    schedule_interval=timedelta(minutes=30),
    default_args=default_args,
    tags=["yolo", "mlflow"],
    description="License Plate YOLOv8 Pipeline with Airflow",
    start_date=datetime(2025, 6, 11),
    catchup=False,
) as dag:
    # Task 1: Chạy DVC pull
    dvc_pull_task = PythonOperator(
        task_id='run_dvc_pull',
        python_callable=run_dvc_pull,
    )

    # Task 2: Khởi tạo tham số
    init_param_task = PythonOperator(
        task_id='init_param_grid',
        python_callable=init_param_grid,
    )

    # Task 3: Chuẩn bị mô hình pretrained
    prepare_model_task = PythonOperator(
        task_id='prepare_pretrained_model',
        python_callable=prepare_pretrained_model,
        op_kwargs={'pretrained_model_path': 'runs/detect/train/weights/yolov8n_v1.pt'},
    )

    # Task 4: Huấn luyện và xác thực
    train_validate_task = PythonOperator(
        task_id='train_and_validate',
        python_callable=train_and_validate,
        op_kwargs={'param_grid': init_param_task.output, 'pretrained_model_path': prepare_model_task.output},
    )

    # Task 5: Đánh giá mô hình
    evaluate_task = PythonOperator(
        task_id='evaluate_models',
        python_callable=evaluate_models,
        op_kwargs={'model_info': train_validate_task.output, 'metric_to_optimize': 'mAP50-95'},
    )

    # Task 6: Lưu mô hình tốt nhất
    save_model_task = PythonOperator(
        task_id='save_best_model',
        python_callable=save_best_model,
        op_kwargs={'result': evaluate_task.output, 'metric_to_optimize': 'mAP50-95'},
    )

    # Định nghĩa luồng
    dvc_pull_task >> init_param_task >> prepare_model_task >> train_validate_task >> evaluate_task >> save_model_task




    # @task()
    # def init_param_grid():
    #     return {
    #         "freeze_layers": [10, 15],
    #         "epochs": [5],
    #         "lr0": [0.01, 0.1]
    #     }
    #
    # @task()
    # def prepare_pretrained_model(pretrained_model_path: str) -> str:
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     mlflow.set_experiment("YOLOv8n_Finetune")
    #
    #     if not os.path.exists(pretrained_model_path):
    #         with mlflow.start_run(run_name="initial_training"):
    #             model_name, config, metrics = train(
    #                 pretrained_model_path=None,
    #                 freeze_layers=0,
    #                 epochs=1,
    #                 lr0=0.01
    #             )
    #             mlflow.log_params(config)
    #             mlflow.log_metrics(metrics)
    #
    #             latest_dir = max(glob.glob("runs/detect/train*"), key=os.path.getctime)
    #             model_path = os.path.join(latest_dir, "weights", f"{model_name}.pt")
    #             mlflow.log_artifact(model_path)
    #             return model_path
    #     return pretrained_model_path
    #
    # @task()
    # def train_and_validate(param_grid: dict, pretrained_model_path: str):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     mlflow.set_experiment("YOLOv8n_Finetune")
    #
    #     metadata = log_dataset_metadata("app/src/data.yaml")
    #     combinations = list(product(*param_grid.values()))
    #     keys = list(param_grid.keys())
    #     model_info = []
    #
    #     with mlflow.start_run(run_name="hyperparameter_tuning"):
    #         for idx, values in enumerate(combinations, 1):
    #             params = dict(zip(keys, values))
    #
    #             with mlflow.start_run(run_name=f"train_{idx}", nested=True):
    #                 mlflow.log_params(metadata)
    #                 mlflow.log_params({**params, "pretrained_model": pretrained_model_path})
    #
    #                 model_name, config, val_metrics = train(pretrained_model_path, **params)
    #                 mlflow.log_params(config)
    #                 mlflow.log_metrics(val_metrics)
    #
    #                 train_dirs = glob.glob("runs/detect/train*")
    #                 train_dir = max(train_dirs, key=os.path.getctime)
    #                 model_path = os.path.join(train_dir, "weights", f"{model_name}.pt")
    #
    #                 if os.path.exists(model_path):
    #                     mlflow.log_artifact(model_path)
    #
    #                 model_info.append({
    #                     "run_name": f"train_{idx}",
    #                     "params": config,
    #                     "model_name": model_name,
    #                     "model_path": model_path
    #                 })
    #
    #     return model_info
    #
    # @task()
    # def evaluate_models(model_info: list, metric_to_optimize="mAP50-95"):
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     mlflow.set_experiment("YOLOv8n_Finetune")
    #
    #     metadata = log_dataset_metadata("app/src/data.yaml")
    #     best_model, best_score, best_params = None, -float("inf"), None
    #
    #     with mlflow.start_run(run_name="hyperparameter_tuning_eval"):
    #         for idx, info in enumerate(model_info, 1):
    #             model_path = info["model_path"]
    #             if not os.path.exists(model_path):
    #                 continue
    #
    #             with mlflow.start_run(run_name=f"eval_{idx}", nested=True):
    #                 mlflow.log_params(metadata)
    #                 mlflow.log_params(info["params"])
    #                 mlflow.log_param("model_name", info["model_name"])
    #
    #                 metrics = evaluate_model(model_path)
    #                 mlflow.log_metrics({f"eval_{k}": v for k, v in metrics.items()})
    #                 # mlflow.log_artifact(model_path)
    #
    #                 score = metrics.get(metric_to_optimize, -1)
    #                 if score > best_score:
    #                     best_model, best_score, best_params = model_path, score, info["params"]
    #
    #     return {
    #         "best_model": best_model,
    #         "best_score": best_score,
    #         "best_params": best_params
    #     }
    #
    # @task()
    # def save_best_model(result: dict, metric_to_optimize="mAP50-95"):
    #     if not result["best_model"]:
    #         return
    #
    #     metadata = log_dataset_metadata("app/src/data.yaml")
    #     final_path = os.path.join("runs/detect", "best_model.pt")
    #     shutil.copy(result["best_model"], final_path)
    #
    #     mlflow.set_tracking_uri(MLFLOW_URI)
    #     mlflow.set_experiment("YOLOv8n_Finetune")
    #
    #     with mlflow.start_run(run_name="best_model"):
    #         mlflow.log_params(metadata)
    #         mlflow.log_params(result["best_params"])
    #         mlflow.log_metrics({metric_to_optimize: result["best_score"]})
    #         mlflow.log_param("model_name", os.path.basename(result["best_model"]))
    #         mlflow.log_artifact(final_path)
    #
    # # DAG Flow
    # param_grid = init_param_grid()
    # pretrained = prepare_pretrained_model("runs/detect/train/weights/yolov8n_v1.pt")
    # model_info = train_and_validate(param_grid, pretrained)
    # result = evaluate_models(model_info)
    # save_best_model(result)
