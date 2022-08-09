import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv

from utils.namespace import Namespace


def reset_environment(environment=None):
    if not environment:
        load_dotenv()
        environment = os.getenv("LIVER_ENVIRONMENT", "colab")

    if environment == "local":
        sources_path = "/mnt/chromeos/MyFiles/Downloads"
        liver_path = "/home/yamatteo/liver"
        dataset_path = "/home/yamatteo/dataset"
        models_path = "/home/yamatteo/saved_models"
    elif environment == "gce":
        sources_path = "/content/drive/MyDrive/COLAB"
        liver_path = "/content/liver"
        dataset_path = "/content/dataset"
        models_path = "/content/saved_models"
    elif environment == "colab":
        from google.colab import drive
        drive.mount("/content/drive")
        sources_path = "/content/drive/MyDrive/COLAB"
        liver_path = "/content/liver"
        dataset_path = "/content/dataset"
        models_path = "/content/saved_models"
    else:
        raise ValueError(f"Unexpected environemnt {environment!r}")

    print(f"Resetting {environment} enviroment in folder {os.getcwd()}")
    with open(os.path.join(os.getcwd(), ".env"), "w") as f:
        f.write(f'LIVER_ENVIRONMENT={environment!r}\n')
        f.write(f'SOURCES_PATH={sources_path!r}\n')
        f.write(f'LIVER_PATH={liver_path!r}\n')
        f.write(f'DATASET_PATH={dataset_path!r}\n')
        f.write(f'MODELS_PATH={models_path!r}\n')
    assert Path(sources_path).exists(), f"Path for sources_path ({sources_path}) should exists."
    if not Path(liver_path).exists():
        Path(liver_path).mkdir(parents=True)
        subprocess.run(f"git clone https://github.com/yamatteo/liver.git {liver_path}".split())
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    Path(models_path).mkdir(parents=True, exist_ok=True)
    update_environment(liver_path)
    subprocess.run("pip install -r requirements.txt".split(), cwd=liver_path)
    subprocess.run("wandb login 6e44b780601adbcb29c5322b462b5da80078222e".split())


def update_environment(liver_path):
    subprocess.run("git fetch origin july_rewrite".split(), cwd=liver_path)
    subprocess.run("git checkout july_rewrite".split(), cwd=liver_path)
    subprocess.run("git pull origin july_rewrite".split(), cwd=liver_path)


def get_env():
    load_dotenv()
    if os.getenv("LIVER_ENVIRONMENT") not in ["local", "gce", "colab"]:
        raise ImportError("Environment is not set up.")

    opts = Namespace(
        sources_path=Path(os.getenv("SOURCES_PATH")),
        liver_path=Path(os.getenv("LIVER_PATH")),
        dataset_path=Path(os.getenv("DATASET_PATH")),
        models_path=Path(os.getenv("MODELS_PATH")),
    )
    for name, path in vars(opts).items():
        assert path.exists(), f"Path for {name} ({path}) does not exist."

    return opts
