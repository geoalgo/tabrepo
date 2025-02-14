import os
from pathlib import Path

from huggingface_hub import HfApi
from tqdm import tqdm


def upload_hugging_face(version: str, repo_id: str, override_existing_files: bool = True):
    """
    Uploads tabrepo data to Hugging Face repository.
    You should set your env variable HF_TOKEN and ask write access to tabrepo before using the script.

    Args:
        version (str): The version of the data to be uploaded, the folder data/results/{version}/ should
        be present and should contain baselines.parquet, configs.parquet and model_predictions/ folder
        repo_id (str): The ID of the Hugging Face repository.
        override_existing_files (bool): Whether to re-upload files if they are already found in HuggingFace.
    Returns:
        None
    """
    commit_message = f"Upload tabrepo new version"
    root = Path(__file__).parent.parent.parent / f"data/results/{version}/"

    for filename in ["baselines.parquet", "configs.parquet", "model_predictions"]:
        assert (root / filename).exists(), f"Expected to found {filename} but could not be found in {root / filename}."
    api = HfApi()
    for filename in ["baselines.parquet", "configs.parquet"]:
        path_in_repo = str(Path(version) / filename)
        if api.file_exists(repo_id=repo_id, filename=path_in_repo, token=os.getenv("HF_TOKEN"), repo_type="dataset") and not override_existing_files:
            print(f"Skipping {path_in_repo} which already exists in the repo.")
            continue

        api.upload_file(
            path_or_fileobj=root / filename,
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="dataset",
            commit_message=commit_message,
            token=os.getenv("HF_TOKEN"),
        )
    files = list(sorted(Path(root / "model_predictions").glob("*")))
    for dataset_path in tqdm(files):
        print(dataset_path)
        try:
            path_in_repo = str(Path(version) / "model_predictions" / dataset_path.name)
            # ideally, we would just check if the folder exists but it is not possible AFAIK, we could alternatively
            # upload per file but it would create a lot of different commits.
            if api.file_exists(repo_id=repo_id, filename=str(Path(path_in_repo) / "0" / "metadata.json"), token=os.getenv("HF_TOKEN"),
                               repo_type="dataset") and not override_existing_files:
                print(f"Skipping {path_in_repo} which already exists in the repo.")
                continue
            api.upload_folder(
                folder_path=dataset_path,
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns="*DS_Store",
                commit_message=f"Upload tabrepo new version {dataset_path.name}",
                token=os.getenv("HF_TOKEN"),
            )
        except Exception as e:
            print(str(e))


def download_from_huggingface(
        version: str,
        force_download: bool = False,
        local_files_only: bool = False,
        datasets: list[str] | None = None,
):
    """
    :param local_files_only: whether to use local files with no internet check on the Hub
    :param force_download: forces files to be downloaded
    :return:
    """
    # https://huggingface.co/datasets/Tabrepo/tabrepo/tree/main/2023_11_14/model_predictions
    api = HfApi()
    local_dir = str(Path(__file__).parent.parent.parent / "data/results/")
    print(f"Going to download tabrepo files to {local_dir}.")
    if datasets is None:
        allow_patterns = None
    else:
        allow_patterns = [f"*{version}*{d}*" for d in datasets]
    print(allow_patterns)
    api.snapshot_download(
        repo_id="Tabrepo/tabrepo",
        repo_type="dataset",
        allow_patterns=allow_patterns,
        local_dir=str(Path(__file__).parent.parent.parent / "data/results/"),
        force_download=force_download,
        local_files_only=local_files_only,
    )
if __name__ == '__main__':
    # upload_hugging_face(
    #     version="2023_11_14",
    #     repo_id="tabrepo/tabrepo",
    #     override_existing_files=False,
    # )
    datasets = [
        'Australian',
        'Bioresponse', 'GAMETES_Epistasis_2-Way_1000atts_0_4H_EDM-1_EDM-1_1', 'GAMETES_Epistasis_2-Way_20atts_0_1H_EDM-1_1', 'GAMETES_Epistasis_2-Way_20atts_0_4H_EDM-1_1', 'GAMETES_Epistasis_3-Way_20atts_0_2H_EDM-1_1', 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_50_EDM-2_001', 'GAMETES_Heterogeneity_20atts_1600_Het_0_4_0_2_75_EDM-2_001', 'GesturePhaseSegmentationProcessed', 'Indian_pines', 'Internet-Advertisements', 'LED-display-domain-7digit', 'MiceProtein', 'OVA_Colon', 'OVA_Endometrium', 'OVA_Kidney', 'OVA_Lung', 'OVA_Ovary', 'OVA_Prostate', 'Satellite', 'SpeedDating', 'Titanic', 'ada', 'analcatdata_authorship', 'analcatdata_dmft', 'arcene', 'arsenic-female-bladder', 'autoUniv-au1-1000', 'autoUniv-au6-750', 'autoUniv-au7-1100', 'autoUniv-au7-700', 'balance-scale', 'bank32nh', 'bank8FM', 'baseball', 'blood-transfusion-service-center', 'boston_corrected', 'car', 'cardiotocography', 'churn', 'climate-model-simulation-crashes', 'cmc', 'cnae-9', 'colleges_usnews', 'cpu_act', 'cpu_small', 'credit-g', 'cylinder-bands', 'delta_ailerons', 'delta_elevators', 'diabetes', 'dna', 'dresses-sales', 'eucalyptus', 'fabert', 'first-order-theorem-proving', 'fri_c0_1000_5', 'fri_c0_500_5', 'fri_c1_1000_50', 'fri_c2_1000_25', 'fri_c2_500_50', 'fri_c3_1000_10', 'fri_c3_1000_25', 'fri_c3_500_10', 'fri_c3_500_50', 'fri_c4_500_100', 'gina', 'hill-valley', 'hiva_agnostic', 'ilpd', 'jasmine', 'kc1', 'kc2', 'kdd_el_nino-small', 'kin8nm', 'led24', 'madeline', 'madelon', 'mc1', 'meta', 'mfeat-factors', 'no2', 'optdigits', 'ozone-level-8hr', 'page-blocks', 'parity5_plus_5', 'pbcseq', 'pc1', 'pc2', 'pc3', 'pc4', 'philippine', 'phoneme', 'pm10', 'pollen', 'puma32H', 'puma8NH', 'qsar-biodeg', 'ringnorm', 'rmftsa_ladata', 'satimage', 'segment', 'semeion', 'spambase', 'splice', 'steel-plates-fault', 'sylvine', 'synthetic_control', 'tokyo1', 'twonorm', 'vehicle', 'visualizing_soil', 'volcanoes-a2', 'volcanoes-a3', 'volcanoes-a4', 'volcanoes-b5', 'volcanoes-d1', 'volcanoes-d4', 'volcanoes-e1', 'wall-robot-navigation', 'waveform-5000', 'wilt', 'wind', 'wine-quality-red', 'wine-quality-white']
    download_from_huggingface(
        datasets=datasets,
        version="2023_11_14",
    )