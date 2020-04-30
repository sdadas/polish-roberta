import os
import zipfile

import requests
from tqdm import tqdm


def get_data(url: str, extract_dir: str, task_name: str, dataset_dir: str=None):
    if os.path.exists(dataset_dir if dataset_dir else extract_dir):
        print("Skipping %s, dataset already exists" % (task_name,))
        return
    print("Downloading %s" % (task_name,))
    file_size = int(requests.head(url).headers["Content-Length"])
    response = requests.get(url, stream=True)
    fname = task_name + ".zip"
    progress = tqdm(total=file_size, unit='B', unit_scale=True, desc=url.split('/')[-1])
    with open(fname, "wb") as zip_file:
        for data in response.iter_content(chunk_size=1024):
            zip_file.write(data)
            progress.update(1024)
    progress.close()
    with zipfile.ZipFile(fname, "r") as zip_file:
        zip_file.extractall(extract_dir)
    os.remove(fname)


if __name__ == '__main__':
    output_dir = "data"
    # KLEJ datasets
    KLEJ = "https://klejbenchmark.com/static/data/{}.zip"
    path = lambda p: os.path.join(output_dir, p)
    get_data(KLEJ.format("klej_nkjp-ner"), path("KLEJ/NKJP-NER"), "NKJP-NER")
    get_data(KLEJ.format("klej_cdsc-e"), path("KLEJ/CDSC-E"), "CDSC-E")
    get_data(KLEJ.format("klej_cdsc-r"), path("KLEJ/CDSC-R"), "CDSC-R")
    get_data(KLEJ.format("klej_cbd"), path("KLEJ/CBD"), "CBD")
    get_data(KLEJ.format("klej_polemo2.0-in"), path("KLEJ/POLEMO2.0-IN"), "POLEMO2.0-IN")
    get_data(KLEJ.format("klej_polemo2.0-out"), path("KLEJ/POLEMO2.0-OUT"), "POLEMO2.0-OUT")
    get_data(KLEJ.format("klej_dyk"), path("KLEJ/DYK"), "DYK")
    get_data(KLEJ.format("klej_psc"), path("KLEJ/PSC"), "PSC")
    get_data(KLEJ.format("klej_ar"), path("KLEJ/ECR"), "ECR")
    # Other datasets
    PSE = "https://github.com/sdadas/polish-sentence-evaluation/releases/download/datasets/{}.zip"
    get_data(PSE.format("8TAGS"), output_dir, "8TAGS", dataset_dir=os.path.join(output_dir, "8TAGS"))
    get_data(PSE.format("SICK"), output_dir, "SICK", dataset_dir=os.path.join(output_dir, "SICK"))
    get_data(PSE.format("WCCRS_HOTELS"), output_dir, "WCCRS_HOTELS", dataset_dir=os.path.join(output_dir, "WCCRS_HOTELS"))
    get_data(PSE.format("WCCRS_MEDICINE"), output_dir, "WCCRS_MEDICINE", dataset_dir=os.path.join(output_dir, "WCCRS_MEDICINE"))