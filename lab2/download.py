import argparse
import os

import requests
from tqdm import tqdm

FILES = [
    ('1TTEJLkcy0mIt23toLnCpoXsKh-522dc8', 'dataset.zip'),
    ('1sXvQbGulF-yZdzChyPGulUTg7rO7TZZE', 'pretrain.zip'),
]
URL = "https://docs.google.com/uc?export=download"
CHUNK_SIZE = 32768


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None


def main(args):
    for file_id, file_name in FILES:
        with open(os.path.join(args.dir, file_name), 'wb') as fout:
            with tqdm(unit="B", unit_scale=True, dynamic_ncols=True) as pbar:
                pbar.set_description(file_name)
                session = requests.Session()
                response = session.get(
                    URL, params={'id': file_id}, stream=True)
                token = get_confirm_token(response)
                if token:
                    params = {'id': file_id, 'confirm': token}
                    response = session.get(URL, params=params, stream=True)

                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:
                        fout.write(chunk)
                        pbar.update(len(chunk))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='./')
    main(parser.parse_args())
