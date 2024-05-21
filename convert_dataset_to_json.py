import json
import os


def load_directory(base_folder: str, output_file_name: str):

    files = []

    for folder in os.listdir(base_folder):
        for file in os.listdir(f"{base_folder}/{folder}"):
            with open(f"{base_folder}/{folder}/{file}") as f:
                file_content = f.read()
                files.append({"text": file_content, "category": folder})

    with open(f"{output_file_name}.json", "w", encoding="utf-8") as output_file:
        json.dump(files, output_file, indent=2)


load_directory("training", "training")
load_directory("testing", "testing")
