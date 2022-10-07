#!/usr/bin/env python3
import yaml
import io
from pathlib import Path
import labelbox

# Opening the Labelbox secrets
def open_yaml(yaml_path):
    with open(yaml_path) as file:
        yaml_dict = yaml.safe_load(file)
    return yaml_dict

# Downloading the labels
def download_labels(api_key, project_id):
    # Enter your Labelbox API key here
    LB_API_KEY = api_key

    # Create Labelbox client
    lb = labelbox.Client(api_key=LB_API_KEY)

    # Get project by ID
    project = lb.get_project(project_id)

    # Export image and text data as an annotation generator:
    labels = project.label_generator()
    
    # Export labels created in the selected date range as a json file:
    labels = project.export_labels(download = True) 

    return labels

def main():
    # Getting the secrets
    yaml_path = Path.home() / '.credentials' / 'labelbox.yaml'
    labelbox_secrets = open_yaml(yaml_path)
    api_key = labelbox_secrets["api_key"]
    project_id = labelbox_secrets["project_id"]

    # Downloading the labels
    labels = download_labels(api_key, project_id)
    print(labels)

if __name__ == "__main__":
    main()
