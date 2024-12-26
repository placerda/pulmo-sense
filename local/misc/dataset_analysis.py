import os
import json
import numpy as np
import pandas as pd

def summarize_mosmed(root_dir):
    studies = os.listdir(os.path.join(root_dir, 'images'))
    num_studies_covid = 0
    num_studies_non_covid = 0
    total_slices_covid = 0
    total_slices_non_covid = 0
    min_slices_covid = float('inf')
    max_slices_covid = 0
    min_slices_non_covid = float('inf')
    max_slices_non_covid = 0

    for study in studies:
        image_path = os.path.join(root_dir, 'images', study, 'image.npy')
        label_path = os.path.join(root_dir, 'covid_labels', study, 'covid_label.json')

        if os.path.exists(image_path) and os.path.exists(label_path):
            slices = np.load(image_path).shape[2]
            with open(label_path, 'r') as label_file:
                label = json.load(label_file)
                if label == True:
                    num_studies_covid += 1
                    total_slices_covid += slices
                    min_slices_covid = min(min_slices_covid, slices)
                    max_slices_covid = max(max_slices_covid, slices)
                else:
                    num_studies_non_covid += 1
                    total_slices_non_covid += slices
                    min_slices_non_covid = min(min_slices_non_covid, slices)
                    max_slices_non_covid = max(max_slices_non_covid, slices)

    avg_slices_covid = round(total_slices_covid / num_studies_covid) if num_studies_covid > 0 else 0
    avg_slices_non_covid = round(total_slices_non_covid / num_studies_non_covid) if num_studies_non_covid > 0 else 0
    min_slices_covid = round(min_slices_covid) if num_studies_covid > 0 else 0
    max_slices_covid = round(max_slices_covid) if num_studies_covid > 0 else 0
    min_slices_non_covid = round(min_slices_non_covid) if num_studies_non_covid > 0 else 0
    max_slices_non_covid = round(max_slices_non_covid) if num_studies_non_covid > 0 else 0

    return [
        {
            'Dataset': 'MosMed - COVID Cases',
            'Num Studies': num_studies_covid,
            'Avg Slices per Study': avg_slices_covid,
            'Min Slices per Study': min_slices_covid,
            'Max Slices per Study': max_slices_covid
        },
        {
            'Dataset': 'MosMed - Normal Cases',
            'Num Studies': num_studies_non_covid,
            'Avg Slices per Study': avg_slices_non_covid,
            'Min Slices per Study': min_slices_non_covid,
            'Max Slices per Study': max_slices_non_covid
        }
    ]

def summarize_covidctmd(root_dir):
    categories = {
        'Cap Cases': 'Pneumonia',
        'COVID-19 Cases': 'COVID-19',
        'Normal Cases': 'Normal'
    }
    summary = []

    for category, label in categories.items():
        studies = os.listdir(os.path.join(root_dir, category))
        num_studies = len(studies)
        total_slices = 0
        min_slices = float('inf')
        max_slices = 0

        for study in studies:
            slices = os.listdir(os.path.join(root_dir, category, study))
            num_slices = len(slices)
            total_slices += num_slices
            min_slices = min(min_slices, num_slices)
            max_slices = max(max_slices, num_slices)

        avg_slices = round(total_slices / num_studies) if num_studies > 0 else 0
        min_slices = round(min_slices) if num_studies > 0 else 0
        max_slices = round(max_slices) if num_studies > 0 else 0

        summary.append({
            'Dataset': f'CovidCTMD - {label}',
            'Num Studies': num_studies,
            'Avg Slices per Study': avg_slices,
            'Min Slices per Study': min_slices,
            'Max Slices per Study': max_slices
        })

    return summary

def summarize_luna16(root_dir):
    subsets = [f'subset{i}' for i in range(10)]
    num_studies = 0
    total_slices = 0
    min_slices = float('inf')
    max_slices = 0

    for subset in subsets:
        studies = [f for f in os.listdir(os.path.join(root_dir, subset)) if f.endswith('.mhd')]
        num_studies += len(studies)

        for study in studies:
            mhd_path = os.path.join(root_dir, subset, study)
            with open(mhd_path, 'r') as mhd_file:
                for line in mhd_file:
                    if line.startswith('DimSize ='):
                        slices = int(line.split('=')[1].strip().split()[2])
                        total_slices += slices
                        min_slices = min(min_slices, slices)
                        max_slices = max(max_slices, slices)
                        break

    avg_slices = round(total_slices / num_studies) if num_studies > 0 else 0
    min_slices = round(min_slices) if num_studies > 0 else 0
    max_slices = round(max_slices) if num_studies > 0 else 0

    return {
        'Dataset': 'LUNA16 - Nodules',
        'Num Studies': num_studies,
        'Avg Slices per Study': avg_slices,
        'Min Slices per Study': min_slices,
        'Max Slices per Study': max_slices
    }

def main():
    mosmed_root = 'data/processed/mosmed'
    covidctmd_root = 'data/raw/COVID-CT-MD'
    luna16_root = 'data/raw/Luna16'

    mosmed_summary = summarize_mosmed(mosmed_root)
    covidctmd_summary = summarize_covidctmd(covidctmd_root)
    luna16_summary = summarize_luna16(luna16_root)

    summary_data = mosmed_summary + covidctmd_summary + [luna16_summary]
    df = pd.DataFrame(summary_data)

    print("Dataset Summary:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
