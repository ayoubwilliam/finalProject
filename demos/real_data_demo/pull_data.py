import pandas as pd
import urllib.request
import os
import ssl
import json

# Bypass SSL verification issues on some local machines
ssl._create_default_https_context = ssl._create_unverified_context


def download_clinical_pairs():
    print("Downloading metadata from IEEE8023 COVID Dataset...")
    csv_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/metadata.csv"
    base_image_url = "https://raw.githubusercontent.com/ieee8023/covid-chestxray-dataset/master/images/"

    df = pd.read_csv(csv_url)

    # 1. Filter for X-rays only
    df = df[df['modality'] == 'X-ray']

    # 2. Clean text for searching (FIXED: 'clinical_notes' uses an underscore)
    df['clinical_notes'] = df['clinical_notes'].fillna('').str.lower()
    df['finding'] = df['finding'].fillna('').str.lower()
    df['intubated'] = df['intubated'].fillna('').str.upper()

    # 3. Tag rows containing our requested pathologies
    df['has_consolidation'] = df['clinical_notes'].str.contains('consolidat') | df['finding'].str.contains('consolidat')
    df['has_tube'] = (df['intubated'] == 'Y') | df['clinical_notes'].str.contains('intubat|tube|ventilat')

    # 4. Group by patient and find longitudinal pairs with these traits
    valid_patients = []
    for patient, group in df.groupby('patientid'):
        if len(group) > 1:  # Must have multiple visits (Prior + Current)
            # Patient must exhibit consolidation and require a tube at some point in their timeline
            if group['has_consolidation'].any() and group['has_tube'].any():
                valid_patients.append(patient)

    multi_visit = df[df['patientid'].isin(valid_patients)].sort_values(by=['patientid', 'offset'])

    # 5. Download the images locally
    output_dir = "consolidation_tube_pairs"
    os.makedirs(output_dir, exist_ok=True)

    downloaded_pairs = 0

    print(f"Found {len(valid_patients)} patients matching criteria. Downloading images...\n")

    for patient, group in multi_visit.groupby('patientid'):
        patient_dir = os.path.join(output_dir, str(patient))
        os.makedirs(patient_dir, exist_ok=True)

        # Save images sequentially (Prior -> Current -> Follow-up)
        for idx, row in enumerate(group.itertuples()):
            filename = row.filename
            img_url = base_image_url + filename
            save_path = os.path.join(patient_dir, f"visit_{idx}_{filename}")

            try:
                urllib.request.urlretrieve(img_url, save_path)
                print(f"Saved: {save_path} (Tube: {row.has_tube}, Consolidation: {row.has_consolidation})")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")

        # Save the clinical notes as the "Changes" ground truth
        # Using fillna("unknown") ensures that any empty Pandas values don't crash the JSON parser
        notes = group[['offset', 'clinical_notes', 'intubated']].fillna("unknown").to_dict('records')

        with open(os.path.join(patient_dir, "clinical_changes.json"), "w") as f:
            json.dump(notes, f, indent=2)

        downloaded_pairs += 1

    print(f"\n✅ Finished downloading {downloaded_pairs} patient timelines.")


if __name__ == "__main__":
    download_clinical_pairs()