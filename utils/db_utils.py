import os
import sqlite3
import json
import csv
from collections import defaultdict


def connect_db(path_to_db):
    ''' Adds a connection to a SQLite3 DB. '''
    conn = None
    try:
        conn = sqlite3.connect(path_to_db)  # connect to the sqlite3 file
    except Exception as e:
        print(e)
    c = conn.cursor()
    return c, conn

def extract_ct_scans(c, output_dir, disease, dataset, seg_type, num_scans, scan_data_format, seg_data_format):
    ''' Extract the ct scans based on the configuration file '''
    csv_path = os.path.join(output_dir, 'paths.csv')
    possible_datasets = ['exact', 'randomcovid19', 'structseg2019']
    for i in dataset:
        print(i)
        if any(i in s for s in possible_datasets):
            dataset_statement = "WHERE d.name IN (" + str(dataset)[1:-1] + ") "
            break
        elif i == "any":
            dataset_statement = "WHERE d.name IN (" + str(possible_datasets)[1:-1] + ") "
            break
        else:
            raise ValueError("Dataset does not exist.")

    if scan_data_format == "dicom":
        ct_path = "c.path_dicom"
    elif scan_data_format == "nifti":
        ct_path = "c.path_nifti"
    else:
        raise ValueError("Scan data format does not exist.")  
    
    if seg_data_format == "mhd":
        seg_path = "s.path_mhd, s.path_zraw"
    elif seg_data_format == "nifti":
        seg_path = "s.path_nifti"
    else:
        raise ValueError("Segmentation data format does not exist.")    
    
    possible_diseases = ['ARDS', 'Cancer', 'Control', 'COPD', 'ILD', 'COVID19', 'Other']
    if disease == [""]:
        disease_statement = ""
    else:
        for i in disease:
            # Only patients of one or multiple diseases
            print(i)
            if any(i in s for s in possible_diseases):
                disease_statement = "AND p.disease IN (" + str(disease)[1:-1] + ") "
                break
            # Only healthy patients
            elif i == "none":
                disease_statement = "AND p.disease = '' "
                break
            # Include all patients
            elif i == "any":
                disease_statement = "AND p.disease IN (" + str(possible_diseases)[1:-1] + ") "
                break
            # Only patients with specific primary diagnosis
            elif ("%" in i):
                disease_statement = "AND p.primary_clinical_diagnosis LIKE '" + i + "' "
                break
            # Else the disease is unkown
            else:
                raise ValueError("Disease condition does not exist.")
    
    possible_seg_types = ['lungs', 'left lung', 'right lung', 'vessels', 'airways', 'lobes', 'cancer', 'effucion', 'covid']
    for i in seg_type:
        if any(i in s for s in possible_seg_types):
            seg_statement = "AND s.type IN (" + str(seg_type)[1:-1] + ") "
            break
        elif i == "any":
            seg_statement = "AND s.type IN (" + str(possible_seg_types)[1:-1] + ") "
            break
        else:
            raise ValueError("Segmentation type does not exist.")

    if num_scans != 0:
        limit = "LIMIT " + str(num_scans)
    else:
        limit = ""
    
    request = "SELECT " + ct_path + " as ct_path, " + seg_path + " as seg_path, s.type FROM dataset d JOIN patients p ON d.id = p.dataset_id JOIN ct_scans c ON p.id = c.patient_id JOIN segmentations s ON c.id = s.ct_scan_id " + dataset_statement + disease_statement + seg_statement + "AND s.latest_version = 1 " + limit
    print(request)
    result = c.execute(request) 
    paths_dict = defaultdict(list)
    for ct_path, seg_path, type in result:
        paths_dict[ct_path].append((seg_path, type))
    return paths_dict