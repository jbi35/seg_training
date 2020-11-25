import boto3
import csv
import os

def connect_s3():
    '''Connect to the s3 boto3 client.
    
    Returns:
        s3: Connection to the s3 boto3 client.
    '''
    # Connect to the s3 bucket
    s3 = boto3.client('s3')
    return s3

def download_file(path, s3, bucket, bucket_path, out_dir):
    '''Downloads a file from a s3 bucket into an output directory.
    
    Args:
        path (str):             Filename (specific path if necessary)
        s3 (s3):                Connection to s3 bucket
        bucket (str):           Bucket name
        bucket_path (str):      Base path in the bucket
        out_dir (str):          Path to the output directory
    '''
    # Download a file from s3 to the out_dir directory
    print("Downloading: ",path)
    s3.download_file(bucket, path, out_dir)

def download_from_s3(csv_file, img_base_path, out_dir):
    '''
    
    Args:
        csv_file (str):         Path to csv file
        img_base_path (str):    Base path within the bucket
        out_dir (str):          Path to the output directory
    
    Returns:
        array, array: Volume and segmentation paths
    '''
    # Define the output directories
    out_vol_dir = out_dir + 'vol'
    out_seg_dir = out_dir + 'seg'
    # Define the bucket and path
    bucket = 'my_name'
    # Connect to s3
    s3 = connect_s3()
    # Open csv and download files
    vol_paths = []
    seg_paths = []
    with open(csv_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            s3_vol_path = img_base_path + row[0]
            out_vol_path = os.path.join(out_vol_dir, os.path.basename(s3_vol_path))
            if not os.path.exists(out_vol_path):
                download_file(s3_vol_path, s3, bucket, img_base_path, out_dir = out_vol_path)
            else:
                print("Already exists", out_vol_path)
            vol_paths.append(out_vol_path)

            s3_seg_path = img_base_path + row[1]
            out_seg_path = os.path.join(out_seg_dir, os.path.basename(s3_seg_path))
            if not os.path.exists(out_seg_path):
                download_file(s3_seg_path, s3, bucket, img_base_path, out_dir = out_seg_path)
            else:
                print("Already exists", out_seg_path)
            seg_paths.append(out_seg_path)
    return vol_paths, seg_paths