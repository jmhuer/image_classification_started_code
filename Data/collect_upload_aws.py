import boto3 as boto3
import cv2
import time
import csv
import argparse
import os
import boto3
from .config import aws_info
'''
get cnt here is optional
    Unessesary function that finds cnt_number per label.
    Requires folder with images named: label_number.jpg. otherwise cnt = 0
'''

#for more info
# https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3-uploading-files.html
def upload_s3_file(filename,aws_filename):
    s3_client = boto3.client('s3',
                             aws_access_key_id = aws_info['aws_access_key_id'],
                             region_name = aws_info['region_name']
                             ,aws_secret_access_key = aws_info['aws_secret_access_key'])
    s3_client.upload_file(filename, aws_info['BUCKET_NAME'], '%s%s'%("new",aws_filename))
    print("AWS upload done")
    """
    Try this for upload to specific folder
    s3_client.upload_file(file_path,bucket_name, '%s%s'%(bucket_folder,destination_file_name))
    if this works, un-comment line 109 in upload.py
    """

def get_cnt(DATASET, label):
    cnt = 0
    files = os.listdir(DATASET)
    paths = [os.path.join(DATASET, basename) for basename in files]
    if len(paths) == 0: return 0
    for p in paths:
        if p.endswith(".jpg"):
            path_label = int(p[p.rfind('/') + 1])
            if label == path_label:
                new_cnt = int(p[p.rfind('_') + 1: p.rfind('.')])
                if cnt < new_cnt:
                    cnt = new_cnt
    return cnt + 1  ##we name the next image cnt + 1


def collect(camera_source, folder_path, label):
    cap = cv2.VideoCapture(camera_source)
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    ##because we append we need to clear existing csv
    if not os.path.exists(folder_path):
        os.makedirs(os.path.dirname(folder_path), exist_ok=True)
        cnt = 0  ##number of images in folder
    else:
        cnt = get_cnt(folder_path, label)

    while True:
        ret, image = cap.read()
        image = cv2.resize(image, (256, 200))  ##maybe i should convert to pil and save. to allow me to use transforms
        cv2.imshow("test", image)
        k = cv2.waitKey(1)
        if not ret:
            break
        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        # elif k % 256 == 32:
        else:
            # SPACE pressed
            path = folder_path + str(label) + "_" + str(cnt) + ".jpg"
            data = []
            data.append([path, label])
            with open(folder_path + "labels.csv", "a") as f:
                writer = csv.writer(f)
                writer.writerows(data)
            cv2.imwrite(path, image)
            print("{} written!".format(path))
            time.sleep(0.5)
            cnt += 1

        upload_s3_file(path, str(cnt) + ".jpg")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    import argparse

    '''
    We default to most recent model created
    '''

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-s', '--camera_source', type=int, default=0)
    parser.add_argument('-f', '--folder_path', type=str, default="Data/images/")
    parser.add_argument('-c', '--class_int', type=int, required=True)

    args = parser.parse_args()

    print('Writing in folder : ', args.folder_path)

    collect(args.camera_source, args.folder_path, args.class_int)
