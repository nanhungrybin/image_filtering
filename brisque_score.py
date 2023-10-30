import os
from PIL import Image
from brisque import BRISQUE
from count_frame import *



###### for test ######

def brisque(path):

    img = Image.open(path)
    obj = BRISQUE(url=False)
    
    score = obj.score(img)

########################

def brisque_classification(frame_idx, threshold, class_number):

    """
    1. 파일을 읽고 threshold 이상이면 애니메이션으로 판단
    threshold 이하면 인간으로 판단

        2. 문제 있는 프레임의 넘버 (파일 명)
        3. 문제 있는 프레임의 경로
    """

    # 문제 있는 프레임 name
    yes_anni = []
    no_anni = []

    # anni filering 프레임 count
    anni_filtered_data = []

    # anni path
    anni_data = []   

    # /workspace/Data/vid_test
    # for root, file_list, dirs in os.walk(directory_path):
    #     print(root, dirs, file_list)
    #     # 파일명(FRAME)을 ID로 매핑
    #     id_file = {str(i): file for i, file in enumerate(file_list)} # 0: 00001.jpg
    #     for label_path in file_list:
    #         for file in os.listdir(os.path.join(root, label_path)):
    #             for file in os.listdir(os.path.join(root, label_path)):
    #                 file_path = os.path.join(root, label_path, file)  # 파일의 전체 경로

    img = Image.open(frame_idx)
    obj = BRISQUE(url=False)
    
    score = obj.score(img)

    # threshold 설정에 따른 분류

    # annimation
    if score >= threshold:
        # annimation 프레임 name
        yes_anni.append(id_file[str(i)] for i in range(len(file_list)))
        
        # annimation 프레임 path
        yes_anni_path = os.path.join(root, file)
        
        anni_data.append(yes_anni_path)

        
    else:
        # 문제 없는 프레임 name
        no_anni.append(id_file[str(i)] for i in range(len(file_list)))
        print("?")

        # 문제 없는 프레임 path
        no_anni_path = os.path.join(root, frame_idx)

        # 문제 없는 프레임 count
        anni_filtered_count = count_filtering_anni(no_anni_path, class_number)

        anni_filtered_data.append(anni_filtered_count)


    return anni_data, anni_filtered_data
