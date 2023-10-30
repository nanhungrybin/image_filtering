import os
from brisque_score import *


def list_files_in_directory(directory_path):
    """
    디렉토리 내의 파일 목록을 반환
    """
    file_list = []
    labeldir_path = []

    for root, dirs, files in os.walk(directory_path):
        labeldir_path.append(root)
        for file in files:
            file_list.append(file)

    return file_list, labeldir_path



def count_frame(directory_path, class_number):
    """
    디렉토리 내의 클래스별 파일 개수를 세고 반환
    """

    # class 별 파일 개수 저장
    data = []

    # vid 내의 디렉토리 목록 읽기 : ID
    video_ds = os.listdir(directory_path)

    # # vid 내의 디렉토리 목록 읽기
    # for root, dirs, files in os.walk(directory_path):
    #     video_ds = dirs  # 모든 하위 디렉토리의 목록

    # 디렉토리명을 ID로 매핑   # 5-60-1920x1080-2 
    id_mapping = {str(i): d for i, d in enumerate(video_ds)}

    for video_d in video_ds:
        
        video_d_path = os.path.join(directory_path, video_d)

        # 디렉토리 내의 label 디렉토리 목록 읽기 
        video_label_path = os.listdir(video_d_path)
        
        # 클래스별 파일 개수를 저장할 딕셔너리 초기화
        class_file_count = {str(c): 0 for c in class_number}

        for label_directory in video_label_path:
            label_path = os.path.join(video_d_path, label_directory)
            
            # label_directory의 이름이 class_numbers 안에 있는 숫자 중 하나와 일치하는 경우
            if int(label_directory) in class_number:

                # label 디렉토리 내의 파일 목록 읽기
                video_files , _= list_files_in_directory(label_path)
                file_count = len(video_files)

                class_file_count[label_directory] = file_count
                #아니면 0
                
                for frame_idx in video_files:
                    data_filter_anni, _ = brisque_classification(frame_idx, 0.78, class_number)


        data.append(class_file_count)

    return data


def count_filtering_anni(filepath, class_number):
    
    # class 별 파일 개수 저장
    data = []

    # 클래스별 파일 개수를 저장할 딕셔너리 초기화
    class_file_count = {str(c): 0 for c in class_number}

    # ex: /workspace/Data/vid/video3/-1/00403.jpg
    label_directory = filepath.split("/")[-2]

    # label_directory의 이름이 class_numbers 안에 있는 숫자 중 하나와 일치하는 경우
    if int(label_directory) in class_number:
        file_count = len(filepath)

        class_file_count[label_directory] = file_count

    data.append(class_file_count)

    return data
