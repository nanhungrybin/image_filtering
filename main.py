import os
import pandas as pd
from PIL import Image

from brisque_score import *
from count_frame import *




#######################################################

def makedf(directory_path, data, class_number):

    # vid 내의 디렉토리 목록 읽기
    video_ds = [d for d in os.listdir(directory_path)]

    # 디렉토리명을 ID로 매핑   # 5-60-1920x1080-2 
    id_mapping = {str(i): d for i, d in enumerate(video_ds)}

    # 결과 데이터를 DataFrame으로 변환
    df = pd.DataFrame(data)

    df.columns = class_number
    df.index = [id_mapping[str(i)] for i in range(0, len(id_mapping))]  # ID를 인덱스 이름으로 변경

    return df
 
######################################################





if __name__ == '__main__': 


    # 디렉토리 경로 설정
    directory_path = '/workspace/Data/vid_test'

    # class
    class_number = [0, 1, 2, 3, 4, 5, 6, 7]  

    ####### 1. origin #######

    data_origin = count_frame(directory_path, class_number)

    df_origin = makedf(directory_path, data_origin, class_number)
    df_origin.to_csv("origin.csv")
    print("origin done")

    ####### 2. brisque #######

    threshold = 0.78
    print(directory_path)
    data_filter_anni, _ = brisque_classification(directory_path, threshold, class_number)
    
    df_filter_anni = makedf(directory_path, data_filter_anni, class_number)
    df_filter_anni.to_csv("filtered_anni.csv")
    print("filtered_anni done")

    # annimation인 프레임 => ID별 anni 프레임의 경로(해당 프레임 이름포함)

    anni_data = brisque_classification(directory_path, threshold, class_number)[0]

    # 디렉토리명을 ID로 매핑
    id_mapping_annidata = {id: anni_data for id in os.listdir(directory_path)}
    df_id_mapping_annidata = makedf(directory_path, id_mapping_annidata, class_number)
    df_id_mapping_annidata.to_csv("id_annidata_path.csv")
    print("anni path done")
