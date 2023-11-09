import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
from brisque import BRISQUE

import argparse
from config import get_config
from Learner import *
from utils import *

import time

################ animation filtering ##################

def brisque_classification(image_path, df):

    """
    1. 파일을 읽고 threshold 이상이면 애니메이션으로 판단
    threshold 이하면 인간으로 판단

    2. 문제 있는 프레임의 경로 list

    3. 문제 없는 프레임의 경로 list

    """

    # CSV 파일 읽기

    img = Image.open(image_path)
    obj = BRISQUE(url=False)

    score = obj.score(img)

    df.loc[df['path'] == image_path, 'out_animation'] = score

    # threshold 설정에 따른 분류


    return df


###################### ID tracking ######################

# extract embedding :아크페이스 loss를 사용한 백본 네트워크

def get_ID_model(conf): 

    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument('-th','--threshold',help='threshold to decide identical faces',default=1.54, type=float)
    args = parser.parse_args() #no need

    #model load => use_mobilfacenet or ir_se
    #head = Arcface 로 training 함
    learner = face_learner(conf, inference=True)

    learner.threshold = args.threshold

    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', from_save_folder=True, model_only=True)
    else:

        # save folder에서 pretained weights 가져오기
        # model_ir_se50.pth
        learner.load_state(conf, 'final.pth', from_save_folder=True, model_only=True)

    #learner.load_state(conf, 'final.pth', from_save_folder=True, model_only=True)


    # Evaluate mode. 
    Model = learner.model

    Model.eval()

    return Model


def ID_tracking(model, image_path, ID_start, list, n, df):

    #out_id = []  # Initialize: 여기안에 두면 함수 출력 할 때마다 초기화됨 즉 out_id는 메인함수에 두기

    if ID_start == True:

        img = Image.open(image_path)

        
        emb_base = model(conf.test_transform(img).to(conf.device).unsqueeze(0)) # initialize

        # GPU에서 CPU로 텐서 이동
        emb_base = emb_base.detach().to('cpu').numpy()

        
        list[n] = emb_base
       
        df['out_ID'] = list

        n = n+1


   

    return df, ID_start, n



    
#--------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__': 

    start_time = time.time()  # 코드 실행 시작 시간 기록


    # CSV 파일의 경로
    csv_file_path = '/workspace/Base.csv'

    df = pd.read_csv(csv_file_path)
    id_list = df['ID'].unique()
    

    # model load
    conf = get_config(False)
    model = get_ID_model(conf) 

    ###############

    cnt = 0     # each ID count
    count = 0   # each image count
    save_n = 0  # for count csv part
    n = 0
    list = [i for i in range(len(df))]

    part_size = 152758  # 분할 파일당 원하는 크기 설정 ##################################### change ###########################################
    prefix = 'mapping_filtering_Base_part_'

    ###############
    

    for idx_id in id_list:

        cnt += 1
        # 코드 실행 종료 시간 기록
        now_start_time = time.time()
        end_time = time.time()
        execution_time = end_time - start_time
        now_execution_time = end_time - now_start_time
        print(f"지금까지 총 실행 시간: {execution_time} 초 & 현재실행 시간: {now_execution_time} 초 & ...ing ", cnt/len(id_list)*100 , '%')
       
        ID_start = True

        for _, each_df in df[df['ID'] == idx_id].iterrows():

            image_path = each_df['path']
            label = each_df['label']

            ########## 2. brisque ###########
            df = brisque_classification(image_path, df)

            ######## 3. id tracking ###########

            df, ID_start, n = ID_tracking(model, image_path, ID_start, list, n, df)   # emb_base를 update하기 위해 return값으로 emb_base를 설정
            # ID_start를 update하기 위해 return값으로 ID_start
            
            count += 1

            # 중간 저장
            if count % part_size == 0:
                save_n += 1
                part_filename = prefix + str(save_n) + '.csv'
                df.to_csv('/workspace/' + part_filename, index=False)


    # 나머지 데이터를 마지막 파일로 저장
    if len(df) > 0:
        save_n += 1
        part_filename = prefix + str(save_n) + '.csv'
        df.to_csv('/workspace/' + part_filename, index=False)


    t_end_time = time.time()
    total_execution_time = t_end_time - start_time
    print(f"total 실행 시간: {total_execution_time} 초")
    print("done")
