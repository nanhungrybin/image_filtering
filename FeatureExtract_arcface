import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
import imquality.brisque as BRISQUE

import argparse
from config import get_config
from Learner import *
from utils import *
import os
from numpy.linalg import norm
import time

################ animation filtering ##################

def brisque_classification(image_path, df):

    """
    1. 파일을 읽고 threshold 이상이면 퀄리티 낮은 이미지로 판단
    threshold 이하면 낮은 퀄리티

    2. 문제 있는 프레임의 경로 list

    3. 문제 없는 프레임의 경로 list

    """

    # CSV 파일 읽기

    img = Image.open(image_path)
    obj = BRISQUE(url=False)

    score = obj.score(img)

    df.loc[df['PATH'] == image_path, 'Brisque'] = score

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

    # learner.load_state(conf, 'final.pth', from_save_folder=True, model_only=True)


    # Evaluate mode. 
    Model = learner.model

    Model.eval()

    return Model



def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = norm(embedding1)
    norm2 = norm(embedding2)
    
    similarity = dot_product / (norm1 * norm2)
    
    return similarity



def ID_tracking(model, image_path, ID_start, list, n, df):

    #out_id = []  # Initialize: 여기안에 두면 함수 출력 할 때마다 초기화됨 즉 out_id는 메인함수에 두기

    if ID_start == True:

        img = Image.open(image_path)

        # 현재 이미지에 대한 임베딩 생성
        emb_current = model(conf.test_transform(img).to(conf.device).unsqueeze(0)).detach().to('cpu').numpy()

        
        # 이전 이미지에 대한 임베딩이 존재하면 코사인 유사도 계산
        if n > 0:
            emb_base = list[n-1]  # 이전 이미지에 대한 임베딩
            similarity = cosine_similarity(emb_base, emb_current)[0, 0]
        else:
            similarity = None

        # 결과를 DataFrame에 추가
        df = df.append({"COS_similarity": similarity}, ignore_index=True)

        # 리스트에 현재 이미지의 임베딩 추가
        list.append(emb_current)

        # ID_start 상태 변경
        ID_start = False

   

    return df, ID_start, n


    
#--------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__': 

    start_time = time.time()  # 코드 실행 시작 시간 기록

    # 저장할 CSV 파일의 경로
    csv_file_path = # 0214_cropped_affwild2의 모든 csv concat하기 '\\workspace\\0123_base.csv'

    df = pd.read_csv(csv_file_path)  # 크로핑 결과 정보 담긴

    # model load
    conf = get_config(False)
    model = get_ID_model(conf) 

    ###############

    cnt = 0     # each ID count
    count = 0   # each image count
    save_n = 0  # for count csv part
    n = 0
    list = [i for i in range(len(df))]

    path = "\\workspace\\~~~~~~~~~~~~\\0214_cropped_affwild2"   # 크로핑된 폴더 경로 적기

    for vid_names in os.listdir(path):
        for vid_name in vid_names:
            vid_path = os.path.join(path, vid_name) 
            for framenums in os.listdir(vid_path):
                for framenum in framenums:
                    framepath = os.path.join(framenum,vid_path)
            

                    cnt += 1
                    # 코드 실행 종료 시간 기록
                    now_start_time = time.time()
                    end_time = time.time()
                    execution_time = end_time - start_time
                    now_execution_time = end_time - now_start_time
                    print(f"지금까지 총 실행 시간: {execution_time} 초 & 현재실행 시간: {now_execution_time} 초 & ...ing ", cnt/len(framenums)*100 , '%')
                
                    ID_start = True

                    # df에 입력
                    # /workspace/aff_wild2/batch1/206.mp4 => 원본경로가 df의 PATH
                    # 크로핑 경로: /workspace/Face-Recognition/Cropped_affwild2/video47/-1

                    for _, each_df in df[(df['PATH'].str.split("/").str[-1] == framepath.str.split("/").str[-2]) & (df['FrameNum'] == framepath.str.split("/").str[-1])].iterrows():


                        ########## 2. brisque ###########
                        df = brisque_classification(framepath, df)

                        ######## 3. id tracking ###########

                        df, ID_start, n = ID_tracking(model, framepath, ID_start, list, n, df)   # emb_base를 update하기 위해 return값으로 emb_base를 설정
                        # ID_start를 update하기 위해 return값으로 ID_start
                        
                        count += 1

                        ######## 4. 구한 embedding값으로 전후 프레임 코사인 유사도 구하기

                        
                        
                        save_n += 1

         



            # 비디오 ID별로 여러개의 프레임을 갖고 있는 csv 저장
            save_n += 1
            part_filename = vid_name + str(save_n) + '.csv'
            df.to_csv('/workspace/' + part_filename, index=False)


            t_end_time = time.time()
            total_execution_time = t_end_time - start_time
            print(f"total 실행 시간: {total_execution_time} 초")
            print("done")
