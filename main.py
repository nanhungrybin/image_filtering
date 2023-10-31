import pandas as pd
import torch
from torchvision.transforms import functional as F
from PIL import Image
from brisque import BRISQUE

import argparse
from config import get_config
from Learner import *
from utils import *



################ animation filtering ##################

def brisque_classification(image_path, threshold, df):

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

    # threshold 설정에 따른 분류

    # animation
    if score >= threshold:
        # annimation 프레임 path
        df.loc[df['path'] == image_path, 'out_animation'] = 0
        #yes_anni.append(image_path)

        
    else:
        # 문제 없는 프레임 name
        df.loc[df['path'] == image_path, 'out_animation'] = 1
        #no_anni.append(image_path)



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

    # Evaluate mode. 
    Model = learner.model

    Model.eval()

    return Model


def ID_tracking(model, image_path, ID_start, df, emb_base):

    #out_id = []  # Initialize: 여기안에 두면 함수 출력 할 때마다 초기화됨 즉 out_id는 메인함수에 두기

    img = Image.open(image_path)

    if ID_start == True:

        emb_base = model(conf.test_transform(img).to(conf.device).unsqueeze(0)) # initialize
                         
        ID_start = False  # Update ID_start

        df.loc[df['path'] == image_path, 'out_ID'] = 1 # dataframe initialize with 1
        #out_id.append(image_path)

    else: ####################################

        emb_new = model(conf.test_transform(img).to(conf.device).unsqueeze(0)) # input
        diff = emb_new.unsqueeze(-1) - emb_base.transpose(1,0).unsqueeze(0) # to update emb_base : main fun return값으로 emb_base를 설정
        
        # Calculate distance
        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, _ = torch.min(dist, dim=1)

        if minimum > conf.threshold:

            # Update 'out_ID' for the current row
            df.loc[df['path'] == image_path, 'out_ID'] = 0
            #out_id.append(image_path)

        else:
            # Update 'out_ID' for the current row
            df.loc[df['path'] == image_path, 'out_ID'] = 1
            

    return df, emb_base



    
#--------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__': 

    # CSV 파일의 경로
    csv_file_path = 'Base.csv'

    df = pd.read_csv(csv_file_path)
    id_list = df['ID'].unique()
    

    # model load
    conf = get_config(False)
    model = get_ID_model(conf) 

    cnt = 0
    ####### 2. brisque #######
    for idx_id in id_list: # idx_id :  unique id
        cnt += 1
        print('ing..', cnt/len(id_list)*100 , '%')

        ID_start = True
        emb_base = 0
        for _, each_df in df[df['ID'] == idx_id].iterrows():

            image_path = each_df['path']
            label = each_df['label']
            threshold = 78 # anni threshold

            df = brisque_classification(image_path, threshold, df)

    ######## 3. id tracking ###########

            df, emb_base = ID_tracking(model, image_path, ID_start, df, emb_base)   # emb_base를 update하기 위해 return값으로 emb_base를 설정

            

    
    # # Save the dataframes to CSV files
    # yes_anni_df = pd.DataFrame(yes_anni, columns=["path"])
    # no_anni_df = pd.DataFrame(no_anni, columns=["path"])

    # yes_anni_df.to_csv("yes_anni_path.csv", index=False)
    # no_anni_df.to_csv("no_anni_path.csv", index=False)
    # print("done")

    # out_ID_df = pd.DataFrame(out_id, columns=["path"])
    # out_ID_df.to_csv("out_ID_path.csv", index=False)
    # print("done")

    # after Id tracking and filtering anni
    df.to_csv("filtering_Base.csv", index=False)
    print("done")
