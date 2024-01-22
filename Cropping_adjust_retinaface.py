from __future__ import print_function
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='/workspace/Pytorch_Retinaface/weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
############################### threshold ############################################################################################
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
################################## save ###############################################################################################
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.9, type=float, help='visualization_threshold')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


if __name__ == '__main__':


    torch.set_grad_enabled(False)
    cfg = None
    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args.cpu else "cuda")
    net = net.to(device)

    resize = 1

    # testing begin


    #####################################
    def adjust_box_around_nose(box, eye, nose, mouth):
    
        width = box[2] - box[0]
        height = box[3] - box[1]

        center_box_x = box[0] + width / 2
        center_box_y = box[1] - height / 2

       
        # 새로운 박스의 중심점 => (코와 눈의 위치로 조절)
        center_eye_x = ( eye[0] + eye[2] )/2
        center_nose_x = nose[0]
        center_mouth_x = ( mouth[0] + mouth[2] )/2

        # y좌표는 고정
        center_nose_y = nose[1]
        center_eye_y = ( eye[1] + eye[3] )/2
        center_mouth_y = ( mouth[1] + mouth[3] )/2

        # 랜드마크 간의 비율 계산 => 정면일때 threshold 만드려고 => 세로 비율 조정을 통해 박스안에 눈코입턱 다 나오게 하려고 

        # 다른 사람 얼굴인데 랜드 마크 기반으로 측정하기

        # frame 18에서 정면 => 실험한 결과 
        # 눈 ~ 코 : 코 ~ 입 => 0.48 : 1 정도  

        # 처음으로 얼굴을 들어 올리기 시작한 frame 46 에서 실험한 결과 
        # 눈 ~ 코 : 코 ~ 입 => 0.3824561403508772 : 1 정도

        # frame 1에서 살짝 왼쪽으로 치우쳐진 정면 
        # 눈 ~ 코 : 코 ~ 입 => 0.37640 : 1 정도

        # 즉 완전 정면 eye_nose_mouth_ratio  0.45

        y_eye_box_ratio = abs(box[1] - center_eye_y)
        y_mouth_box_ratio = abs(center_mouth_y - box[3])

        # 턱 안잘리려면 y_eye_box_ratio > height/4



        eye_nose_mouth_ratio = y_eye_box_ratio/y_mouth_box_ratio


        y_eye_nose_ratio = abs(center_eye_y - center_nose_y)
        y_nose_mouth_ratio = abs(center_nose_y - center_mouth_y)
        eye_nose_mouth_ratio = y_eye_nose_ratio/y_nose_mouth_ratio







        # 왼쪽을 바라볼때 ################################################################################################
        if center_eye_x > center_nose_x  and center_box_x > center_nose_x:

            # 왼쪽을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ######################################
            #if abs(center_eye_y - center_nose_y) < abs(center_mouth_y - center_nose_y) :
            if eye_nose_mouth_ratio < 0.4 :

                new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/3 - ( height / 2))    ##### y 좌표가 아래로 내려가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (center_eye_x - center_nose_x)/1.1 - ( width / 2))
                    
                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (center_mouth_x - center_nose_x)/1.1 - ( width / 2))


            # 왼쪽을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 #########################################
            # elif abs(center_eye_y - center_nose_y) > abs(center_mouth_y - center_nose_y) : 
            elif eye_nose_mouth_ratio > 0.6 : 

                new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/3 - ( height / 2))     ##### y 좌표가 위로 올라가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (center_eye_x - center_nose_x)/1.1 - ( width / 2))
                    

                else :

                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (center_mouth_x - center_nose_x)/1.1 - ( width / 2))
      


            else : # 왼쪽을 바라보는데 정면 ##################################################################################
                
                new_y1 = int(center_nose_y - (height / 2))

                 # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (center_eye_x - center_nose_x)/1.1 - ( width / 2))
                    
                else :

                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (center_mouth_x - center_nose_x)/1.1 - ( width / 2))


        # 오른쪽을 바라볼때 ###################################################################################################
        elif center_eye_x < center_nose_x and center_box_x < center_nose_x:

            # 오른쪽을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ############################################
            if eye_nose_mouth_ratio < 0.4 : 

                new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/3 - ( height / 2))   ##### y 좌표가 아래로 내려가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))

                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


            # 오른쪽을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 ############################################
            elif eye_nose_mouth_ratio > 0.6 : 

                new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/3 - ( height / 2))    ##### y 좌표가 위로 올라가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    
                else :

                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


            else : # 오른쪽을 바라보는데 정면 #############################################################################

                new_y1 = int(center_nose_y - (height / 2))

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    

                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


        # 정면을 바라볼때 #############################################################################################
        else:

            # 정면을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ######################################
            if eye_nose_mouth_ratio < 0.4:
                # 새로운 박스의 x1, y1
                new_x1 = int(center_eye_x - ( width / 2))
                new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/3 - ( height / 2))


            # 정면을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 #####################################
            elif eye_nose_mouth_ratio > 0.6  :
                # 새로운 박스의 x1, y1
                new_x1 = int(center_eye_x - ( width / 2))
                new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/3 - ( height / 2))

            # 완전 정면 ###############################################################################################
            else :
                # 새로운 박스의 x1, y1
                new_x1 = int(center_eye_x - ( width / 2))
                new_y1 = int(center_nose_y - (height / 2))



        # 새로운 박스의 x2, y2
        new_x2 = new_x1 + width
        new_y2 = new_y1 + height

        return new_x1, new_y1, new_x2, new_y2 # 코의 위치로 조절한 새로운 박스
    
    #######################################

# 왼쪽 위를 쳐다보고 있는 47번 프레임에서 바운딩 박스의 좌표는 (713, 121, 1139, 653) 눈의 좌표는 (811, 292, 1003, 313), 코의 좌표는 (872, 357) 입의 좌표는 (798, 490, 947, 509) 라고 할때 프레임의 바운딩 박스가 정상적
# 근데 똑같이 왼쪽 위를 쳐다보고 있는 48번 프레임에서 바운딩 박스의 좌표는 (714, 105, 1148, 640) 눈의 좌표는 (823, 279, 1013, 305), 코의 좌표는 (884, 355) 입의 좌표는 (805, 479, 953, 502)인데 이때는 턱이 잘려서 바운딩 박스가 비정상적 이야


    data_path = "/workspace/aff_wild2/"

    batchfolders = os.listdir(data_path)
    batchfolderpaths = [os.path.join(data_path, batchfolder) for batchfolder in batchfolders]

    # for batchfolderpath in batchfolderpaths:
    #     videos = os.listdir(batchfolderpath)
    #     videopaths = [os.path.join(batchfolderpath, ID_video) for ID_video in videos]

    #     #for videopath in videopaths:


    # testing 
    # cap = cv2.VideoCapture( videopaths[0] )
    cap = cv2.VideoCapture("/workspace/aff_wild2/batch1/7-60-1920x1080.mp4")
    i = 0
    all_boxes = []  # 각 프레임에서 계산된 바운딩 박스 
    all_lm = []
    #inital_max_box = 0

    anomaly_frame_box_idx = [] # 갑자기 얼굴이 큰 사람이 등장하는 프레임


    if not cap.isOpened():
        print("error video")

    # 현재프레임
    now_frame = 0
    all_frame = []

    while cap.isOpened():

        ret, frame = cap.read()


        img = np.float32(frame)

        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)

        tic = time.time()
        loc, conf, landms = net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data

        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                            img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:args.keep_top_k, :]
        landms = landms[:args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)




        

        # show image
        if args.save_image:

            

            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))

                
                # nose

                nose_x = b[9]
                nose_y = b[10]

                # eye 

                eye_x1 = b[5]
                eye_x2 = b[7]
                eye_y1 = b[6]
                eye_y2 = b[8]

                # mouth

                mouth_x1 = b[11]
                mouth_x2 = b[13]
                mouth_y1 = b[12]
                mouth_y2 = b[14]




                # 박스의 가로와 세로

                width = b[2] - b[0]
                height = b[3] - b[1]

                # 만들어진 박스의 중심이 코일 수 있도록 조정 하기
                adjusted_box = adjust_box_around_nose((b[0], b[1], b[2], b[3]), (eye_x1,eye_y1,eye_x2,eye_y2), (nose_x, nose_y), (mouth_x1,mouth_y1,mouth_x2,mouth_y2) )

                # 코와 눈의 좌표로 조정된 박스의 가로와 세로

                adjust_width = adjusted_box[2] - adjusted_box[0] 
                adjust_height = adjusted_box[3] - adjusted_box[1]
                
               
                if now_frame == 0 :
                    
                    all_frame.append([adjust_width, adjust_height])

                else :
                    # 이전 프레임 박스의 넓이
                    all_frame.append([adjust_width, adjust_height])
                    previous_box_area = all_frame[now_frame-1][0] * all_frame[now_frame-1][1]
                    now_box_area = all_frame[now_frame][0] * all_frame[now_frame][1]

                    # 이전프레임이랑 현재프레임의 넓이 차이가 많이 나면 ########################################################## 조정 필요 #####################################
                    if ( previous_box_area > now_box_area )  and abs(previous_box_area / now_box_area) > 4:
                        anomaly_frame_box_idx.append(now_frame) # anomaly한 프레임의 인덱스를 저장

                    elif ( now_box_area > previous_box_area ) and abs(now_box_area / previous_box_area ) > 4:
                        anomaly_frame_box_idx.append(now_frame) # anomaly한 프레임의 인덱스를 저장
                    

                

                # 크로핑 에러 방지
                adjusted_box = (
                    int(adjusted_box[0]),
                    int(adjusted_box[1]),
                    int(adjusted_box[2]),
                    int(adjusted_box[3])
)

                adjusted_box = (
                    max(0, adjusted_box[0]),  # 음수일때는 0 으로 맞추기
                    max(0, adjusted_box[1]),  # 음수일때는 0 으로 맞추기
                    min(im_width, adjusted_box[2]),  # 바운딩박스가 이미지보다 크지 않도록 설정
                    min(im_height, adjusted_box[3])
                )
                

                
                cx = adjusted_box[0]
                cy = adjusted_box[1] + 12
                #print(f"cx: {cx}, cy: {cy}")


                cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # landms
                cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)


                # 저장된 바운딩 박스에 현재 프레임의 정보 추가
                # all_boxes.append(adjusted_box)
                eye_middle = (eye_x1 + eye_x2) /2
                all_lm.append((eye_middle, nose_y))


            

                

                if now_frame == 0 :
                    print("정면 - 디버깅")

                # if now_frame == 48 :
                #     print("이상 - 디버깅")

                now_frame += 1

            cropped_padding = np.zeros((int(adjust_height) , int(adjust_width), 3), dtype=np.uint8)

    
            cv2.rectangle(frame, (adjusted_box[0], adjusted_box[1]), (adjusted_box[2], adjusted_box[3]), (0, 0, 255), 2)

            # 이미지를 크롭할 부분 추출
            cropped_face = frame[adjusted_box[1]:adjusted_box[3], adjusted_box[0]:adjusted_box[2]]


            # cropped_face 을  cropped_padding에 삽입
            cropped_padding[:cropped_face.shape[0], :cropped_face.shape[1], : ] = cropped_face


            cropped_padding[cropped_face.shape[0]:, :, :] = 0
            cropped_padding[:, cropped_face.shape[1]: , :] = 0

    






            cv2.imwrite(f'/workspace/Pytorch_Retinaface/moutheyenose_cropping_output_test3/{i}.jpg', cv2.resize(cropped_padding,(224,224)))

    
            # i += 1
            # if i == 18 :
            #     continue

            # 11일떄는 갠춘
            # 12일때는 갑자기 이상함
            
            # 여기까지는 11이라는 뜻
            i += 1
            if i == 12 :
                continue

            # i += 1
            # if i == 13 :
            #     continue

            # i += 1
            # if i == 100: 
            #     break




print(anomaly_frame_box_idx)   


            
cap.release()


#####################################################################################################

# cap2 = cv2.VideoCapture("/workspace/aff_wild2/batch1/7-60-1920x1080.mp4")
# i = 0
# frame_num = 0

# if not cap2.isOpened():
#     print("error video")

# #  모든 프레임의 바운딩 박스 중에서 최대 크기의 박스 추출
# max_width_in_all_box = max(all_boxes, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))

# max_width_all = max_width_in_all_box[2] - max_width_in_all_box[0] # 전체 프레임 중에 제일 큰 박스의 가로 길이
# max_height_all = max_width_in_all_box[3] - max_width_in_all_box[1] # 전체 프레임 중에 제일 큰 박스의 세로 길이

# now_nose_width_all = adjusted_box[2] - adjusted_box[0]  # 코 기준으로 구축된 박스의 현재 가로 길이
# now_nose_height_all = adjusted_box[3] - adjusted_box[1] # 코 기준으로 구축된 박스의 현재 세로 길이


# # 모든 프레임의 바운딩 박스 크기를 max_width_in_all_box로 설정
# #max_width = max_width_in_all_box[2] - max_width_in_all_box[0]       ###### fix ######
# #max_height = max_width_in_all_box[3] - max_width_in_all_box[1]      ###### fix ######


# max_width_ratio =   max_width_all / now_nose_width_all   ###### fix ######
# max_height_ratio =  max_height_all / now_nose_height_all   ###### fix ######



# while cap2.isOpened():

#     ret, frame = cap2.read()
    

#     max_all_box = (int(all_nose_lm[frame_num][0] - ( max_width_ratio * now_nose_width_all/2)), int(all_nose_lm[frame_num][1] - ( max_height_ratio * now_nose_height_all/2 ))
#                    ,int(all_nose_lm[frame_num][0] + ( max_width_ratio * now_nose_width_all/2)), int(all_nose_lm[frame_num][1] + ( max_height_ratio * now_nose_height_all/2)) )
    
    

#     # ll_nose_lm[frame_num] : 각각의 코의 좌표
#     # max_width_ratio * width/2 : 코에서 어느정도로 이동해야 하는지 => 커져야하는 비율 * 현재 프레임의 가로값 * 1/2 


#     # max_all_box = (int(all_nose_lm[frame_num][0] - ( max_width_ratio * width/2)), int(all_nose_lm[frame_num][1] - ( max_height_ratio * height/2 ))
#     #                ,int(all_nose_lm[frame_num][0] + ( max_width_ratio * width/2)), int(all_nose_lm[frame_num][1] + ( max_height_ratio * height/2)) )

    

#     # img의 NumPy 배열 생성
#     # 검정색으로 채우기
#     cropped_padding = np.zeros((int(max_height_ratio * max_height_all) , int(max_width_ratio * max_width_all), 3), dtype=np.uint8)


    
#     max_all_box = (
#                     max(0, max_all_box[0]),  # 음수일때는 0 으로 맞추기
#                     max(0, max_all_box[1]),  # 음수일때는 0 으로 맞추기
#                     min(im_width, max_all_box[2]),  # 바운딩박스가 이미지보다 크지 않도록 설정
#                     min(im_height, max_all_box[3])
#                 )


#     frame_num += 1

    
#     cv2.rectangle(frame, (max_all_box[0], max_all_box[1]), (max_all_box[2], max_all_box[3]), (0, 0, 255), 2)

#     # 이미지를 크롭할 부분 추출
#     cropped_face = frame[max_all_box[1]:max_all_box[3], max_all_box[0]:max_all_box[2]]


#     # cropped_face 을  cropped_padding에 삽입
#     cropped_padding[:cropped_face.shape[0], :cropped_face.shape[1], : ] = cropped_face


#     cropped_padding[cropped_face.shape[0]:, :, :] = 0
#     cropped_padding[:, cropped_face.shape[1]: , :] = 0

#     """  
#     cropped_face.shape() : (573, 497, 3)
#     cropped_padding.shape() : (629, 497, 3)
#     cropped_padding: (max_height, max_width, 3 => channel num )
#     cropped_face.shape[0]: cropped_face의 높이
#     cropped_face.shape[1]: cropped_face의 너비 
    
#     """






#     cv2.imwrite(f'/workspace/Pytorch_Retinaface/other_cropping_output_test1/{i}.jpg', cropped_padding)

    

#     i += 1
#     if i == 50: 
#         break

# print(anomaly_frame_box_idx)            
            

# 1. cropping => gittering still
# 2. 최대넓이 박스 while문 밖으로 빼기 => cropping_output_test2
# 3. 코 말고 다른 조건 걸어서 박스 해보기

# 3) numpy로 프레임 하나씩 저장해서 while문 안해도 될 수 있게 하기
# 4) 얼굴이 멀리있거나 가까이 있거나 크롭핑했을때 다 같은 곳에 있는것처럼 보이려고
# 5) 바운딩박스가 코 기준으로 제일 크게 나오게 하기
# 6) 얼굴을 돌렸을때 기준을 어떻게 할지 생각하기


# 코 랜드마크 잡아서 다시 하기
