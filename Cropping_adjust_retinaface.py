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
import pandas as pd

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
    
        width = box[2] - box[0] # 박스의 가로
        height = box[3] - box[1] # 박스 세로

       
        # 새로운 박스의 중심점 => (코와 눈의 위치로 조절)
        center_eye_x = ( eye[0] + eye[2] )/2
        center_nose_x = nose[0]
        center_mouth_x = ( mouth[0] + mouth[2] )/2
        # y좌표는 고정
        center_nose_y = nose[1]
        center_eye_y = ( eye[1] + eye[3] )/2
        center_mouth_y = ( mouth[1] + mouth[3] )/2



        # 왼쪽을 바라볼때 ################################################################################################
        if (center_eye_x > center_nose_x) and (center_mouth_x > center_nose_x):

            ########################################## 정면인데 이상하게 나오는거 추가 ####################################################3
            if (( mouth[0] + mouth[2] )/3 <  center_nose_x) and (mouth[0]-eye[0] >= 10 ) :
                    
                # 정면을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ######################################
                if abs(center_eye_y - center_nose_y)  <=  abs(center_mouth_y - center_nose_y)*0.35  :
                    # 새로운 박스의 x1, y1
                    new_x1 = int(center_eye_x - ( width / 2))
                    new_y1 = int(center_nose_y + (- center_nose_y + center_eye_y)/2 - ( height / 2))


                # 정면을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 #####################################
                elif abs(center_eye_y - center_nose_y)*0.5  >= abs(center_mouth_y - center_nose_y) :

                    # 새로운 박스의 x1, y1
                    new_x1 = int(center_eye_x - ( width / 2))
                    #new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/2 - ( height / 2))
                    #new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/2 - ( height / 2)) # 박스를 위로 올리기
                    new_y1 = int(center_nose_y + (- center_nose_y + center_eye_y)/2 - ( height / 2)) # 박스를 위로 올리기

                    
                

                # 완전 정면 ###############################################################################################
                else :
                    # 새로운 박스의 x1, y1
                    new_x1 = int(center_eye_x - ( width / 2))
                    new_y1 = int(center_nose_y - (height / 2))


            # 왼쪽을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ######################################
            elif abs(center_mouth_y - center_nose_y)*0.29 <= abs(center_eye_y - center_nose_y) <= abs(center_mouth_y - center_nose_y)*0.45 :

                new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/2 - ( height / 2))    ##### y 좌표가 아래로 내려가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (center_eye_x - center_nose_x)/1.1 - ( width / 2))
                    
                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (center_mouth_x - center_nose_x)/1.1 - ( width / 2))


            # 왼쪽을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 #########################################
            elif abs(center_eye_y - center_nose_y)*0.5 >= abs(center_mouth_y - center_nose_y) : 

                new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/2 - ( height / 2))     ##### y 좌표가 위로 올라가야 함

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
        elif (center_eye_x < center_nose_x) and (center_mouth_x < center_nose_x) :

            # 오른쪽을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ############################################
            # if abs(center_eye_y - center_nose_y) <= abs(center_mouth_y - center_nose_y)*0.47  :
            # if abs(center_eye_y - center_nose_y) >= abs(center_mouth_y - center_nose_y) * 0.45:
            if abs(center_mouth_y - center_nose_y)*0.29 <= abs(center_eye_y - center_nose_y) <= abs(center_mouth_y - center_nose_y)*0.45 :

                new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/2 - ( height / 2))   ##### y 좌표가 아래로 내려가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))

                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


            # 오른쪽을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 ############################################
            elif abs(center_eye_y - center_nose_y)* 0.75 >= abs(center_mouth_y - center_nose_y) :
            #elif abs(center_eye_y - center_nose_y) >= abs(center_mouth_y - center_nose_y) * 0.45:

                new_y1 = int(center_nose_y + (- center_nose_y + center_eye_y)/2 - ( height / 2))    ##### y 좌표가 위로 올라가야 함

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    # new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    new_x1 = int(center_eye_x - (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    
                else :

                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    # new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))
                    new_x1 = int(center_eye_x - (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


            else : # 오른쪽을 바라보는데 정면 #############################################################################

                new_y1 = int(center_nose_y - (height / 2))

                # 눈 사이의 중앙값이 입 사이의 중앙값보다 클때 => 눈 사이의 중앙값을 이용
                if center_eye_x > center_mouth_x :

                    #new_x1 = int(center_eye_x + (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    new_x1 = int(center_eye_x - (- center_eye_x + center_nose_x)/1.1 - ( width / 2))
                    

                else :
                    # 눈 사이의 중앙값이 입 사이의 중앙값보다 작을때 => 입 사이의 중앙값을 이용
                    #new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))
                    new_x1 = int(center_eye_x + (- center_mouth_x + center_nose_x)/1.1 - ( width / 2))


        # 정면을 바라볼때 #############################################################################################
        else:

            # 정면을 바라보는데 위를 바라볼때 => 코가 눈이랑 가까움 => 입사이의 중앙값을 이용 ######################################
            if abs(center_eye_y - center_nose_y)  <=  abs(center_mouth_y - center_nose_y)*0.35  :
                # 새로운 박스의 x1, y1
                new_x1 = int(center_eye_x - ( width / 2))
                new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/2 - ( height / 2)) # 박스를 내리기


            # 정면을 바라보는데 아래를 바라볼때 => 코가 입이랑 가까움 => 눈사이의 중앙값을 이용 #####################################
            elif abs(center_eye_y - center_nose_y)*0.5  >= abs(center_mouth_y - center_nose_y) :

                # 새로운 박스의 x1, y1
                new_x1 = int(center_eye_x - ( width / 2))
                #new_y1 = int(center_nose_y - (- center_nose_y + center_eye_y)/2 - ( height / 2))
                #new_y1 = int(center_nose_y + (- center_nose_y + center_mouth_y)/2 - ( height / 2)) # 박스를 위로 올리기
                new_y1 = int(center_nose_y + (- center_nose_y + center_eye_y)/2 - ( height / 2)) # 박스를 위로 올리기

                
               

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

    # DataFrame 초기화
    df = pd.DataFrame(columns=["PATH", "BOX"])

    batchfolders = os.listdir(data_path)
    batchfolderpaths = [os.path.join(data_path, batchfolder) for batchfolder in batchfolders]

    for batchfolderpath in batchfolderpaths:
        videos = os.listdir(batchfolderpath)
        videopaths = [os.path.join(batchfolderpath, video) for video in videos]

        for ID_video in videopaths:


            cap = cv2.VideoCapture(ID_video)
            i = 0
            all_boxes = []  # 각 프레임에서 계산된 바운딩 박스 
            all_lm = []
            framenum = [0]
            #inital_max_box = 0

            anomaly_frame_box_idx = [] # 갑자기 얼굴이 큰 사람이 등장하는 프레임


            if not cap.isOpened():
                print("error video")

            # 현재프레임
            now_frame = 0
            all_frame = []
            prev_box = None

            while cap.isOpened():

                ret, frame = cap.read()

                if not ret:
                    print("프레임을 읽는 데 실패했습니다. 종료 중...")
                    break


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


                        # cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                        # # landms
                        # cv2.circle(frame, (b[5], b[6]), 1, (0, 0, 255), 4)
                        # cv2.circle(frame, (b[7], b[8]), 1, (0, 255, 255), 4)
                        # cv2.circle(frame, (b[9], b[10]), 1, (255, 0, 255), 4)
                        # cv2.circle(frame, (b[11], b[12]), 1, (0, 255, 0), 4)
                        # cv2.circle(frame, (b[13], b[14]), 1, (255, 0, 0), 4)


                        # 저장된 바운딩 박스에 현재 프레임의 정보 추가
                        # all_boxes.append(adjusted_box)
                        eye_middle = (eye_x1 + eye_x2) /2
                        all_lm.append((eye_middle, nose_y))


                        # 현재프레임 박스 : 이전 프레임에서 집힌 박스랑 가장 가까운 박스 선택
                        if prev_box is not None:


                            # 해당 신뢰도 값중에 max인거 선정
                            max_confidence_idx = np.argmax(dets[:, 4])

                            # Confidence Score가 가장 높은 박스 선택
                            b[0], b[1], b[2], b[3] = dets[:,:4][max_confidence_idx]

                            ###############################################################################


                            distances = [
                                np.sqrt((adjusted_box[0] - prev_box[0])**2 + (adjusted_box[1] - prev_box[1])**2) + np.sqrt((adjusted_box[2] - prev_box[2])**2 + (adjusted_box[3] - prev_box[3])**2)

                                for adjusted_box in dets[:,:4]
                            ]
                            
                            # 거리가 최소인 박스 선택하기 위해 오름차순 정렬
                            distances.sort(reverse=False)
                            
                            # 신뢰도 값 인덱스
                            th_list = [dets[:, 4][index] for index,th in enumerate(distances)]
                            
                            # 해당 신뢰도 값중에 max인거 선정
                            dis_max_confidence_idx = np.argmax(th_list)

                            # Confidence Score가 가장 높은 박스 선택
                            b[0], b[1], b[2], b[3] = dets[:,:4][dis_max_confidence_idx]
                            #############################################################################

                

                            if max_confidence_idx == dis_max_confidence_idx: 
                            
                                # 만들어진 박스의 중심이 코일 수 있도록 조정 하기
                                adjusted_box = adjust_box_around_nose((b[0], b[1], b[2], b[3]), (eye_x1,eye_y1,eye_x2,eye_y2), (nose_x, nose_y), (mouth_x1,mouth_y1,mouth_x2,mouth_y2) )


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

                                break

                                
                                

                            else: # 최대 신뢰도 값이 최소 길이 박스랑 다를때


                                # 최소 거리의 index값
                                min_distance_idx = np.argmin(distances)

                                
                                # 거리가 제일 비슷한 값 찾기
                                b[0], b[1], b[2], b[3] = dets[:,:4][min_distance_idx]

                                # 만들어진 박스의 중심이 코일 수 있도록 조정 하기
                                adjusted_box = adjust_box_around_nose((b[0], b[1], b[2], b[3]), (eye_x1,eye_y1,eye_x2,eye_y2), (nose_x, nose_y), (mouth_x1,mouth_y1,mouth_x2,mouth_y2) )


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

                                
                                break


                                


                        else:

                            # 0번째 프레임일때 Confidence Score가 가장 높은 박스 선택
                            max_confidence_idx = np.argmax(dets[:, 4])

                            # Confidence Score가 가장 높은 박스 선택
                            b[0], b[1], b[2], b[3] = dets[:,:4][max_confidence_idx]

                            # 만들어진 박스의 중심이 코일 수 있도록 조정 하기
                            adjusted_box = adjust_box_around_nose((b[0], b[1], b[2], b[3]), (eye_x1,eye_y1,eye_x2,eye_y2), (nose_x, nose_y), (mouth_x1,mouth_y1,mouth_x2,mouth_y2) )

                            # 코와 눈의 좌표로 조정된 박스의 가로와 세로

                            adjust_width = adjusted_box[2] - adjusted_box[0] 
                            adjust_height = adjusted_box[3] - adjusted_box[1]
                            
                        
                            

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
                            break

                    
                        
                       

                    

                    ################## break 걸기 중요 #################### => for 문 나가기 ###############
                    # 현재 프레임에서 선택된 박스를 이전 박스로 저장
                    prev_box = adjusted_box
                    now_frame += 1


        

                        

                    cropped_padding = np.zeros((int(adjust_height) , int(adjust_width), 3), dtype=np.uint8)

            
                    # cv2.rectangle(frame, (adjusted_box[0], adjusted_box[1]), (adjusted_box[2], adjusted_box[3]), (0, 0, 255), 2)

                    # 이미지를 크롭할 부분 추출
                    cropped_face = frame[adjusted_box[1]:adjusted_box[3], adjusted_box[0]:adjusted_box[2]]


                    # cropped_face 을  cropped_padding에 삽입
                    cropped_padding[:cropped_face.shape[0], :cropped_face.shape[1], : ] = cropped_face


                    cropped_padding[cropped_face.shape[0]:, :, :] = 0
                    cropped_padding[:, cropped_face.shape[1]: , :] = 0

            



                    # 파일명에서 확장자 제거
                    ID_video_name = os.path.splitext(os.path.basename(ID_video))[0]

                    # 폴더가 없으면 생성
                    save_folder = f'/workspace/Pytorch_Retinaface/Cropped_affwild2/{ID_video_name}/'
                    os.makedirs(save_folder, exist_ok=True)

                    cv2.imwrite(f'{save_folder}{i}.jpg', cv2.resize(cropped_padding, (224, 224)))

                    # DataFrame에 정보 추가
                    # df = df.append({"PATH": ID_video, "BOX": adjusted_box}, ignore_index=True)
                    df = pd.concat([df, pd.DataFrame({"PATH": [ID_video], "BOX": [adjusted_box]})], ignore_index=True)

                    i += 1
                    framenum.append(i)


                    # if i == 113 :
                    #      continue

                    # i += 1
                    # if i == 502: 
                    #     break

            print(f" #################################{ID_video_name} done")
            # CSV 파일로 저장
            csv_save_path = f'/workspace/Pytorch_Retinaface/CSV_cropped_affwild2/{ID_video_name}.csv'
            df.to_csv(csv_save_path, index=False)

            
                    
        cap.release()


                        


