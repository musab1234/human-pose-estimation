import argparse

import cv2
import numpy as np
import torch

from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints, BODY_PARTS_KPT_IDS, BODY_PARTS_PAF_IDS, BODY_PARTS_IDS
from modules.load_state import load_state
from val import normalize, pad_width
import threading
import zmq
import json
import time
from torch.autograd import Variable
from scipy.signal import savgol_filter


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:5556")

class VideoCapture:
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Threaded video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update,daemon=True, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

def infer_fast(net, img, net_input_height_size, stride, upsample_ratio, cpu, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1 / 256):
    height, width, _ = img.shape
    scale = net_input_height_size / height

    scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    scaled_img = normalize(scaled_img, img_mean, img_scale)
    min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
    padded_img, pad = pad_width(scaled_img, stride, pad_value, min_dims)

    tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float()
    tensor_img = tensor_img.cuda()

    stages_output = net(tensor_img)

    stage2_heatmaps = stages_output[-2]
    heatmaps = stage2_heatmaps.squeeze().permute(1, 2, 0).cpu().data.numpy()
    #print(heatmaps.shape)
    heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    stage2_pafs = stages_output[-1]
    pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
    pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)

    return heatmaps, pafs, scale, pad

def run_demo(net, image_provider, height_size, cpu):
    net = net.eval()
    net = net.cuda()

    stride = 8
    upsample_ratio = 4
    color = [0, 0, 255]

    global socket

    cv2.namedWindow("Human Pose", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Human Pose", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        grabbed, img = image_provider.read()
        orig_img = img.copy()
        heatmaps, pafs, scale, pad = infer_fast(net, img, height_size, stride, upsample_ratio, cpu)

        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(18):  # 19th for bg
            total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)

        

        for kpt_id in range(all_keypoints.shape[0]):
            all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
            all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale

        humans = []

        for n in range(len(pose_entries)):
            if len(pose_entries[n]) == 0:
                continue
            
            human = {};

            #print(all_keypoints[n])
            #all_keypoints[n] = savgol_filter(all_keypoints[n], 9, 3)

            for part_id in range(17):
                kpt_a_id = BODY_PARTS_KPT_IDS[part_id][0]

                global_kpt_a_id = pose_entries[n][kpt_a_id]
                if global_kpt_a_id != -1:
                    x_a, y_a = all_keypoints[int(global_kpt_a_id), 0:2]
                    cv2.circle(img, (int(x_a), int(y_a)), 8, color, -1)
                    human[BODY_PARTS_IDS[int(kpt_a_id)]] = {'position':{'x' : x_a  , 'y' : y_a} }

                kpt_b_id = BODY_PARTS_KPT_IDS[part_id][1]
                global_kpt_b_id = pose_entries[n][kpt_b_id]

                if global_kpt_b_id != -1:
                    x_b, y_b = all_keypoints[int(global_kpt_b_id), 0:2]
                    cv2.circle(img, (int(x_b), int(y_b)), 8, color, -1)
                    human[BODY_PARTS_IDS[int(kpt_b_id)]] = {'position':{'x' : x_b , 'y' : y_b} }
                

                if global_kpt_a_id != -1 and global_kpt_b_id != -1:
                    cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), color, 2)

            if any(human):
                humans.append(human)

        #socket.send_string(json.dumps(humans))

        img = cv2.addWeighted(orig_img, 0.8, img, 0.8, 0)
        cv2.imshow('Human Pose', img)
        key = cv2.waitKey(1)
        if key == 27:  # esc
            return

if __name__ == '__main__':
        
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load("checkpoint.pth.tar", map_location='cpu')
    load_state(net, checkpoint)


    net.eval()

    #example = torch.randn(1, 3, 256, 456)
    #traced_script_module = torch.jit.script(net, example)
    #traced_script_module.save('human-pose.pt')
    #x = torch.rand(1, 3, 256, 456)
    #sm = torch.jit.trace(net,[Variable(x)])
    #sm.save("human-pose.pt")


    frame_provider = VideoCapture(0)
    frame_provider.start()

    run_demo(net, frame_provider,256, "cuda")
    
