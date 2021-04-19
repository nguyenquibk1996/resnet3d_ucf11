import pandas as pd
import os
import numpy as np
import cv2
import glob
import shutil

from tqdm import tqdm
import subprocess

def extract_frame(destination_path, video_path, fps):
    frame_format_name = video_path.split('/')[-1][:-4]
    ffmpeg_cmd = ['ffmpeg', '-i', video_path, '-r', str(fps), '{}/{}_%03d.jpg'.format(destination_path, frame_format_name)]
    print(ffmpeg_cmd)
    subprocess.run(ffmpeg_cmd)
    print('\n')

def stack_frames(frames, clip_len):
    frames_stacked = [frames[x:x+16] for x in range(0, len(frames)-15)]
    return frames_stacked

if __name__ == '__main__':
    destination_path = 'frames_UCF11'
    videos_path = 'data_UCF11'
    videos = glob.glob(f'{videos_path}/*')
    cap = cv2.VideoCapture(videos[0])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    data = {
        'clip' : [],
        'label' : []
    }
    temp_dir = 'frames_temp'
    frames_dir = 'frames_UCF11'
    for video in tqdm(videos):
        os.makedirs(temp_dir, exist_ok=True)
        extract_frame(temp_dir, video, fps)
        frames = glob.glob(f'{temp_dir}/*')
        frames_stacked = stack_frames(frames, 16)
        data['clip'].extend(frame_stacked for frame_stacked in frames_stacked)
        data['label'].extend(frame_stacked[0].split('/')[-1].split('_')[1] for frame_stacked in frames_stacked)
        shutil.rmtree(temp_dir)
        extract_frame(frames_dir, video, fps)
  
    df = pd.DataFrame(data, columns=['clip', 'label'])
    df.to_csv('data.csv', index=False)
    


