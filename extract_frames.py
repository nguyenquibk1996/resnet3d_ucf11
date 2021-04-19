import cv2
import os
from tqdm import tqdm


video_path = 'data'
videos = os.listdir(video_path)
print('Number of video: ', len(videos))

clip_len = 16

for video in tqdm(videos):
    cap = cv2.VideoCapture(os.path.join(video_path, video))
    frame_count = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame_path = 'frames_UCF11/{}_{}.jpg'.format(video[:-4], frame_count)
            frame_count += 1   
            cv2.imwrite(frame_path, frame)
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
