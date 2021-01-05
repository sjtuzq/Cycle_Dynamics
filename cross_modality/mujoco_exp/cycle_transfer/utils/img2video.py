import os
import cv2


class Agent():
    def __init__(self,img_root,video_path):
        self.img_root = img_root
        self.fps = 12
        self.img_limit = self.fps*4
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        self.videoWriter = cv2.VideoWriter(video_path, fourcc, self.fps, (256, 256))

    def run(self):
        frame_list = os.listdir(self.img_root)
        frame_list = sorted(frame_list,key=lambda x:int(x.split('.')[0]))
        frame_list = frame_list[:self.img_limit]

        for path in frame_list:
            frame = cv2.imread(os.path.join(self.img_root,path))
            self.videoWriter.write(frame)
            print(path)
        self.videoWriter.release()


if __name__ == '__main__':
    img_root = '../../robotlogs/test11/epoch-1'
    video_path = '../../robotlogs/test11/video_epoch1.avi'
    agent = Agent(img_root,video_path)
    agent.run()