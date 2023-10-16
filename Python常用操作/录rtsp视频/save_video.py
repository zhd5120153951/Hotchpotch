from utils import record_video_increment_name
"""
根据所给的rtsp流来录制视频
假如给定./data_dir/xxx.mp4，自动创建./data_dir文件夹
视频保存为./data_dir文件夹下的xxx.mp4，若xxx.mp4存在，则保存为xxx_1.mp4
依次递增xxx_2.mp4、xxx_3.mp4。。。
"""

if __name__ == "__main__":
    rtsp_url = "rtsp://172.30.16.135:554/PR0"
    write_name = "D:\\FilePackage\\datasets\\thermal\\thermal.mp4"

    record_video_increment_name(rtsp_url, write_name)
