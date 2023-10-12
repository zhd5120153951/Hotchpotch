import os
import cv2
from PIL import Image


def ensure_parent_directory_exists(file_path):
    """
    确定文件所在父文件夹存在

    参数:
    - file_path (str): 文件路径

    返回值:
    - None
    """
    parent_dir = os.path.dirname(file_path)
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
        print(f"Created directory: {parent_dir}")
    else:
        print(f"Directory {parent_dir} already exists.")


def save_file_with_incrementing_number(filename):
    """
    确定文件所在父文件夹存在

    参数:
    - file_name (str): 文件名

    返回值:
    - file_name (str): 文件名自增长
    """
    base_name, ext = os.path.splitext(filename)
    counter = 1
    while os.path.exists(filename):
        filename = f"{base_name}_{counter}{ext}"
        counter += 1
    return filename


def ensure_dir_existsr(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_video_from_frames(input_folder, output_video_path, fps=30):
    ensure_parent_directory_exists(output_video_path)
    frame_files = sorted(os.listdir(input_folder))

    first_frame = cv2.imread(os.path.join(input_folder, frame_files[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = save_file_with_incrementing_number(output_video_path)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(input_folder, frame_file)
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()


def record_video_increment_name(rtsp_url, write_name, is_show=True):
    # 处理实时视频流一般is_show打开，后台处理视频的话is_show可为False
    ensure_parent_directory_exists(write_name)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"无法打开 {rtsp_url}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_file = save_file_with_incrementing_number(write_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用H.264编码
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #处理红外视频
        # 转换颜色空间

        # gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # pil_image = Image.fromarray(gray_image)
        # img = pil_image.convert('RGB')

        video_writer.write(frame)
        if is_show:
            cv2.imshow('RTSP Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def find_mp4_files(directory):
    mp4_files = []

    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".mp4")]:
            mp4_files.append(os.path.join(dirpath, filename))

    return mp4_files


def save_frames_from_video(rtsp_url, output_folder, is_show=True):
    ensure_dir_existsr(output_folder)
    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print(f"无法打开 {rtsp_url}")
        return

    frame_num = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        while os.path.exists(os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")):
            frame_num += 1
        frame_filename = os.path.join(output_folder, f"frame_{frame_num:05d}.jpg")
        cv2.imwrite(frame_filename, frame)

        frame_num += 1

        if is_show:
            cv2.imshow('RTSP Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    print(f"抽取第 {frame_num} 帧 并保存在 {output_folder}")