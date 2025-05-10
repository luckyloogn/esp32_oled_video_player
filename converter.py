import struct
import tkinter as tk
from tkinter import filedialog, messagebox
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import glob
import os
import cv2
import numpy as np

TRANSLATIONS = {
    "chinese": {
        "app_title": "OLED视频取模工具",
        "preview": "预览",
        "rgb": "彩图",
        "grayscale": "灰度图",
        "binary": "二值图",
        "current_frame_index": "当前帧：",
        "previous_frame": "上一帧",
        "next_frame": "下一帧",
        "input": "输入",
        "input_type": "输入类型：",
        "video": "视频",
        "image_sequence": "图像序列",
        "select_video": "选择视频",
        "select_folder": "选择文件夹",
        "rotation": "旋转模式：",
        "scaling_mode": "缩放模式：",
        "fit": "适应",
        "fill": "填充",
        "fps": "帧率：",
        "start_time": "起始时间：",
        "end_time": "结束时间：",
        "process_video": "处理视频",
        "process_image_sequence": "处理图像序列",
        "binarization": "二值化",
        "threshold_options": "阈值选项：",
        "otsu": "自动 (OTSU)",
        "custom": "自定义",
        "binary_invert": "反转",
        "threshold": "阈值：",
        "binarize_all": "全部二值化",
        "binarization_adjustment": "二值化调整",
        "adjustment_options": "调整模式：",
        "single_frame": "单帧",
        "multi_frames": "多帧",
        "range_of_frames": "帧范围：",
        "start_frame": "起始帧：",
        "end_frame": "结束帧：",
        "adjust_single_frame": "调整单帧",
        "adjust_multi_frames": "调整多帧",
        "output": "输出",
        "select_output_path": "选择输出路径",
        "generate_target_file": "生成字模文件",
        "settings": "设置",
        "language": "语言：",
        "chinese": "中文",
        "english": "English",
        "theme": "主题：",
        "dark": "深色",
        "light": "浅色",
        "help": "帮助",
        "usage_guide_video": (
            "使用步骤：\n"
            "1. 设置“输入类型”\n"
            "2. 点击“选择视频”按钮设置\n"
            "    视频文件的路径\n"
            "3. 设置“旋转模式”\n"
            "4. 设置“缩放模式”\n"
            "5. 设置“帧率”\n"
            "6. 设置视频的“起始时间”\n"
            "7. 设置视频的“结束时间”\n"
            "8. 点击“处理视频”按钮，并\n"
            "    等待处理完毕\n"
            "9. 设置“阈值选项”\n"
            "10. 点击“全部二值化”按钮，\n"
            "     并等待处理完毕\n"
            "11. 在“二值化调整”处对选定\n"
            "     帧进行二值调整（可选）\n"
            "12. 点击“选择输出路径”按钮\n"
            "     设置字模文件的保存路径\n"
            "13. 点击“生成字模文件”按\n"
            "    钮，并等待处理完毕\n\n"
            "其他说明：\n"
            "· 可通过键盘上的左右方向\n"
            "  键切换当前预览的帧\n"
            "· 更新“旋转模式”、“缩放\n"
            "  模式”、“帧率”、“起始时\n"
            "  间”、“结束时间”时，需\n"
            "  要重新执行步骤8-13\n"
            "· 更新“阈值选项”时，需要\n"
            "  重新执行步骤10-13\n"
            "· 进行“二值化调整”时，需\n"
            "  要点击“调整单帧”或“调整\n"
            "  多帧”按钮保存，然后重新\n"
            "  执行步骤12-13\n"
        ),
        "usage_guide_image_sequence": (
            "使用步骤：\n"
            "1. 设置“输入类型”\n"
            "2. 点击“选择文件夹”按钮设\n"
            "    置存储图像序列的文件夹\n"
            "3. 设置“旋转模式”\n"
            "4. 设置“缩放模式”\n"
            "5. 设置“帧率”\n"
            "6. 点击“处理图像序列”按\n"
            "    钮，并等待处理完毕\n"
            "7. 设置“阈值选项”\n"
            "8. 点击“全部二值化”按钮，\n"
            "    并等待处理完毕\n"
            "9. 在“二值化调整”处对选定\n"
            "    帧进行二值调整（可选）\n"
            "10. 点击“选择输出路径”按钮\n"
            "     设置字模文件的保存路径\n"
            "11. 点击“生成字模文件”按\n"
            "    钮，并等待处理完毕\n\n"
            "其他说明：\n"
            "· 可通过键盘上的左右方向\n"
            "  键切换当前预览的帧\n"
            "· 更新“旋转模式”、“缩放\n"
            "  模式”、“帧率”时，需\n"
            "  要重新执行步骤6-11\n"
            "· 更新“阈值选项”时，需要\n"
            "  重新执行步骤8-11\n"
            "· 进行“二值化调整”时，需\n"
            "  要点击“调整单帧”或“调整\n"
            "  多帧”按钮保存，然后重新\n"
            "  执行步骤10-11\n"
        ),
        "error": "错误",
        "info": "信息",
        "processing": "处理中",
        "invalid_time_range": "无效的时间范围",
        "process_completed": "处理完成",
        "binarization_completed": "二值化完成",
        "adjustment_completed": "二值化调整完成",
        "no_video_selected": "未选择视频文件",
        "no_folder_selected": "未选择文件夹",
        "click_process_video_first": "请先点击“处理视频”",
        "click_process_image_sequence_first": "请先点击“处理图像序列”",
        "click_binarize_all_first": "请先点击“全部二值化”",
        "file_generation_success": "文件生成成功",
        "no_output_path_selected": "未选择输出路径",
    },
    "english": {
        "app_title": "OLED Video Converter",
        "preview": "Preview",
        "rgb": "RGB",
        "grayscale": "Grayscale",
        "binary": "Binary",
        "current_frame_index": "Current Frame Index:",
        "previous_frame": "Previous Frame",
        "next_frame": "Next Frame",
        "input": "Input",
        "input_type": "Input Type:",
        "video": "Video",
        "image_sequence": "Image Sequence",
        "select_video": "Select Video",
        "select_folder": "Select Folder",
        "rotation": "Rotation:",
        "scaling_mode": "Scaling Mode:",
        "fit": "Fit",
        "fill": "Fill",
        "fps": "Frame Rate:",
        "start_time": "Start Time:",
        "end_time": "End Time: ",
        "process_video": "Process Video",
        "process_image_sequence": "Process Image Sequence",
        "binarization": "Binarization",
        "threshold_options": "Threshold Options:",
        "otsu": "OTSU",
        "custom": "Custom",
        "binary_invert": "Invert",
        "threshold": "Threshold:",
        "binarize_all": "Binarize All Frames",
        "binarization_adjustment": "Binarization Adjustment",
        "adjustment_options": "Adjustment Options:",
        "single_frame": "Single Frame",
        "multi_frames": "Multi Frame",
        "range_of_frames": "Frame Range:",
        "start_frame": "Start Frame:",
        "end_frame": "End Frame: ",
        "adjust_single_frame": "Adjust Single Frame",
        "adjust_multi_frames": "Adjust Multi Frames",
        "output": "Output",
        "select_output_path": "Select Output Path",
        "generate_target_file": "Generate Target File",
        "settings": "Settings",
        "language": "Language:",
        "chinese": "Chinese",
        "english": "English",
        "theme": "Theme:",
        "dark": "Dark",
        "light": "Light",
        "help": "Help",
        "usage_guide_video": (
            "Usage Procedure:\n"
            "1. Set the 'Input Type'\n"
            "2. Click 'Select Video' to specify\n"
            "    the path of a video\n"
            "3. Set the 'Rotation'\n"
            "4. Set the 'Scaling Mode'\n"
            "5. Set the 'Frame Rate'\n"
            "6. Set the 'Start Time'\n"
            "7. Set the 'End Time'\n"
            "8. Click 'Process Video'\n"
            "9. Configure 'Threshold Options'\n"
            "10. Click 'Binarize All Frames'\n"
            "11. Perform adjustments on the\n"
            "      selected frame(s) in the\n"
            "     'Binarization Adjustment'\n"
            "     section (optional)\n"
            "12. Click 'Select Output Path' to\n"
            "     set the save path for the\n"
            "     target file\n"
            "13. Click 'Generate Target File'\n\n"
            "Other Notes:\n"
            "· You can switch the currently\n"
            "  previewed frame by using the\n"
            "  left and right arrow keys\n"
            "· When modifying the 'Rotation',\n"
            "  'Scaling Mode', 'Frame Rate',\n"
            "  'Start Time' or 'End Time',\n"
            "  repeat steps 8-13\n"
            "· When modifying 'Threshold \n"
            "  Options', repeat steps 10-13\n"
            "· For binarization adjustments: \n"
            "  Click 'Adjust Single Frame' or\n"
            "  'Adjust Multi Frames' to\n"
            "  save adjustments, then repeat\n"
            "  steps 12-13\n"
        ),
        "usage_guide_image_sequence": (
            "Usage Procedure:\n"
            "1. Set the 'Input Type'\n"
            "2. Click 'Select Folder' to set \n"
            "   the folder for storing image\n"
            "   sequence\n"
            "3. Set the 'Rotation'\n"
            "4. Set the 'Scaling Mode'\n"
            "5. Set the 'Frame Rate'\n"
            "6. Click 'Process Image Sequence'\n"
            "7. Configure 'Threshold Options'\n"
            "8. Click 'Binarize All Frames'\n"
            "9. Perform adjustments on the\n"
            "     selected frame(s) in the\n"
            "    'Binarization Adjustment'\n"
            "    section (optional)\n"
            "10. Click 'Select Output Path' to\n"
            "     set the save path for the\n"
            "     target file\n"
            "11. Click 'Generate Target File'\n\n"
            "Other Notes:\n"
            "· You can switch the currently\n"
            "  previewed frame by using the\n"
            "  left and right arrow keys\n"
            "· When modifying the 'Rotation',\n"
            "  'Scaling Mode' or 'Frame Rate',\n"
            "  repeat steps 6-11\n"
            "· When modifying 'Threshold \n"
            "  Options', repeat steps 8-11\n"
            "· For binarization adjustments: \n"
            "  Click 'Adjust Single Frame' or\n"
            "  'Adjust Multi Frames' to\n"
            "  save adjustments, then repeat\n"
            "  steps 10-11\n"
        ),
        "error": "Error",
        "info": "Info",
        "processing": "Processing",
        "invalid_time_range": "Invalid time range",
        "process_completed": "Process completed",
        "binarization_completed": "Binarization completed",
        "adjustment_completed": "Binarization adjustment completed",
        "no_video_selected": "No video file selected",
        "no_folder_selected": "No folder selected",
        "click_process_video_first": "Please click 'Process Video' first",
        "click_process_image_sequence_first": "Please click 'Process Image Sequence' first",
        "click_binarize_all_first": "Please click 'Binarize All' first",
        "file_generation_success": "File generation successful",
        "no_output_path_selected": "No output path selected",
    },
}


class Model:

    def __init__(self):
        self.rgb_images: list[Image.Image] = []
        self.gray_images: list[Image.Image] = []
        self.binary_images: list[Image.Image] = []
        self.threshold_values: list[int] = []
        self.current_frame_index = 0
        self.video_duration = 0.0
        self.video_fps = 0.0

    def reset(self):
        self.rgb_images.clear()
        self.gray_images.clear()
        self.binary_images.clear()
        self.threshold_values.clear()
        self.current_frame_index = 0
        self.video_duration = 0.0
        self.video_fps = 0.0

    def update_video_duration_and_fps(self, video_path: str):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
        cap.release()
        self.video_duration = video_duration
        self.video_fps = video_fps

    def rotate_image(self, image, angle: int):
        if angle == 0:
            return image
        elif angle == 90:
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            return cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return image

    def resize_image(self, image, target_width=128, target_height=64, mode="fit"):
        h, w = image.shape[:2]
        aspect_ratio = w / h
        target_aspect_ratio = target_width / target_height
        if mode == "fill":
            if aspect_ratio > target_aspect_ratio:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
                resized = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
                )
                start_x = (new_width - target_width) // 2
                cropped = resized[:, start_x : start_x + target_width]
                return cropped
            else:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
                resized = cv2.resize(
                    image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
                )
                start_y = (new_height - target_height) // 2
                cropped = resized[start_y : start_y + target_height, :]
                return cropped
        elif mode == "fit":
            if aspect_ratio > target_aspect_ratio:
                new_width = target_width
                new_height = int(target_width / aspect_ratio)
            else:
                new_height = target_height
                new_width = int(target_height * aspect_ratio)
            resized = cv2.resize(
                image, (new_width, new_height), interpolation=cv2.INTER_NEAREST
            )
            canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            x_offset = (target_width - new_width) // 2
            y_offset = (target_height - new_height) // 2
            canvas[
                y_offset : y_offset + new_height, x_offset : x_offset + new_width
            ] = resized
            return canvas
        else:
            raise ValueError("Invalid mode. Choose 'fill' or 'fit'.")

    def process_video(
        self,
        video_path,
        output_fps,
        start_time,
        end_time,
        rotation,
        scaling_mode,
        progress_callback,
    ):
        self.rgb_images.clear()
        self.gray_images.clear()
        self.binary_images.clear()
        self.threshold_values.clear()
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Failed to open video file: {video_path}")
        if start_time < 0 or end_time > self.video_duration or start_time >= end_time:
            cap.release()
            raise ValueError(
                f"Invalid time range: start_time={start_time}, end_time={end_time}. "
                f"Must satisfy 0 ≤ start_time < end_time ≤ {self.video_duration}"
            )
        frame_interval = max(1, round(self.video_fps / output_fps))
        start_frame = int(start_time * self.video_fps)
        end_frame = int(end_time * self.video_fps)
        total_frames_in_range = end_frame - start_frame
        total_extracted_frames = total_frames_in_range // frame_interval
        frame_count = 0
        frame_index = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        while frame_count + start_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                frame_rotated = self.rotate_image(frame, rotation)
                frame_resized = self.resize_image(frame_rotated, 128, 64, scaling_mode)
                frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                frame_rgb_pil = Image.fromarray(
                    cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                )
                frame_gray_pil = Image.fromarray(frame_gray)
                self.rgb_images.append(frame_rgb_pil)
                self.gray_images.append(frame_gray_pil)
                progress_callback(
                    frame_rgb_pil,
                    frame_gray_pil,
                    frame_index,
                    total_extracted_frames - 1,
                )
                frame_index += 1
            frame_count += 1
        cap.release()

    def process_image_sequence(
        self, dir_path, rotation, scaling_mode, progress_callback
    ):
        self.rgb_images.clear()
        self.gray_images.clear()
        self.binary_images.clear()
        self.threshold_values.clear()
        images_path, count = self.get_images_path(
            dir_path, ["jpg", "jpeg", "png", "bmp"]
        )
        for i, filename in enumerate(images_path):
            frame = cv2.imread(filename)
            frame_rotated = self.rotate_image(frame, rotation)
            frame_resized = self.resize_image(frame_rotated, 128, 64, scaling_mode)
            frame_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
            frame_rgb_pil = Image.fromarray(
                cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            )
            frame_gray_pil = Image.fromarray(frame_gray)
            self.rgb_images.append(frame_rgb_pil)
            self.gray_images.append(frame_gray_pil)
            progress_callback(frame_rgb_pil, frame_gray_pil, i, count - 1)

    def get_images_path(self, dir_path, pic_format):
        images_path = []
        for fmt in pic_format:
            images_path.extend(glob.glob(os.path.join(dir_path, f"*.{fmt}")))
        return images_path, len(images_path)

    def binarize_one_frame(self, frame_index, threshold_type, threshold, binary_invert):
        if not (0 <= frame_index < len(self.gray_images)):
            raise ValueError(
                f"Invalid frame index {frame_index} for {len(self.gray_images)} images"
            )
        image_gray = self.gray_images[frame_index]
        image_gray_cv = np.array(image_gray)
        if threshold_type == cv2.THRESH_OTSU:
            threshold_value_used, image_binary_cv = cv2.threshold(
                image_gray_cv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            if binary_invert:
                image_binary_cv = 255 - image_binary_cv
        else:
            threshold_value_used, image_binary_cv = cv2.threshold(
                image_gray_cv, threshold, 255, cv2.THRESH_BINARY
            )
            if binary_invert:
                image_binary_cv = 255 - image_binary_cv
        image_binary_pil = Image.fromarray(image_binary_cv)
        return image_binary_pil, threshold_value_used

    def update_single_binary_frame(self, frame_index, threshold, binary_invert):
        image_binary_pil, threshold_value_used = self.binarize_one_frame(
            frame_index, cv2.THRESH_BINARY, threshold, binary_invert
        )
        self.binary_images[frame_index] = image_binary_pil
        self.threshold_values[frame_index] = threshold_value_used

    def update_multiple_binary_frames(
        self,
        frame_index_from,
        frame_index_to,
        threshold,
        binary_invert,
        progress_callback,
    ):
        if not (0 <= frame_index_from <= frame_index_to < len(self.binary_images)):
            raise ValueError(
                f"Invalid range {frame_index_from}-{frame_index_to} for {len(self.binary_images)} images"
            )
        for i in range(frame_index_from, frame_index_to + 1):
            image_binary_pil, threshold_value_used = self.binarize_one_frame(
                i, cv2.THRESH_BINARY, threshold, binary_invert
            )
            self.binary_images[i] = image_binary_pil
            self.threshold_values[i] = threshold_value_used
            progress_callback(
                image_binary_pil,
                i - frame_index_from,
                frame_index_to - frame_index_from,
            )

    def binarize_all_frames(
        self, threshold_type, threshold, binary_invert, progress_callback
    ):
        self.binary_images.clear()
        self.threshold_values.clear()
        count = len(self.gray_images)
        for i in range(count):
            image_binary_pil, threshold_value_used = self.binarize_one_frame(
                i, threshold_type, threshold, binary_invert
            )
            self.binary_images.append(image_binary_pil)
            self.threshold_values.append(threshold_value_used)
            progress_callback(image_binary_pil, i, len(self.gray_images) - 1)

    def generate_target_file(self, output_path, output_fps, progress_callback):
        if os.path.exists(output_path):
            os.remove(output_path)

        with open(output_path, "wb") as f:
            f.write(struct.pack("I", output_fps))

        for i, image_binary in enumerate(self.binary_images):
            ret = self.image_to_oled_data(image_binary)
            with open(output_path, "ab+") as f:
                f.write(bytearray(ret))

            progress_callback(i, len(self.binary_images) - 1)

    def reverse_bit(self, dat: int) -> int:
        # reference: https://github.com/coloz/image-converter/blob/ee5d55b36db31184d6465328f3a4a6a47199d36e/converter.py#L59
        res = 0
        for _ in range(8):
            res = (res << 1) | (dat & 1)
            dat >>= 1
        return res

    def image_to_oled_data(self, image_binary: Image.Image) -> list[int]:
        # reference: https://github.com/coloz/image-converter/blob/ee5d55b36db31184d6465328f3a4a6a47199d36e/converter.py#L67
        output_data = []
        width, height = image_binary.size
        for y in range(height):
            next_value = 0
            for x in range(width):
                if (x % 8 == 0 or x == width - 1) and x > 0:
                    next_value = self.reverse_bit(next_value)
                    output_data.append(next_value)
                    next_value = 0
                if image_binary.getpixel((x, y)) > 0:
                    next_value += 2 ** (7 - (x % 8))
        return output_data

    def get_current_frame(self):
        index = self.current_frame_index
        if index < 0 or index >= len(self.rgb_images):
            return None, None, None
        image_rgb = self.rgb_images[index]
        image_gray = self.gray_images[index] if index < len(self.gray_images) else None
        image_binary = (
            self.binary_images[index] if index < len(self.binary_images) else None
        )
        return image_rgb, image_gray, image_binary


class View:

    def __init__(self, root: ttk.Window):
        self.root = root

        self.var_current_frame_index = tk.IntVar(value=0)

        self.var_input_type = tk.StringVar(value="video")  # video or image_sequence
        self.var_input_path = tk.StringVar()
        self.var_rotation = tk.IntVar(value=0)  # 0, 90, 180 or 270
        self.var_fps = tk.IntVar(value=30)
        self.var_scale_mode = tk.StringVar(value="fit")  # fit or fill
        self.var_start_time = tk.DoubleVar(value=0)
        self.var_end_time = tk.DoubleVar(value=0)

        self.var_threshold_type = tk.IntVar(
            value=cv2.THRESH_OTSU
        )  # cv2.THRESH_OTSU or cv2.THRESH_BINARY
        self.var_threshold = tk.IntVar(value=127)

        self.var_adjustment_options = tk.StringVar(
            value="single_frame"
        )  # single_frame or multi_frames
        self.var_threshold_adjustment = tk.IntVar(value=127)
        self.var_start_frame = tk.IntVar(value=0)
        self.var_end_frame = tk.IntVar(value=0)

        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        drive, path = os.path.splitdrive(script_dir)
        drive = drive.upper()
        script_dir = (drive + path).replace("\\", "/")
        default_output_path = f"{script_dir}/data/video.data"
        self.var_output_path = tk.StringVar(value=default_output_path)

        self.var_language = tk.StringVar(value="chinese")
        self.var_theme = tk.StringVar(value="light")

        self.root.title(self.get_text("app_title"))

        self.update_theme()
        self.setup_ui()
        self.update_language()

    def get_text(self, key, *args):
        current_language = self.var_language.get()
        text = TRANSLATIONS[current_language].get(key, key)
        if args and isinstance(text, str) and "{}" in text:
            return text.format(*args)
        return text

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

    def create_placeholder_image(self):
        image = Image.new("L", (128, 64), color=200)
        return image

    def create_progress_bar_style(self):
        if not hasattr(self, "progress_bar_style"):
            self.progress_bar_style = ttk.Style()
        primary_color = self.progress_bar_style.colors.primary
        success_color = self.progress_bar_style.colors.success
        self.progress_bar_style.configure(
            "Custom.Horizontal.TProgressbar",
            thickness=0.1,
            troughcolor=self.progress_bar_style.colors.bg,
            background=success_color,
            lightcolor=success_color,
            darkcolor=success_color,
            troughrelief="flat",
            relief="flat",
        )

    def update_theme(self, *args):
        new_theme = self.var_theme.get()
        if new_theme == "dark":
            theme = "darkly"
        elif new_theme == "light":
            theme = "litera"
        self.root.style.theme_use(theme)
        self.create_progress_bar_style()

    def setup_ui(self):
        self.frame_left = ttk.Frame(self.root)
        self.frame_left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.frame_right = ttk.Frame(self.root)
        self.frame_right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # preview section =========================================
        self.placeholder_image = ImageTk.PhotoImage(self.create_placeholder_image())

        self.label_frame_preview = ttk.LabelFrame(
            self.frame_left, text=self.get_text("preview"), padding=5
        )
        self.label_frame_preview.pack(side=tk.TOP, fill=tk.X)

        self.frame_preview_images = ttk.Frame(self.label_frame_preview)
        self.frame_preview_images.pack(side=tk.TOP, fill=tk.X)

        # rgb image
        self.frame_rgb = ttk.Frame(self.frame_preview_images)
        self.frame_rgb.pack(side=tk.LEFT)
        self.label_rgb_image = ttk.Label(self.frame_rgb, image=self.placeholder_image)
        self.label_rgb_image.pack(side=tk.TOP)
        self.label_rgb = ttk.Label(self.frame_rgb, text=self.get_text("rgb"))
        self.label_rgb.pack(side=tk.TOP)

        # gray image
        self.frame_gray = ttk.Frame(self.frame_preview_images)
        self.frame_gray.pack(side=tk.LEFT)
        self.label_gray_image = ttk.Label(self.frame_gray, image=self.placeholder_image)
        self.label_gray_image.pack(side=tk.TOP)
        self.label_gray = ttk.Label(self.frame_gray, text=self.get_text("grayscale"))
        self.label_gray.pack(side=tk.TOP)

        # binary image
        self.frame_binary = ttk.Frame(self.frame_preview_images)
        self.frame_binary.pack(side=tk.LEFT)
        self.label_binary_image = ttk.Label(
            self.frame_binary, image=self.placeholder_image
        )
        self.label_binary_image.pack(side=tk.TOP)
        self.label_binary = ttk.Label(self.frame_binary, text=self.get_text("binary"))
        self.label_binary.pack(side=tk.TOP)

        # current image index
        self.frame_current_frame_index = ttk.Frame(self.label_frame_preview)
        self.frame_current_frame_index.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_current_frame_index = ttk.Label(
            self.frame_current_frame_index, text=self.get_text("current_frame_index")
        )
        self.label_current_frame_index.pack(side=tk.LEFT)
        self.scale_current_frame_index = ttk.Scale(
            self.frame_current_frame_index,
            from_=0,
            to=0,
            variable=self.var_current_frame_index,
        )
        self.scale_current_frame_index.pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=5
        )
        self.label_current_frame_index_value = ttk.Label(
            self.frame_current_frame_index, text="0/0"
        )
        self.label_current_frame_index_value.pack(side=tk.LEFT)

        # navigation buttons
        self.frame_navigation = ttk.Frame(self.label_frame_preview)
        self.frame_navigation.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.button_previous_frame = ttk.Button(
            self.frame_navigation,
            text=self.get_text("previous_frame"),
        )
        self.button_previous_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.button_next_frame = ttk.Button(
            self.frame_navigation, text=self.get_text("next_frame"), bootstyle=PRIMARY
        )
        self.button_next_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

        # input section ============================================
        self.label_frame_input = ttk.LabelFrame(
            self.frame_left, text=self.get_text("input"), padding=5
        )
        self.label_frame_input.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        # input type
        self.frame_input_type = ttk.Frame(self.label_frame_input)
        self.frame_input_type.pack(side=tk.TOP, fill=tk.X)
        self.label_input_type = ttk.Label(
            self.frame_input_type, text=self.get_text("input_type")
        )
        self.label_input_type.pack(side=tk.LEFT)
        self.radio_button_video = ttk.Radiobutton(
            self.frame_input_type,
            text=self.get_text("video"),
            variable=self.var_input_type,
            value="video",
        )
        self.radio_button_video.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_button_image_sequence = ttk.Radiobutton(
            self.frame_input_type,
            text=self.get_text("image_sequence"),
            variable=self.var_input_type,
            value="image_sequence",
        )
        self.radio_button_image_sequence.pack(side=tk.LEFT, padx=(10, 0))

        # input path
        self.frame_input_path = ttk.Frame(self.label_frame_input)
        self.frame_input_path.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.entry_input_path = ttk.Entry(
            self.frame_input_path, textvariable=self.var_input_path
        )
        self.entry_input_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.button_select_input = ttk.Button(
            self.frame_input_path,
            text=self.get_text("select_video"),
        )
        self.button_select_input.pack(side=tk.LEFT, padx=(5, 0))

        # rotation
        self.frame_rotation = ttk.Frame(self.label_frame_input)
        self.frame_rotation.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_rotation = ttk.Label(
            self.frame_rotation, text=self.get_text("rotation")
        )
        self.label_rotation.pack(side=tk.LEFT)
        self.radio_button_0_degree = ttk.Radiobutton(
            self.frame_rotation,
            text="0°",
            variable=self.var_rotation,
            value=0,
        )
        self.radio_button_0_degree.pack(side=tk.LEFT, padx=5)
        self.radio_button_90_degree = ttk.Radiobutton(
            self.frame_rotation,
            text="90°",
            variable=self.var_rotation,
            value=90,
        )
        self.radio_button_90_degree.pack(side=tk.LEFT, padx=5)
        self.radio_button_180_degree = ttk.Radiobutton(
            self.frame_rotation,
            text="180°",
            variable=self.var_rotation,
            value=180,
        )
        self.radio_button_180_degree.pack(side=tk.LEFT, padx=5)
        self.radio_button_270_degree = ttk.Radiobutton(
            self.frame_rotation,
            text="270°",
            variable=self.var_rotation,
            value=270,
        )
        self.radio_button_270_degree.pack(side=tk.LEFT, padx=5)

        # scale mode
        self.frame_scale_mode = ttk.Frame(self.label_frame_input)
        self.frame_scale_mode.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_scale_mode = ttk.Label(
            self.frame_scale_mode, text=self.get_text("scaling_mode")
        )
        self.label_scale_mode.pack(side=tk.LEFT)
        self.radio_button_fit = ttk.Radiobutton(
            self.frame_scale_mode,
            text=self.get_text("fit"),
            variable=self.var_scale_mode,
            value="fit",
        )
        self.radio_button_fit.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_button_fill = ttk.Radiobutton(
            self.frame_scale_mode,
            text=self.get_text("fill"),
            variable=self.var_scale_mode,
            value="fill",
        )
        self.radio_button_fill.pack(side=tk.LEFT, padx=(5, 0))

        # frame rate
        self.frame_fps = ttk.Frame(self.label_frame_input)
        self.frame_fps.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_fps = ttk.Label(self.frame_fps, text=self.get_text("fps"))
        self.label_fps.pack(side=tk.LEFT)
        self.scale_fps = ttk.Scale(
            self.frame_fps,
            from_=1,
            to=30,
            variable=self.var_fps,
        )
        self.scale_fps.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.label_fps_value = ttk.Label(
            self.frame_fps, text=f"{self.var_fps.get()} fps"
        )
        self.label_fps_value.pack(side=tk.LEFT)

        # start time
        self.frame_start_time = ttk.Frame(self.label_frame_input)
        self.frame_start_time.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_start_time = ttk.Label(
            self.frame_start_time,
            text=self.get_text("start_time"),
        )
        self.label_start_time.pack(side=tk.LEFT)
        self.scale_start_time = ttk.Scale(
            self.frame_start_time,
            from_=0,
            to=100,
            variable=self.var_start_time,
        )
        self.scale_start_time.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.label_start_time_value = ttk.Label(
            self.frame_start_time,
            text=self.format_time(self.var_start_time.get()),
        )
        self.label_start_time_value.pack(side=tk.LEFT)

        # end time
        self.frame_end_time = ttk.Frame(self.label_frame_input)
        self.frame_end_time.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_end_time = ttk.Label(
            self.frame_end_time,
            text=self.get_text("end_time"),
        )
        self.label_end_time.pack(side=tk.LEFT)
        self.scale_end_time = ttk.Scale(
            self.frame_end_time,
            from_=0,
            to=100,
            variable=self.var_end_time,
        )
        self.scale_end_time.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.label_end_time_value = ttk.Label(
            self.frame_end_time,
            text=self.format_time(self.var_end_time.get()),
        )
        self.label_end_time_value.pack(side=tk.LEFT)

        # process input button
        self.frame_process_input = ttk.Frame(self.label_frame_input)
        self.frame_process_input.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_process_input.grid_columnconfigure(0, weight=1)
        self.button_process_input = ttk.Button(
            self.frame_process_input,
            text=self.get_text("process_video"),
        )
        self.button_process_input.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.progress_bar_process_input = ttk.Progressbar(
            self.frame_process_input, style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar_process_input.grid(row=0, column=0, sticky="sew")
        self.progress_bar_process_input.grid_remove()

        # binarization section =========================================
        self.label_frame_binarization = ttk.LabelFrame(
            self.frame_left, text=self.get_text("binarization"), padding=5
        )
        self.label_frame_binarization.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        # threshold options
        self.frame_threshold_options = ttk.Frame(self.label_frame_binarization)
        self.frame_threshold_options.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_threshold_options = ttk.Label(
            self.frame_threshold_options, text=self.get_text("threshold_options")
        )
        self.label_threshold_options.pack(side=tk.LEFT)

        # threshold type
        self.radio_button_otsu = ttk.Radiobutton(
            self.frame_threshold_options,
            text=self.get_text("otsu"),
            variable=self.var_threshold_type,
            value=cv2.THRESH_OTSU,
        )
        self.radio_button_otsu.pack(side=tk.LEFT, padx=5)
        self.radio_button_custom = ttk.Radiobutton(
            self.frame_threshold_options,
            text=self.get_text("custom"),
            variable=self.var_threshold_type,
            value=cv2.THRESH_BINARY,
        )
        self.radio_button_custom.pack(side=tk.LEFT, padx=5)

        # binary invert
        self.switch_binary_invert = ttk.Checkbutton(
            self.frame_threshold_options,
            text=self.get_text("binary_invert"),
            bootstyle="success-round-toggle",
        )
        self.switch_binary_invert.pack(side=tk.RIGHT)

        # custom threshold
        self.frame_threshold = ttk.Frame(self.label_frame_binarization)
        self.frame_threshold.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_threshold.pack_forget()

        self.label_threshold = ttk.Label(
            self.frame_threshold, text=self.get_text("threshold")
        )
        self.label_threshold.pack(side=tk.LEFT)
        self.scale_threshold = ttk.Scale(
            self.frame_threshold,
            from_=0,
            to=255,
            variable=self.var_threshold,
        )
        self.scale_threshold.pack(side=tk.LEFT, fill=tk.X, padx=5, expand=True)
        self.label_threshold_value = ttk.Label(
            self.frame_threshold, text=f"{self.var_threshold.get()}"
        )
        self.label_threshold_value.pack(side=tk.RIGHT)

        # button binarization
        self.frame_binarize_all = ttk.Frame(self.label_frame_binarization)
        self.frame_binarize_all.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_binarize_all.grid_columnconfigure(0, weight=1)
        self.button_binarize_all = ttk.Button(
            self.frame_binarize_all,
            text=self.get_text("binarize_all"),
        )
        self.button_binarize_all.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.progress_bar_binarize_all = ttk.Progressbar(
            self.frame_binarize_all, style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar_binarize_all.grid(row=0, column=0, sticky="sew")
        self.progress_bar_binarize_all.grid_remove()

        # binarization adjustment section =========================
        self.label_frame_binarization_adjustment = ttk.LabelFrame(
            self.frame_left, text=self.get_text("binarization_adjustment"), padding=5
        )
        self.label_frame_binarization_adjustment.pack(
            side=tk.TOP, fill=tk.X, pady=(5, 0)
        )

        # adjustment mode
        self.frame_adjustment_options = ttk.Frame(
            self.label_frame_binarization_adjustment
        )
        self.frame_adjustment_options.pack(side=tk.TOP, fill=tk.X)
        self.label_adjustment_options = ttk.Label(
            self.frame_adjustment_options, text=self.get_text("adjustment_options")
        )
        self.label_adjustment_options.pack(side=tk.LEFT)
        self.radio_button_single_frame = ttk.Radiobutton(
            self.frame_adjustment_options,
            text=self.get_text("single_frame"),
            variable=self.var_adjustment_options,
            value="single_frame",
        )
        self.radio_button_single_frame.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_button_multi_frame = ttk.Radiobutton(
            self.frame_adjustment_options,
            text=self.get_text("multi_frames"),
            variable=self.var_adjustment_options,
            value="multi_frames",
        )
        self.radio_button_multi_frame.pack(side=tk.LEFT, padx=(10, 0))

        # threshold adjustment
        self.frame_threshold_adjustment = ttk.Frame(
            self.label_frame_binarization_adjustment
        )
        self.frame_threshold_adjustment.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_threshold_adjustment = ttk.Label(
            self.frame_threshold_adjustment,
            text=self.get_text("threshold"),
        )
        self.label_threshold_adjustment.pack(side=tk.LEFT)
        self.scale_threshold_adjustment = ttk.Scale(
            self.frame_threshold_adjustment,
            from_=0,
            to=255,
            variable=self.var_threshold_adjustment,
        )
        self.scale_threshold_adjustment.pack(
            side=tk.LEFT, fill=tk.X, padx=(5, 0), expand=True
        )
        self.label_threshold_value_adjustment = ttk.Label(
            self.frame_threshold_adjustment,
            text=f"{self.var_threshold_adjustment.get()}",
        )
        self.label_threshold_value_adjustment.pack(side=tk.LEFT, fill=tk.X, padx=(5, 0))
        self.switch_binary_invert_adjustment = ttk.Checkbutton(
            self.frame_threshold_adjustment,
            text=self.get_text("binary_invert"),
            bootstyle="success-round-toggle",
        )
        self.switch_binary_invert_adjustment.pack(side=tk.LEFT, fill=tk.X, padx=(5, 0))

        # target frame range
        self.frame_range_of_frames = ttk.Frame(self.label_frame_binarization_adjustment)
        self.frame_range_of_frames.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_range_of_frames.pack_forget()

        # start frame
        self.frame_start_frame = ttk.Frame(self.frame_range_of_frames)
        self.frame_start_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_start_frame = ttk.Label(
            self.frame_start_frame,
            text=self.get_text("start_frame"),
        )
        self.label_start_frame.pack(side=tk.LEFT)
        self.scale_start_frame = ttk.Scale(
            self.frame_start_frame, from_=0, to=0, variable=self.var_start_frame
        )
        self.scale_start_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.label_start_frame_value = ttk.Label(
            self.frame_start_frame,
            text="0",
        )
        self.label_start_frame_value.pack(side=tk.LEFT)

        # end frame
        self.frame_end_frame = ttk.Frame(self.frame_range_of_frames)
        self.frame_end_frame.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_end_frame = ttk.Label(
            self.frame_end_frame,
            text=self.get_text("end_frame"),
        )
        self.label_end_frame.pack(side=tk.LEFT)
        self.scale_end_frame = ttk.Scale(
            self.frame_end_frame, from_=0, to=0, variable=self.var_end_frame
        )
        self.scale_end_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.label_end_frame_value = ttk.Label(
            self.frame_end_frame,
            text="0",
        )
        self.label_end_frame_value.pack(side=tk.LEFT)

        # adjustment button
        self.frame_perform_adjustments = ttk.Frame(
            self.label_frame_binarization_adjustment
        )
        self.frame_perform_adjustments.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_perform_adjustments.grid_columnconfigure(0, weight=1)
        self.button_perform_adjustments = ttk.Button(
            self.frame_perform_adjustments,
            text=self.get_text("adjust_single_frame"),
        )
        self.button_perform_adjustments.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.progress_bar_perform_adjustments = ttk.Progressbar(
            self.frame_perform_adjustments, style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar_perform_adjustments.grid(row=0, column=0, sticky="sew")
        self.progress_bar_perform_adjustments.grid_remove()

        # output section =========================================
        self.label_frame_output = ttk.LabelFrame(
            self.frame_left, text=self.get_text("output"), padding=5
        )
        self.label_frame_output.pack(side=tk.TOP, fill=tk.X, pady=(5, 5))

        self.frame_output_path = ttk.Frame(self.label_frame_output)
        self.frame_output_path.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        # output path
        self.entry_output_path = ttk.Entry(
            self.frame_output_path, textvariable=self.var_output_path
        )
        self.entry_output_path.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.button_select_output = ttk.Button(
            self.frame_output_path,
            text=self.get_text("select_output_path"),
        )
        self.button_select_output.pack(side=tk.LEFT, padx=(5, 0))

        # generate target file button
        self.frame_generate_target_file = ttk.Frame(self.label_frame_output)
        self.frame_generate_target_file.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.frame_generate_target_file.grid_columnconfigure(0, weight=1)
        self.button_generate_target_file = ttk.Button(
            self.frame_generate_target_file,
            text=self.get_text("generate_target_file"),
        )
        self.button_generate_target_file.grid(row=0, column=0, sticky="ew", pady=(0, 2))
        self.progress_bar_generate_target_file = ttk.Progressbar(
            self.frame_generate_target_file, style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar_generate_target_file.grid(row=0, column=0, sticky="sew")
        self.progress_bar_generate_target_file.grid_remove()

        # settings section =========================================
        self.label_frame_settings = ttk.LabelFrame(
            self.frame_right, text=self.get_text("settings"), padding=5
        )
        self.label_frame_settings.pack(side=tk.TOP, fill=tk.X)

        # language
        self.frame_language = ttk.Frame(self.label_frame_settings)
        self.frame_language.pack(side=tk.TOP, fill=tk.X)
        self.label_language = ttk.Label(
            self.frame_language, text=self.get_text("language")
        )
        self.label_language.pack(side=tk.LEFT)
        self.radio_button_chinese = ttk.Radiobutton(
            self.frame_language,
            text=self.get_text("chinese"),
            variable=self.var_language,
            value="chinese",
        )
        self.radio_button_chinese.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_button_english = ttk.Radiobutton(
            self.frame_language,
            text=self.get_text("english"),
            variable=self.var_language,
            value="english",
        )
        self.radio_button_english.pack(side=tk.LEFT, padx=(5, 0))

        # theme
        self.frame_theme = ttk.Frame(self.label_frame_settings)
        self.frame_theme.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))
        self.label_theme = ttk.Label(self.frame_theme, text=self.get_text("theme"))
        self.label_theme.pack(side=tk.LEFT)
        self.radio_button_dark = ttk.Radiobutton(
            self.frame_theme,
            text=self.get_text("dark"),
            variable=self.var_theme,
            value="dark",
        )
        self.radio_button_dark.pack(side=tk.LEFT, padx=(5, 0))
        self.radio_button_light = ttk.Radiobutton(
            self.frame_theme,
            text=self.get_text("light"),
            variable=self.var_theme,
            value="light",
        )
        self.radio_button_light.pack(side=tk.LEFT, padx=(5, 0))

        # help section =========================================
        self.label_frame_help = ttk.LabelFrame(
            self.frame_right, text=self.get_text("help"), padding=5
        )
        self.label_frame_help.pack(side=tk.TOP, fill=tk.X, pady=(5, 0))

        self.label_usage_guide = tk.Label(
            self.label_frame_help,
            text=self.get_text("usage_guide_video"),
            anchor="w",
            justify="left",
        )
        self.label_usage_guide.pack(side=tk.TOP, fill=tk.X)

    def update_language(self, *args):
        self.root.title(self.get_text("app_title"))

        self.label_frame_preview.config(text=self.get_text("preview"))
        self.label_rgb.config(text=self.get_text("rgb"))
        self.label_gray.config(text=self.get_text("grayscale"))
        self.label_binary.config(text=self.get_text("binary"))
        self.label_current_frame_index.config(text=self.get_text("current_frame_index"))
        self.button_previous_frame.config(text=self.get_text("previous_frame"))
        self.button_next_frame.config(text=self.get_text("next_frame"))

        self.label_frame_input.config(text=self.get_text("input"))
        self.label_input_type.config(text=self.get_text("input_type"))
        self.radio_button_video.config(text=self.get_text("video"))
        self.radio_button_image_sequence.config(text=self.get_text("image_sequence"))
        self.button_select_input.config(
            text=self.get_text(
                "select_video"
                if self.var_input_type.get() == "video"
                else "select_folder"
            )
        )
        self.label_rotation.config(text=self.get_text("rotation"))
        self.label_scale_mode.config(text=self.get_text("scaling_mode"))
        self.radio_button_fit.config(text=self.get_text("fit"))
        self.radio_button_fill.config(text=self.get_text("fill"))
        self.label_fps.config(text=self.get_text("fps"))
        self.label_start_time.config(text=self.get_text("start_time"))
        self.label_end_time.config(text=self.get_text("end_time"))
        self.button_process_input.config(
            text=self.get_text(
                "process_video"
                if self.var_input_type.get() == "video"
                else "process_image_sequence"
            )
        )

        self.label_frame_binarization.config(text=self.get_text("binarization"))
        self.label_threshold_options.config(text=self.get_text("threshold_options"))
        self.radio_button_otsu.config(text=self.get_text("otsu"))
        self.radio_button_custom.config(text=self.get_text("custom"))
        self.switch_binary_invert.config(text=self.get_text("binary_invert"))
        self.label_threshold.config(text=self.get_text("threshold"))
        self.button_binarize_all.config(text=self.get_text("binarize_all"))

        self.label_frame_binarization_adjustment.config(
            text=self.get_text("binarization_adjustment")
        )
        self.label_adjustment_options.config(text=self.get_text("adjustment_options"))
        self.radio_button_single_frame.config(text=self.get_text("single_frame"))
        self.radio_button_multi_frame.config(text=self.get_text("multi_frames"))
        self.label_threshold_adjustment.config(text=self.get_text("threshold"))
        self.switch_binary_invert_adjustment.config(text=self.get_text("binary_invert"))
        self.label_start_frame.config(text=self.get_text("start_frame"))
        self.label_end_frame.config(text=self.get_text("end_frame"))
        self.button_perform_adjustments.config(
            text=(
                self.get_text("adjust_single_frame")
                if self.var_adjustment_options.get() == "single_frame"
                else self.get_text("adjust_multi_frames")
            )
        )

        self.label_frame_output.config(text=self.get_text("output"))
        self.button_select_output.config(text=self.get_text("select_output_path"))
        self.button_generate_target_file.config(
            text=self.get_text("generate_target_file")
        )

        self.label_frame_settings.config(text=self.get_text("settings"))
        self.label_language.config(text=self.get_text("language"))
        self.radio_button_chinese.config(text=self.get_text("chinese"))
        self.radio_button_english.config(text=self.get_text("english"))
        self.label_theme.config(text=self.get_text("theme"))
        self.radio_button_dark.config(text=self.get_text("dark"))
        self.radio_button_light.config(text=self.get_text("light"))

        self.label_frame_help.config(text=self.get_text("help"))
        if self.var_input_type.get() == "video":
            self.label_usage_guide.config(text=self.get_text("usage_guide_video"))
        else:
            self.label_usage_guide.config(
                text=self.get_text("usage_guide_image_sequence")
            )

    def update_current_frame_index(self, *args):
        max_index = int(self.scale_current_frame_index["to"]) + 1
        max_index_width = len(str(max_index))
        index = self.var_current_frame_index.get() + 1
        self.label_current_frame_index_value.config(
            text=f"{index:>{max_index_width}}/{max_index}"
        )

    def update_input_type(self, *args):
        self.var_input_path.set("")

        self.var_start_time.set(0)
        self.update_start_time()

        self.var_end_time.set(0)
        self.update_end_time()

        self.var_current_frame_index.set(0)
        self.scale_current_frame_index.config(to=0)
        self.update_current_frame_index()

        self.progress_bar_process_input["value"] = 0
        self.progress_bar_binarize_all["value"] = 0
        self.progress_bar_perform_adjustments["value"] = 0
        self.progress_bar_generate_target_file["value"] = 0

        if self.var_input_type.get() == "video":
            self.button_select_input.config(text=self.get_text("select_video"))
            self.frame_start_time.pack(
                side=tk.TOP, fill=tk.X, pady=(5, 0), after=self.frame_fps
            )
            self.frame_end_time.pack(
                side=tk.TOP,
                fill=tk.X,
                pady=(5, 0),
                after=self.frame_start_time,
            )
            self.button_process_input.config(text=self.get_text("process_video"))
            self.label_usage_guide.config(text=self.get_text("usage_guide_video"))
        else:
            self.button_select_input.config(text=self.get_text("select_folder"))
            self.frame_start_time.pack_forget()
            self.frame_end_time.pack_forget()
            self.button_process_input.config(
                text=self.get_text("process_image_sequence")
            )
            self.label_usage_guide.config(
                text=self.get_text("usage_guide_image_sequence")
            )

    def update_fps(self, *args):
        to = int(self.scale_fps["to"])
        to_width = len(str(to))
        self.label_fps_value.config(text=f"{self.var_fps.get():>{to_width}} fps")

    def update_start_time(self, *args):
        start_time = self.var_start_time.get()
        end_time = self.var_end_time.get()
        if start_time >= end_time:
            self.scale_start_time.set(end_time)
            start_time = end_time
        self.label_start_time_value.config(text=self.format_time(start_time))

    def update_end_time(self, *args):
        start_time = self.var_start_time.get()
        end_time = self.var_end_time.get()
        if end_time <= start_time:
            self.scale_end_time.set(start_time)
            end_time = start_time
        self.label_end_time_value.config(text=self.format_time(end_time))

    def update_threshold_type(self, *args):
        if self.var_threshold_type.get() == cv2.THRESH_OTSU:
            self.frame_threshold.pack_forget()
        else:
            self.frame_threshold.pack(
                side=tk.TOP,
                fill=tk.X,
                pady=(5, 0),
                before=self.frame_binarize_all,
            )

    def update_threshold(self, *args):
        self.label_threshold_value.config(text=f"{self.var_threshold.get()}")

    def update_adjustment_options(self, *args):
        if self.var_adjustment_options.get() == "multi_frames":
            self.frame_range_of_frames.pack(
                side=tk.TOP,
                fill=tk.X,
                pady=(5, 0),
                before=self.frame_perform_adjustments,
            )
            self.button_perform_adjustments.config(
                text=self.get_text("adjust_multi_frames")
            )
        else:
            self.frame_range_of_frames.pack_forget()
            self.button_perform_adjustments.config(
                text=self.get_text("adjust_single_frame")
            )

    def update_threshold_adjustment(self, *args):
        self.label_threshold_value_adjustment.config(
            text=f"{self.var_threshold_adjustment.get()}"
        )

    def update_start_frame(self, *args):
        start_frame = self.var_start_frame.get()
        end_frame = self.var_end_frame.get()
        if start_frame >= end_frame:
            self.scale_start_frame.set(end_frame)
            start_frame = end_frame

        to = int(self.scale_start_frame["to"]) + 1
        to_width = len(str(to))
        self.label_start_frame_value.config(text=f"{start_frame + 1:>{to_width}}")

    def update_end_frame(self, *args):
        start_frame = self.var_start_frame.get()
        end_frame = self.var_end_frame.get()
        if end_frame <= start_frame:
            self.scale_end_frame.set(start_frame)
            end_frame = start_frame

        to = int(self.scale_end_frame["to"]) + 1
        to_width = len(str(to))
        self.label_end_frame_value.config(text=f"{end_frame + 1:>{to_width}}")

    def update_rgb_image(self, image: Image = None):
        if image is None:
            photo = self.placeholder_image
        else:
            photo = ImageTk.PhotoImage(image)
            self.label_rgb_image.image = photo
        self.label_rgb_image.config(image=photo)
        self.label_rgb_image.update()

    def update_gray_image(self, image: Image = None):
        if image is None:
            photo = self.placeholder_image
        else:
            photo = ImageTk.PhotoImage(image)
            self.label_gray_image.image = photo
        self.label_gray_image.config(image=photo)
        self.label_gray_image.update()

    def update_binary_image(self, image: Image = None):
        if image is None:
            photo = self.placeholder_image
        else:
            photo = ImageTk.PhotoImage(image)
            self.label_binary_image.image = photo
        self.label_binary_image.config(image=photo)
        self.label_binary_image.update()

    def show_error(self, message_key, *args):
        messagebox.showerror(self.get_text("error"), self.get_text(message_key, *args))

    def show_info(self, message_key, *args):
        messagebox.showinfo(self.get_text("info"), self.get_text(message_key, *args))


class ViewModel:

    def __init__(self, model: Model, view: View):
        self.model = model
        self.view = view
        self.bind_events()

    def bind_events(self):
        self.view.var_current_frame_index.trace_add("write", self.update_current_frame)
        self.view.button_previous_frame.config(command=self.previous_frame)
        self.view.button_next_frame.config(command=self.next_frame)
        self.view.root.bind("<KeyPress-Left>", self.previous_frame)
        self.view.root.bind("<KeyPress-Right>", self.next_frame)

        self.view.var_input_type.trace_add("write", self.update_input_type)
        self.view.var_fps.trace_add("write", self.view.update_fps)
        self.view.var_start_time.trace_add("write", self.view.update_start_time)
        self.view.var_end_time.trace_add("write", self.view.update_end_time)
        self.view.entry_input_path.bind(
            "<Return>", lambda event: self.update_video_duration_and_fps()
        )
        self.view.entry_input_path.bind(
            "<FocusOut>", lambda event: self.update_video_duration_and_fps()
        )
        self.view.button_select_input.config(command=self.select_input)
        self.view.button_process_input.config(command=self.process_input)

        self.view.var_threshold_type.trace_add("write", self.update_threshold_type)
        self.view.switch_binary_invert.config(command=self.handle_binary_invert)
        self.view.var_threshold.trace_add("write", self.update_threshold)
        self.view.button_binarize_all.config(command=self.binarize_all)

        self.view.var_adjustment_options.trace_add(
            "write", self.view.update_adjustment_options
        )
        self.view.var_threshold_adjustment.trace_add(
            "write", self.update_threshold_adjustment
        )
        self.view.switch_binary_invert_adjustment.config(
            command=self.handle_binary_invert_adjustment
        )
        self.view.var_start_frame.trace_add("write", self.view.update_start_frame)
        self.view.var_end_frame.trace_add("write", self.view.update_end_frame)
        self.view.button_perform_adjustments.config(command=self.perform_adjustments)

        self.view.button_select_output.config(command=self.select_output)
        self.view.button_generate_target_file.config(command=self.generate_target_file)

        self.view.var_language.trace_add("write", self.view.update_language)
        self.view.var_theme.trace_add("write", self.view.update_theme)

    def update_current_frame(self, *args):
        self.view.update_current_frame_index()
        index = self.view.var_current_frame_index.get()
        self.model.current_frame_index = index
        current_frame = self.model.get_current_frame()
        self.view.update_rgb_image(current_frame[0])
        self.view.update_gray_image(current_frame[1])
        self.view.update_binary_image(current_frame[2])

    def previous_frame(self, *args):
        if len(self.model.rgb_images) == 0:
            self.view.show_error(
                "click_process_video_first"
                if self.view.var_input_type.get() == "video"
                else "click_process_image_sequence_first"
            )
            return

        if self.model.current_frame_index > 0:
            self.model.current_frame_index -= 1
        else:
            return

        self.view.var_current_frame_index.set(self.model.current_frame_index)

        if self.model.current_frame_index < len(self.model.binary_images):
            index = self.model.current_frame_index
            current_frame_threshold = self.model.threshold_values[index]
            self.view.var_threshold_adjustment.set(current_frame_threshold)

        current_frame = self.model.get_current_frame()
        self.view.update_rgb_image(current_frame[0])
        self.view.update_gray_image(current_frame[1])
        self.view.update_binary_image(current_frame[2])

    def next_frame(self, *args):
        if len(self.model.rgb_images) == 0:
            self.view.show_error(
                "click_process_video_first"
                if self.view.var_input_type.get() == "video"
                else "click_process_image_sequence_first"
            )
            return

        if self.model.current_frame_index < len(self.model.rgb_images) - 1:
            self.model.current_frame_index += 1
        else:
            return

        self.view.var_current_frame_index.set(self.model.current_frame_index)

        if self.model.current_frame_index < len(self.model.binary_images):
            index = self.model.current_frame_index
            current_frame_threshold = self.model.threshold_values[index]
            self.view.var_threshold_adjustment.set(current_frame_threshold)

        current_frame = self.model.get_current_frame()
        self.view.update_rgb_image(current_frame[0])
        self.view.update_gray_image(current_frame[1])
        self.view.update_binary_image(current_frame[2])

    def update_input_type(self, *args):
        self.model.reset()

        self.view.update_input_type()

        self.view.scale_current_frame_index.config(to=0)
        self.view.var_current_frame_index.set(0)

        self.view.scale_start_time.config(to=0)
        self.view.var_start_time.set(0)

        self.view.scale_end_time.config(to=0)
        self.view.var_end_time.set(0)

        self.view.scale_fps.config(to=30)
        self.view.var_fps.set(30)

        self.view.progress_bar_process_input.config(value=0, maximum=0)
        self.view.progress_bar_binarize_all.config(value=0, maximum=0)
        self.view.progress_bar_perform_adjustments.config(value=0, maximum=0)
        self.view.progress_bar_generate_target_file.config(value=0, maximum=0)

    def update_video_duration_and_fps(self):
        if (
            self.view.var_input_path.get() == ""
            or self.view.var_input_type.get() != "video"
        ):
            return
        self.model.update_video_duration_and_fps(self.view.var_input_path.get())

        self.view.scale_start_time.config(to=self.model.video_duration)
        self.view.var_start_time.set(0)

        self.view.scale_end_time.config(to=self.model.video_duration)
        self.view.var_end_time.set(self.model.video_duration)

        self.view.scale_fps.config(to=int(self.model.video_fps))
        self.view.var_fps.set(int(self.model.video_fps))

    def select_input(self):
        input_type = self.view.var_input_type.get()
        if input_type == "video":
            input_path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
            )
            if not input_path:
                return
            self.view.var_input_path.set(input_path)
            self.update_video_duration_and_fps()
        else:
            input_path = filedialog.askdirectory()
            if not input_path:
                return
            self.view.var_input_path.set(input_path)

    def process_input(self):
        input_type = self.view.var_input_type.get()
        input_path = self.view.var_input_path.get()

        if input_path == "":
            self.view.show_error(
                "no_video_selected" if input_type == "video" else "no_folder_selected"
            )
            return

        start_time = self.view.var_start_time.get()
        end_time = self.view.var_end_time.get()
        if start_time >= end_time and input_type == "video":
            self.view.show_error("invalid_time_range")
            return

        rotation = self.view.var_rotation.get()
        scaling_mode = self.view.var_scale_mode.get()
        fps = self.view.var_fps.get()

        def progress_callback(frame_rgb_pil, frame_gray_pil, value, maximum):
            self.view.progress_bar_process_input.config(value=value, maximum=maximum)
            self.view.scale_current_frame_index.config(to=maximum)
            self.view.var_current_frame_index.set(value)

        self.view.progress_bar_process_input.grid()
        self.view.button_process_input.config(
            state=tk.DISABLED, text=self.view.get_text("processing")
        )

        if input_type == "video":
            self.model.process_video(
                input_path,
                fps,
                start_time,
                end_time,
                rotation,
                scaling_mode,
                progress_callback,
            )
        else:
            self.model.process_image_sequence(
                input_path, rotation, scaling_mode, progress_callback
            )

        self.view.progress_bar_process_input.grid_remove()
        self.view.button_process_input.config(
            state=tk.NORMAL,
            text=self.view.get_text(
                "process_video" if input_type == "video" else "process_image_sequence"
            ),
        )

        self.model.current_frame_index = 0
        self.view.var_current_frame_index.set(0)

        if len(self.model.rgb_images) == 0:
            self.view.show_error(
                "click_process_video_first"
                if self.view.var_input_type.get() == "video"
                else "click_process_image_sequence_first"
            )
            return

        self.view.show_info("process_completed")

    def update_threshold_type(self, *args):
        self.view.update_threshold_type()
        if len(self.model.gray_images) == 0:
            return
        index = self.model.current_frame_index
        thereshold = self.view.var_threshold.get()
        threshold_type = self.view.var_threshold_type.get()
        binary_invert = self.view.switch_binary_invert.instate(["selected"])
        image_binary_pil, _ = self.model.binarize_one_frame(
            index, threshold_type, thereshold, binary_invert
        )
        self.view.update_binary_image(image_binary_pil)

    def handle_binary_invert(self):
        if len(self.model.gray_images) == 0:
            return
        index = self.model.current_frame_index
        thereshold = self.view.var_threshold.get()
        threshold_type = self.view.var_threshold_type.get()
        binary_invert = self.view.switch_binary_invert.instate(["selected"])
        image_binary_pil, _ = self.model.binarize_one_frame(
            index, threshold_type, thereshold, binary_invert
        )
        self.view.update_binary_image(image_binary_pil)

    def update_threshold(self, *args):
        self.view.update_threshold()
        if len(self.model.gray_images) == 0:
            return
        index = self.model.current_frame_index
        thereshold = self.view.var_threshold.get()
        threshold_type = self.view.var_threshold_type.get()
        binary_invert = self.view.switch_binary_invert.instate(["selected"])
        image_binary_pil, _ = self.model.binarize_one_frame(
            index, threshold_type, thereshold, binary_invert
        )
        self.view.update_binary_image(image_binary_pil)

    def binarize_all(self):
        if len(self.model.gray_images) == 0:
            self.view.show_error(
                "click_process_video_first"
                if self.view.var_input_type.get() == "video"
                else "click_process_image_sequence_first"
            )
            return

        def progress_callback(image_binary, value, maximum):
            self.view.progress_bar_binarize_all.config(value=value, maximum=maximum)
            self.view.var_current_frame_index.set(value)

        self.view.progress_bar_binarize_all.grid()
        self.view.button_binarize_all.config(
            state=tk.DISABLED, text=self.view.get_text("processing")
        )

        self.model.binarize_all_frames(
            self.view.var_threshold_type.get(),
            self.view.var_threshold.get(),
            self.view.switch_binary_invert.instate(["selected"]),
            progress_callback,
        )
        self.view.progress_bar_binarize_all.grid_remove()
        self.view.button_binarize_all.config(
            state=tk.NORMAL, text=self.view.get_text("binarize_all")
        )

        self.model.current_frame_index = 0
        self.view.var_current_frame_index.set(0)

        self.view.var_threshold_adjustment.set(self.model.threshold_values[0])
        self.view.scale_start_frame.config(to=len(self.model.binary_images) - 1)
        self.view.var_start_frame.set(0)
        self.view.scale_end_frame.config(to=len(self.model.binary_images) - 1)
        self.view.var_end_frame.set(len(self.model.binary_images) - 1)

        self.view.show_info("binarization_completed")

    def update_threshold_adjustment(self, *args):
        self.view.update_threshold_adjustment()
        if len(self.model.gray_images) == 0:
            return
        index = self.model.current_frame_index
        thereshold = self.view.var_threshold_adjustment.get()
        binary_invert = self.view.switch_binary_invert_adjustment.instate(["selected"])
        image_binary_pil, _ = self.model.binarize_one_frame(
            index, cv2.THRESH_BINARY, thereshold, binary_invert
        )
        self.view.update_binary_image(image_binary_pil)

    def handle_binary_invert_adjustment(self):
        if len(self.model.gray_images) == 0:
            return
        index = self.model.current_frame_index
        thereshold = self.view.var_threshold_adjustment.get()
        binary_invert = self.view.switch_binary_invert_adjustment.instate(["selected"])
        image_binary_pil, _ = self.model.binarize_one_frame(
            index, cv2.THRESH_BINARY, thereshold, binary_invert
        )
        self.view.update_binary_image(image_binary_pil)

    def perform_adjustments(self):
        if len(self.model.gray_images) == 0:
            self.view.show_error(
                "click_process_video_first"
                if self.view.var_input_type.get() == "video"
                else "click_process_image_sequence_first"
            )
            return

        if len(self.model.binary_images) == 0:
            self.view.show_error("click_binarize_all_first")
            return

        self.view.progress_bar_perform_adjustments.grid()
        self.view.button_perform_adjustments.config(
            state=tk.DISABLED, text=self.view.get_text("processing")
        )

        option = self.view.var_adjustment_options.get()
        if option == "single_frame":
            frame_index = self.model.current_frame_index
            self.model.update_single_binary_frame(
                frame_index,
                self.view.var_threshold_adjustment.get(),
                self.view.switch_binary_invert_adjustment.instate(["selected"]),
            )
            self.view.var_current_frame_index.set(frame_index)
        else:
            start_frame = self.view.var_start_frame.get()
            end_frame = self.view.var_end_frame.get()
            threshold = self.view.var_threshold_adjustment.get()
            invert = self.view.switch_binary_invert_adjustment.instate(["selected"])

            def progress_callback(image_binary, value, maximum):
                self.view.progress_bar_perform_adjustments.config(
                    value=value, maximum=maximum
                )
                self.view.var_current_frame_index.set(start_frame + value)

            self.model.update_multiple_binary_frames(
                start_frame,
                end_frame,
                threshold,
                invert,
                progress_callback,
            )

            self.view.var_current_frame_index.set(start_frame)

        self.view.progress_bar_perform_adjustments.grid_remove()
        self.view.button_perform_adjustments.config(
            state=tk.NORMAL,
            text=self.view.get_text(
                "adjust_single_frame"
                if option == "single_frame"
                else "adjust_multi_frames"
            ),
        )

        self.view.show_info("adjustment_completed")

    def select_output(self):
        output_filename = filedialog.asksaveasfilename(
            defaultextension=".data",
            filetypes=[("Data files", "*.data")],
            initialfile="video.data",
        )
        if output_filename:
            self.view.var_output_path.set(output_filename)

    def generate_target_file(self):
        if len(self.model.binary_images) == 0:
            self.view.show_error("click_binarize_all_first")
            return

        output_filename = self.view.var_output_path.get()
        if output_filename == "":
            self.view.show_error("no_output_path_selected")
            return

        def progress_callback(value, maximum):
            self.view.progress_bar_generate_target_file.config(
                value=value, maximum=maximum
            )
            self.view.var_current_frame_index.set(value)

        self.view.progress_bar_generate_target_file.grid()
        self.view.button_generate_target_file.config(
            state=tk.DISABLED, text=self.view.get_text("processing")
        )

        self.model.generate_target_file(
            output_filename, self.view.var_fps.get(), progress_callback
        )

        self.view.progress_bar_generate_target_file.grid_remove()
        self.view.button_generate_target_file.config(
            state=tk.NORMAL, text=self.view.get_text("generate_target_file")
        )

        self.view.show_info("file_generation_success")


def main():
    root = ttk.Window()
    model = Model()
    view = View(root)
    ViewModel(model, view)
    root.mainloop()


if __name__ == "__main__":
    main()
