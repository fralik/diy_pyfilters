from calendar import c
from concurrent.futures import process
from tkinter import SEL
from typing import Tuple
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import pyvirtualcam
import numpy as np
import mediapipe as mp

from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Signal, Slot, QThreadPool

import winenumerator


def sepia(src_image):
    gray = cv2.cvtColor(src_image, cv2.COLOR_BGR2GRAY)
    normalized_gray = np.array(gray, np.float32) / 255
    # solid color
    sepia = np.ones(src_image.shape)
    sepia[:, :, 0] *= 153  # B
    sepia[:, :, 1] *= 204  # G
    sepia[:, :, 2] *= 255  # R
    # hadamard
    sepia[:, :, 0] *= normalized_gray  # B
    sepia[:, :, 1] *= normalized_gray  # G
    sepia[:, :, 2] *= normalized_gray  # R
    return np.array(sepia, np.uint8)


def shift_image(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    img = np.roll(img, dy, axis=0)
    img = np.roll(img, dx, axis=1)
    if dy > 0:
        img[:dy, :] = 0
    elif dy < 0:
        img[dy:, :] = 0
    if dx > 0:
        img[:, :dx] = 0
    elif dx < 0:
        img[:, dx:] = 0
    return img


def stylization(frame):
    image_blur = cv2.GaussianBlur(frame, (15, 15), 0, 0)
    image_style = cv2.stylization(image_blur, sigma_s=60, sigma_r=0.2)
    return image_style


def opencv_list_cameras(used_camera: int = -1) -> list[str]:
    index = 0
    results = []
    max_gap = 3
    cur_gap = 0

    while index < 1e7:
        if index == used_camera:
            index += 1
            cur_gap = 0
            results.append(str(used_camera))
            continue

        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

        if not cap.read()[0]:
            if cur_gap == max_gap:
                break
            cur_gap += 1
        else:
            results.append(str(index))
        cap.release()
        index += 1
    return results


class VideoEffect(ABC):
    """Interface for all video effects"""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    @abstractmethod
    def run(self, frame: np.ndarray, stopped: bool = False) -> np.ndarray:
        pass


class RedEyeEffect(VideoEffect):
    def __init__(self, **kwargs):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def run(self, frame: np.array, stopped: bool = False) -> np.array:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        for x, y, w, h in faces:
            if stopped:
                return frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[x : x + w, y : y + w]
            roi_color = frame[x : x + h, y : y + w]

            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.15, 4)
            if len(eyes) == 2:
                # sort eyes by x
                eyes = sorted(eyes, key=lambda x: x[0], reverse=True)
                # draw red circle instead of an eye
                (ex, ey, ew, eh) = eyes[0]
                cv2.circle(
                    roi_color, (ex + ew // 2, ey + eh // 2), ew // 2, (0, 0, 255), -1
                )

        return frame


class SepiaEffect(VideoEffect):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, frame: np.array, stopped: bool = False) -> np.array:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        normalized_gray = np.array(gray, np.float32) / 255
        # solid color
        sepia = np.ones(frame.shape)
        sepia[:, :, 0] *= 153  # B
        sepia[:, :, 1] *= 204  # G
        sepia[:, :, 2] *= 255  # R
        # hadamard
        sepia[:, :, 0] *= normalized_gray  # B
        sepia[:, :, 1] *= normalized_gray  # G
        sepia[:, :, 2] *= normalized_gray  # R
        return np.array(sepia, np.uint8)


class NoneEffect(VideoEffect):
    def run(self, frame: np.ndarray, stopped: bool = False) -> np.ndarray:
        return frame


class StarWarsEffect(VideoEffect):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # model_path = Path("./data")
        # model_path.mkdir(exist_ok=True, parents=True)
        # self.bodypix_model = load_model(download_model(
        #    BodyPixModelPaths.MOBILENET_FLOAT_50_STRIDE_16
        # ))
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segment = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=0)
        self.down_factor = kwargs.get("down_factor", 0.6)
        self.background_img = cv2.imread("star_wars_background.jpg")
        print(f"background_imd: {self.background_img.shape}")

    def _post_process_mask(_, mask):
        mask = cv2.dilate(mask, np.ones((10, 10), np.uint8), iterations=1)
        mask = cv2.blur(mask.astype(float), (30, 30))
        return mask

    def _get_mask(self, frame, threshold: float = 0.5) -> np.ndarray:
        results = self.segment.process(frame)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image_seg_mask = results.segmentation_mask
        binary_mask = (image_seg_mask > threshold).astype(np.uint8)
        return binary_mask

    def _hologram_effect(_, img):
        # add a blue tint
        holo = cv2.applyColorMap(img, cv2.COLORMAP_WINTER)
        # add a halftone effect
        bandLength, bandGap = 2, 3
        for y in range(holo.shape[0]):
            if y % (bandLength + bandGap) < bandLength:
                holo[y, :, :] = holo[y, :, :] * np.random.uniform(0.1, 0.3)
        # add some ghosting
        holo_blur = cv2.addWeighted(holo, 0.2, shift_image(holo.copy(), 5, 5), 0.8, 0)
        holo_blur = cv2.addWeighted(
            holo_blur, 0.4, shift_image(holo.copy(), -5, -5), 0.6, 0
        )
        # combine with the original color, oversaturated
        out = cv2.addWeighted(img, 0.5, holo_blur, 0.6, 0)
        return out

    def run(self, frame: np.ndarray, stopped: bool = False) -> np.ndarray:
        if stopped:
            return frame

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mask = self._get_mask(frame)
        mask = self._post_process_mask(mask)

        frame = self._hologram_effect(frame)

        frame_down = cv2.resize(
            frame,
            None,
            fx=self.down_factor,
            fy=self.down_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        mask_down = cv2.resize(
            mask,
            None,
            fx=self.down_factor,
            fy=self.down_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        background_scaled = cv2.resize(
            self.background_img, (frame.shape[1], frame.shape[0])
        )

        foreground_down = np.zeros(frame_down.shape, dtype=frame.dtype)
        full_frame = np.zeros(frame.shape, dtype=frame.dtype)
        full_mask = np.zeros(mask.shape, dtype=mask.dtype)
        for c in range(frame_down.shape[2]):
            foreground_down[:, :, c] = frame_down[:, :, c] * mask_down

        y_start = full_frame.shape[0] - foreground_down.shape[0]
        x_start = int(full_frame.shape[1] / 2 - frame_down.shape[1] / 2)
        x_end = int(full_frame.shape[1] / 2 + frame_down.shape[1] / 2)
        full_frame[y_start:, x_start:x_end, :] = foreground_down
        full_mask[y_start:, x_start:x_end] = mask_down

        # composite the foreground and background
        inv_mask = 1 - full_mask
        for c in range(full_frame.shape[2]):
            full_frame[:, :, c] = (
                full_frame[:, :, c] + background_scaled[:, :, c] * inv_mask
            )
        return full_frame


class CameraReader(QObject):
    """Class that reads from a camera and emits frames as QImage

    This class has a slot for capturing camera change
    """

    def __init__(self, camera_id: str = ""):
        super().__init__()
        self.stopped = True
        self.thread_manager = QThreadPool()
        self._cap = None
        self._virtual = None
        self.origin_width = -1
        self.origin_height = -1
        self.origin_fps = 0
        if camera_id:
            self.set_camera(camera_id)

        self.effect = "None"
        self._effects = {
            "None": NoneEffect(),
            "Sepia": SepiaEffect(),
            "RedEye": RedEyeEffect(),
            "Star Wars": StarWarsEffect(),
        }

    def stop(self):
        self.stopped = True
        self.thread_manager.waitForDone()

    def start(self):
        self.stopped = False
        self.thread_manager.start(self._read_camera_wrapper)

    def _read_camera_wrapper(self):
        while not self.stopped:
            self._read_camera()
            time.sleep(0.01)

        print(f"[reader] Stop camera reader in wrapper")
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._virtual:
            self._virtual.close()
            self._virtual = None

    def _read_camera(self):
        if not self._cap or not self._virtual:
            # print(f"[reader] No camera or virtual camera")
            return

        ret, frame = self._cap.read()
        if not ret:
            return

        # print(f"[reader] Read frame: {frame.shape}")
        # print(f"[reader] virtual: {self._virtual.width}x{self._virtual.height} @ {self._virtual.fps}fps")

        if self.effect in self._effects:
            processed = self._effects[self.effect].run(frame, self.stopped)
        else:
            processed = frame

        self._virtual.send(processed)
        self._virtual.sleep_until_next_frame()

    @Slot(str)
    def set_camera(self, camera: str):
        print(f"[reader] Set camera: {camera}")
        if "None" in camera:
            return

        camera_id = [x.strip() for x in camera.split(",")][0]

        if self._cap is not None:
            self._cap.release()
        if self._virtual is not None:
            self._virtual.close()

        self._cap = cv2.VideoCapture(int(camera_id), cv2.CAP_DSHOW)
        # self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.origin_width)
        # self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.origin_height)

        self.origin_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.origin_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.origin_fps = int(self._cap.get(cv2.CAP_PROP_FPS))
        if self.origin_fps == 0:
            self.origin_fps = 30

        print(
            f"[reader] origin: {self.origin_width}x{self.origin_height} @ {self.origin_fps}fps"
        )

        if not self._virtual:
            self._virtual = pyvirtualcam.Camera(
                width=int(self.origin_width),
                height=int(self.origin_height),
                fps=int(self.origin_fps),
                fmt=pyvirtualcam.PixelFormat.BGR,
                backend="unitycapture",
            )
            print(
                f"[reader] virtual inited: {self._virtual.width}x{self._virtual.height} @ {self._virtual.fps}fps"
            )

    @Slot(str)
    def set_effect(self, effect_name: str):
        self.effect = effect_name


class CameraEnumerator(QObject):
    updated = Signal(list, arguments=["cameras"])

    def __init__(self, cameras: list[Tuple[int, str, Tuple[int, int]]]):
        super().__init__()
        self._init_cameras = cameras
        self._cameras = []
        self.thread_manager = QThreadPool()
        self.stopped = False
        self.used_camera = -1

    @Slot(str)
    def setUsedCamera(self, camera: str):
        self.used_camera = int(camera)

    def stop(self):
        self.stopped = True
        self.thread_manager.waitForDone()

    def start(self):
        self.stopped = False
        self.thread_manager.start(self._update_cameras_wrapper)

    def _update_cameras_wrapper(self):
        while not self.stopped:
            self._update_cameras()
            time.sleep(10)

    def _emit_cameras(self):
        cameras_signal = [f"{x[0]}, {x[1]}" for x in self._cameras]
        print(f"Emitting cameras: {cameras_signal}")
        self.updated.emit(cameras_signal)

    def _update_cameras(self):
        if self._init_cameras:
            # print(f"Emitting init cameras: {self._init_cameras}")
            self._cameras = self._init_cameras
            self._init_cameras = []
            self._emit_cameras()
            return

        # results = opencv_list_cameras(self.used_camera)
        # cameras = [x for x in winenumerator.list_cameras() if filter_camera(x)]
        cameras = [x for x in winenumerator.list_cameras() if filter_camera(x)]

        if self._cameras != cameras:
            print(f"Cameras updated: {cameras}")
            self._cameras = cameras
            self._emit_cameras()


def filter_camera(camera: Tuple[int, str]) -> bool:
    """Filter some cameras in order to avoid a long list"""
    checks = [
        # Do not include Unity camera as a source
        not "Unity"
        in camera[1],
    ]
    return all(checks)


def main():
    # print('wait for input')
    #  input()

    cameras = [x for x in winenumerator.list_cameras() if filter_camera(x)]

    # cameras = [x for x in winenumerator.list_cameras() if filter_camera(x)]
    # cameras.sort(key=lambda x: (x[0], x[1]))
    # for camera in cameras:
    #    print(camera)

    # cameras = opencv_list_cameras()

    app = QGuiApplication(sys.argv)

    engine = QQmlApplicationEngine()
    engine.quit.connect(app.quit)
    engine.load("app.qml")

    camera_enumerator = CameraEnumerator(cameras)
    camera_reader = CameraReader()
    engine.rootObjects()[0].setProperty("enumerator", camera_enumerator)
    engine.rootObjects()[0].setProperty("reader", camera_reader)

    camera_enumerator.start()
    camera_reader.start()

    ret_code = app.exec()
    camera_enumerator.stop()
    camera_reader.stop()
    sys.exit(ret_code)


if __name__ == "__main__":
    main()
