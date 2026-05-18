"""
透视变换标定工具
用法: python calibrate_perspective.py <图片路径> [config.json]

流程:
1. 屏幕上依次点击 左上→右上→右下→左下 四个点
2. 输入鸟瞰图输出尺寸
3. 输入所选区域的物理尺寸（米）→ 自动计算 PPM
4. 预览鸟瞰效果，确认/重选
5. 更新 config.json 的 "calib" 字段（不改动其他字段）
"""

import cv2
import numpy as np
import sys
import json
import os


class Calibrator:
    def __init__(self, img_path, config_path):
        self.img = cv2.imread(img_path)
        if self.img is None:
            print(f"无法读取图片: {img_path}")
            sys.exit(1)

        self.h, self.w = self.img.shape[:2]
        self.src_pts = []
        self.config_path = config_path

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.src_pts) < 4:
            self.src_pts.append((x, y))
            print(f"  点 {len(self.src_pts)}: ({x}, {y})")
            cv2.circle(self.display, (x, y), 6, (0, 0, 255), -1)
            if len(self.src_pts) >= 2:
                pts = np.array(self.src_pts, np.int32).reshape((-1, 1, 2))
                cv2.polylines(self.display, [pts], False, (0, 255, 0), 2)
            cv2.imshow("选择4个角点", self.display)

    def pick_points(self):
        self.display = self.img.copy()
        cv2.namedWindow("选择4个角点")
        cv2.setMouseCallback("选择4个角点", self.mouse_callback)

        print("\n请在图片上依次点击: 左上 → 右上 → 右下 → 左下")
        print("按 ESC 重新选择, 选完后按任意键继续...\n")

        cv2.imshow("选择4个角点", self.display)
        while len(self.src_pts) < 4:
            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                self.src_pts = []
                self.display = self.img.copy()
                cv2.imshow("选择4个角点", self.display)
                print("  已重置，重新选择")
        cv2.destroyWindow("选择4个角点")

    def get_params(self):
        print(f"\n图片原始尺寸: {self.w} x {self.h}")
        ow = int(input("鸟瞰图宽度 (推荐 640): ") or 640)
        oh = int(input("鸟瞰图高度 (推荐 480): ") or 480)
        rw = float(input("所选区域实际宽度 (米, 如 1.0): ") or 1.0)
        rh = float(input("所选区域实际长度 (米, 如 2.0): ") or 2.0)
        ppm = round((ow / rw + oh / rh) / 2, 2)
        return ow, oh, ppm

    def preview(self, M, ow, oh):
        bird = cv2.warpPerspective(self.img, M, (ow, oh))
        preview = np.hstack([
            cv2.resize(self.img, (ow, oh)),
            bird
        ])
        cv2.imshow("原图(缩放) | 鸟瞰图 (任意键确认, ESC重新标定)", preview)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        return key != 27

    def save_to_config(self, ow, oh, ppm):
        if not self.config_path:
            print("\n未指定配置文件，跳过写入")
            return

        # 读取现有 JSON（保留所有字段）
        if os.path.exists(self.config_path):
            with open(self.config_path, "r") as f:
                config = json.load(f)
        else:
            config = {}

        # 仅更新 calib 字段
        config["calib"] = {
            "src_quad": self.src_pts,
            "bird_width": ow,
            "bird_height": oh,
            "ppm": ppm
        }

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"\n标定参数已写入: {self.config_path}")

    def run(self):
        while True:
            self.pick_points()
            ow, oh, ppm = self.get_params()

            src = np.array(self.src_pts, dtype=np.float32)
            dst = np.array([
                [0, 0],
                [ow - 1, 0],
                [ow - 1, oh - 1],
                [0, oh - 1]
            ], dtype=np.float32)
            M = cv2.getPerspectiveTransform(src, dst)

            print(f"\n  PPM = {ppm} (px/m)")
            print("\n--- 鸟瞰预览 ---")
            if self.preview(M, ow, oh):
                break

        self.save_to_config(ow, oh, ppm)

        # 同时打印便于检查
        print("\n" + "=" * 60)
        print("标定结果")
        print("=" * 60)
        print(f"  源四点: {self.src_pts}")
        print(f"  鸟瞰尺寸: {ow} x {oh} px")
        print(f"  PPM: {ppm} px/m")
        print(f"  矩阵:\n{np.array2string(M, precision=3)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python calibrate_perspective.py <图片路径> [config.json]")
        print("示例: python calibrate_perspective.py track.jpg config/default.json")
        sys.exit(1)

    img_path = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else None
    cal = Calibrator(img_path, cfg_path)
    cal.run()
