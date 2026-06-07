#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Flask共享内存视频流推流服务器（开发板版本）
功能：从共享内存读取图像数据，通过Flask Web服务器实现MJPEG视频流推送
适用场景：嵌入式开发板（如RK3588、树莓派等）与主控之间的视频传输
作者：嵌入式开发工程师
日期：2024
"""

import time
import struct
import numpy as np
import cv2
from multiprocessing import shared_memory, resource_tracker
from flask import Flask, Response

# ============================================================================
# 全局配置参数
# ============================================================================
SHM_NAME = "shm_ar_video"           # 共享内存名称
SHM_HEADER_SIZE = 16                # 共享内存头部大小（帧号8字节 + 宽度4字节 + 高度4字节）

# 创建Flask应用实例
app = Flask(__name__)


# ============================================================================
# 共享内存管理函数
# ============================================================================
def remove_shm_from_resource_tracker():
    """
    功能：从资源追踪器中移除共享内存，防止客户端退出时误删服务器的共享内存
    参数：无
    返回值：无
    注意事项：这是关键修复，避免多进程共享内存冲突
    """
    try:
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except:
        pass


def read_frame_from_shm(shm, last_fid):
    """
    功能：从共享内存读取一帧图像
    参数：
        shm: 共享内存对象
        last_fid: 上一帧的帧号，用于判断是否有新帧
    返回值：
        frame: 图像数据（BGR格式），如果没有新帧则返回None
        fid: 当前帧号
    注意事项：使用深拷贝避免长时间占用共享内存
    """
    try:
        # 读取头部信息（帧号 + 宽度 + 高度）
        header = bytes(shm.buf[:SHM_HEADER_SIZE])
        fid, w, h = struct.unpack('QII', header)
        
        # 帧号没变，说明没有新帧
        if fid == last_fid:
            return None, last_fid
        
        # 计算图像数据大小
        size = w * h * 3
        
        # 从共享内存读取图像数据（深拷贝，避免长时间占用）
        img_view = np.ndarray((h, w, 3), dtype=np.uint8, 
                             buffer=shm.buf[SHM_HEADER_SIZE : SHM_HEADER_SIZE + size])
        frame = img_view.copy()
        del img_view  # 立即释放视图
        
        return frame, fid
        
    except (ValueError, struct.error, BufferError):
        # 读取失败，返回None
        return None, last_fid


def process_frame(frame):
    """
    功能：处理图像帧（翻转、颜色转换）
    参数：
        frame: 原始图像数据
    返回值：
        processed_frame: 处理后的图像数据
    注意事项：根据实际需求修改处理逻辑
    """
    # 图像翻转（根据摄像头安装方向调整）
    frame = cv2.flip(frame, 0)
    
    # 颜色空间转换：RGB -> BGR（OpenCV默认格式）
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    return frame


def generate_frames():
    """
    功能：生成MJPEG视频流（从共享内存读取）
    参数：无
    返回值：生成器，持续返回MJPEG格式的视频帧
    注意事项：这是一个生成器函数，会持续运行直到客户端断开连接
    """
    print("🎥 视频流客户端启动，等待连接共享内存...")
    
    shm = None
    last_fid = 0
    
    # FPS统计变量
    fps_t = time.time()
    fps_n = 0
    cur_fps = 0.0
    
    try:
        # ====================================================================
        # 连接共享内存
        # ====================================================================
        while True:
            try:
                shm = shared_memory.SharedMemory(name=SHM_NAME)
                # 连接成功后，立即从清理名单中移除
                remove_shm_from_resource_tracker()
                print("✅ 已连接到共享内存！")
                break
            except FileNotFoundError:
                print("⏳ 等待共享内存服务器启动...")
                time.sleep(1.0)
        
        # ====================================================================
        # 持续读取并推送视频流
        # ====================================================================
        while True:
            # 从共享内存读取帧
            frame, fid = read_frame_from_shm(shm, last_fid)
            
            if frame is None:
                # 没有新帧，短暂休眠
                time.sleep(0.002)
                continue
            
            last_fid = fid
            
            # 处理图像（翻转、颜色转换）
            processed_frame = process_frame(frame)
            
            # FPS统计
            fps_n += 1
            if time.time() - fps_t >= 1.0:
                cur_fps = fps_n / (time.time() - fps_t)
                fps_n = 0
                fps_t = time.time()
            
            # 在画面上显示FPS
            cv2.putText(processed_frame, f"FPS: {cur_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 编码为JPEG格式
            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            
            # 拼接成MJPEG流格式
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    except GeneratorExit:
        # 客户端断开连接
        print("📱 客户端断开连接")
    finally:
        # 清理资源
        if shm:
            try:
                shm.close()
            except:
                pass
        print("🔒 共享内存连接已关闭")


# ============================================================================
# Flask路由定义
# ============================================================================
@app.route('/video_feed')
def video_feed():
    """
    功能：返回视频流响应
    参数：无
    返回值：Flask Response对象，包含MJPEG视频流
    注意事项：浏览器访问此路由时会持续接收视频流
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    """
    功能：返回简单的HTML页面，用来展示视频流
    参数：无
    返回值：HTML字符串
    注意事项：页面使用全屏黑色背景，视频自适应显示
    """
    return """
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AR视频流 - 开发板</title>
        <style>
          body {
            margin: 0;
            padding: 0;
            background: #000;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            font-family: Arial, sans-serif;
          }
          .container {
            text-align: center;
            color: #fff;
          }
          img {
            width: 100vw;
            height: auto;
            object-fit: contain;
            border: 2px solid #333;
          }
          .info {
            margin-top: 10px;
            font-size: 14px;
            color: #888;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <img src="/video_feed" alt="AR视频流">
          <div class="info">
            📡 共享内存视频流 | 按 ESC 退出
          </div>
        </div>
      </body>
    </html>
    """


@app.route('/ar/feed')
def ar_feed():
    """
    功能：返回AR视频流（自定义路径，与用户截图中的路径一致）
    参数：无
    返回值：Flask Response对象，包含MJPEG视频流
    注意事项：这个路径与用户提供的截图路径一致
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# ============================================================================
# 主程序入口
# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("🚀 Flask共享内存视频流推流服务器（开发板版本）")
    print("=" * 70)
    print("📡 共享内存名称:", SHM_NAME)
    print("🌐 本地访问地址: http://127.0.0.1:5000")
    print("🌐 局域网访问地址: http://<开发板IP>:5000")
    print("🎯 AR视频流路径: http://<开发板IP>:5000/ar/feed")
    print("=" * 70)
    print("按 Ctrl+C 停止服务器")
    print("=" * 70)
    
    # 启动Flask服务器
    # host='0.0.0.0': 监听所有网络接口，允许局域网访问
    # port=5000: 监听端口为5000
    # debug=False: 生产环境关闭调试模式
    # threaded=True: 启用多线程，支持多个客户端同时访问
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
