
import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import numpy as np
import random

def paint_until_filled(
    height: int,
    width: int,
    max_iter: int = 100000,
    min_radius: int = 5,
    max_radius: int = 30
) -> np.ndarray:
    """
    真っ白な画像を作り、ランダムに円を描き続ける。
    白いピクセルがなくなったら(全部塗られたら)描画を終了し、その画像を返す。

    Args:
        height (int): 画像の高さ
        width (int): 画像の幅
        max_iter (int): 最大試行回数（無限ループ防止用）
        min_radius (int): 円の最小半径
        max_radius (int): 円の最大半径

    Returns:
        np.ndarray: 最終的に全てが塗りつぶされて白い部分がなくなった画像 (BGR)
    """

    # (height, width, 3) の白画像(BGR=255,255,255)を用意
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)

    for i in range(max_iter):
        # ランダムに中心座標 (x, y) を決める
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # ランダムに円の色 (B, G, R) [0..255]
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )

        # ランダム半径 [min_radius..max_radius]
        radius = random.randint(min_radius, max_radius)

        # cv2.circle(描画先, 中心( x, y ), 半径, 色(BGR), thickness=-1 => 塗りつぶし)
        cv2.circle(canvas, (x, y), radius, color, thickness=-1)

        # 白いピクセルが一つも無いかどうか確認
        # 「各ピクセル == [255,255,255]」が成り立つかを軸2(=RGB)ですべて True か判定
        white_mask = np.all(canvas == [255, 255, 255], axis=2)
        if not np.any(white_mask):
            # まだ白い部分が無い→全ピクセルがカラーで塗り潰された
            print(f"Filled at iteration: {i+1}")
            break

    return canvas

def random_circles(img: np.ndarray, n_points=2000) -> np.ndarray:
    """
    画像の中からランダムに n_points (デフォルト: 100) 点をサンプリングし、  
    それぞれの座標に該当するピクセルの色で円を描いて返す。
    円の大きさ(半径)はランダムに決定する。
    
    Args:
        img (np.ndarray): BGR 画像 (shape: (H, W, 3))
        n_points (int): 円を描く点の数
    
    Returns:
        np.ndarray: 円を描画した後の BGR 画像
    """
    # 出力用に元画像をコピー
    out_img = img.copy()
    height, width, _ = out_img.shape

    for _ in range(n_points):
        # ランダムに座標を決める
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)

        # 現在のピクセル値(BGR)を取り出す
        b, g, r = out_img[y, x]

        # ランダムな半径 (適宜範囲は調整してください)
        radius = random.randint(5, 20)

        # 中心 (x, y) に円を塗りつぶし
        cv2.circle(out_img, center=(x, y), radius=radius, 
                   color=(int(b), int(g), int(r)), thickness=-1)

    return out_img

def some_filter(img: np.ndarray) -> np.ndarray:
    """
    例: グレースケールに変換してからエッジを検出し、3チャンネルに戻す
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return edges_3ch

def mosaic_by_resize(img: np.ndarray, factor: int = 200) -> np.ndarray:
    """
    縮小 → 再拡大でブロック状のモザイクを作る。
    factor: 縮小率の目安。たとえば factor=10 なら 1/10 に縮小してから元サイズに拡大する。

    Args:
        img (np.ndarray): BGR形式の画像
        factor (int): 縮小倍率 (大きいほど粗くなる)

    Returns:
        np.ndarray: モザイク(ブロック)状の画像 (BGR)
    """
    h, w = img.shape[:2]

    # 1) 縮小（補間方法はお好みで調整。INTER_LINEAR や INTER_AREA など）
    small = cv2.resize(img, (w // factor, h // factor),
                       interpolation=cv2.INTER_LINEAR)
    
    # 2) 再拡大（"nearest" で拡大するとブロックがはっきり残る）
    mosaic = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    return mosaic
    
def callback(frame):
    # WebRTC から受け取ったフレームを NumPy (BGR) に変換
    img = frame.to_ndarray(format="bgr24")

    # フィルタ処理関数を呼び出して、戻りを得る
    # processed_img = mosaic_by_resize(img)
    processed_img = random_circles(img)

    # 再度 av.VideoFrame に包んで返す
    return av.VideoFrame.from_ndarray(processed_img, format="bgr24")

if __name__ == "__main__":
    st.title("Simple Filter Example")

    webrtc_streamer(
        key="example",
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
    )