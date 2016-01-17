
# -*- coding: utf-8 -*-

import cv2
import math
import numpy as np
import os

if __name__ == '__main__':

    # 顔判定で使うxmlファイルを指定する。
    cascade_path =  os.path.dirname(os.path.abspath(__file__)) + '/data/haarcascades/haarcascade_cars.xml'
    print(cascade_path)
    cascade = cv2.CascadeClassifier(cascade_path)

    # 動画の読み込み
    cap = cv2.VideoCapture('./assets/cars.mp4')
    # 出力先
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = 'DIVX'

    out = cv2.VideoWriter('./output.mp4', cv2.VideoWriter_fourcc(*fourcc), fps, size)
    
    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:

            # 結果を保存するための変数を用意しておく。
            frame_result = frame

            # グレースケールに変換
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # ヒストグラムの平坦化
            frame_gray = cv2.equalizeHist(frame_gray);

            #顔判定
            """
            minSize で顔判定する際の最小の四角の大きさを指定できる。a
            (小さい値を指定し過ぎると顔っぽい小さなシミのような部分も判定されてしまう。)
            """
            faces = cascade.detectMultiScale(frame_gray, scaleFactor = 1.1, minNeighbors = 1, minSize = (100, 100))

            # 顔があった場合
            if len(faces) > 0:

                #顔認識の枠の色
                color = (255, 0, 0)

                # 複数の顔があった場合、１つずつ四角で囲っていく
                for face in faces:

                    # faceには(四角の左上のx座標, 四角の左上のy座標, 四角の横の長さ, 四角の縦の長さ) が格納されている。
                    # 囲う四角の左上の座標
                    coordinates = tuple(face[0:2])
                    # (囲う四角の横の長さ, 囲う四角の縦の長さ)
                    length = tuple(face[0:2] + face[2:4])

                    # 四角で囲う処理
                    cv2.rectangle(frame_result, coordinates, length, color, thickness = 3)

                # 表示
                cv2.imshow("Show FACES Image", frame_result)
        
            out.write(frame_result)

        # qを押したら終了。
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cv2.destroyAllWindows()
    cap.release()
    out.release()
