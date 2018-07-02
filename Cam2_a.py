import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

def realtime_graph(x, y):
    line, = plt.plot(x, y,label="S Curve") # (x,y)のプロット
    line.set_ydata(y)   # y値を更新
    plt.title("S Curve")  # グラフタイトル
    plt.xlabel("x")     # x軸ラベル
    plt.ylabel("y")     # y軸ラベル
    plt.legend()        # 凡例表示
    plt.grid()          # グリッド表示
    plt.xlim([0,255])    # x軸範囲
    plt.ylim([0,255])    # y軸範囲
    plt.draw()          # グラフの描画
    plt.pause(0.01)     # 更新時間間隔
    plt.clf()

def myfunc(i):
    pass # do nothing

def S_curve(x):
    y = (np.sin(np.pi * (x/255 - s/255)) + 1)/2 * 255
    return y

cv2.namedWindow('title') # create win with win name
#RGB trackbar
cv2.createTrackbar('R','title',0,100,myfunc)
cv2.createTrackbar('G','title',0,100,myfunc)
cv2.createTrackbar('B','title',0,100,myfunc)

cv2.createTrackbar('s_curve','title',127,255,myfunc)
cv2.createTrackbar('grayscale','title',0,1,myfunc)
cv2.createTrackbar('S Curve:on/off','title',0,1,myfunc)

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)



while(True):
    k = cv2.waitKey(1)
    if k == ord('q') or k == 27:
        break

    ret, frame = cap.read()
    if not ret: continue

    #frame resize
    frame = cv2.resize(frame,(int(frame.shape[1]/3), int(frame.shape[0]/3)))

    #grayscale
    gray = cv2.getTrackbarPos('grayscale','title')
    if gray == 1:
        frame = rgb2gray(frame)

    r = cv2.getTrackbarPos('R','title')
    g = cv2.getTrackbarPos('G','title')
    b = cv2.getTrackbarPos('B','title')

    #rbg
R = frame[:,:,2]
    G = frame[:,:,1]
    B = frame[:,:,0]
    R = R/255
    G = G/255
    B = B/255
    R = R*(r/100)
    G = G*(g/100)
    B = B*(b/100)
    frame[:,:,2] = R*255
    frame[:,:,1] = G*255
    frame[:,:,0] = B*255
    #S Curve
    s = cv2.getTrackbarPos('s_curve','title')
    x = np.linspace(0, 255, 100)
    y = S_curve(x)
    realtime_graph(x,y)

    #S Curve on/off
    sw = cv2.getTrackbarPos('S Curve:on/off','title')
    if sw == 1:
        frame = S_curve(frame)
    cv2.imshow('title', frame)  # show in the win



plt.close()
cap.release()
cv2.destroyAllWindows()
