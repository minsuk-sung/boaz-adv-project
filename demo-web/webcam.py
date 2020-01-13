import cv2

cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('save.avi', fourcc, 25.0, (640, 480))

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

while(True):
    ret, frame = cap.read()    # Read 결과와 frame
    if(ret) :
        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        out.write(frame)
        if cv2.waitKey(1) == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
