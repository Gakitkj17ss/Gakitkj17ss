import cv2
import dlib
from ultralytics import YOLO

# YOLOv8モデルをロード
model = YOLO("yolov8s.pt")

# dlibの顔検出器と顔ランドマーク予測器をロード
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# 動画ファイルを指定
video_path = "kouno.mp4"
cap = cv2.VideoCapture(video_path)

# 出力動画の設定
output_path = "kounoda.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
out = None  # 初期状態では未設定

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # フレームごとにYOLOv8で物体検出
    results = model(frame)

    # "person"オブジェクトにのみバウンディングボックスを処理
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls.item())
            if model.names[cls] == "person":
                # バウンディングボックスの座標を取得
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # バウンディングボックスの領域をクロップ
                cropped_frame = frame[y1:y2, x1:x2]

                # 顔検出を行う
                gray = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
                faces = detector(gray)

                # 検出された顔に対してランドマークを描画
                for face in faces:
                    landmarks = predictor(gray, face)
                    for n in range(0, 68):
                        x = landmarks.part(n).x
                        y = landmarks.part(n).y
                        cv2.circle(cropped_frame, (x, y), 2, (0, 255, 0), -1)

                # 初回のみ出力動画の設定（クロップした領域のサイズに合わせる）
                if out is None:
                    out = cv2.VideoWriter(output_path, fourcc, 30.0, (x2 - x1, y2 - y1))

                # クロップしたフレームを動画として保存
                out.write(cropped_frame)

    # クロップした結果をリアルタイムで表示（省略可能）
    cv2.imshow("Cropped YOLOv8 Detection with Landmarks", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
