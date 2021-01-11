import cv2


class CascadesDetector:
    def __init__(self, path='/home/nopepsi/PycharmProjects/Vision-System/faceDetect'
                            '/haarcascade_frontalface_default_cuda.xml'):
        self.detector = cv2.cuda.CascadeClassifier_create(path)

    def predict(self, frame):
        gpuFrame = cv2.cuda_GpuMat()
        gpuFrame.upload(frame)
        gpuGrayFrame = cv2.cuda.cvtColor(gpuFrame, cv2.COLOR_BGR2GRAY)
        gpuResult = self.detector.detectMultiScale(gpuGrayFrame)
        return gpuResult.download()
