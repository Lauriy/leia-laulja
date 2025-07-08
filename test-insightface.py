from insightface.app import FaceAnalysis

model = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider',])
model.prepare(ctx_id=0, det_size=(640, 640))