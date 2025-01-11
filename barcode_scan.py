from roboflow import Roboflow

rf = Roboflow(api_key="")
project = rf.workspace().project("barcodes-zmxjq")
model = project.version(4).model

print(model.predict("barcode.jpg", confidence=40, overlap=30).json())

model.predict("barcode.jpg", confidence=40, overlap=30).save("prediction.jpg")