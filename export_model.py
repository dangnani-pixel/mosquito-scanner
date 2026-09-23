import torch,sys
from ultralytics import YOLO
torch.set_num_threads(4)
original_export=torch.onnx.export
def legacy_export(*args,**kwargs):
 kwargs['dynamo']=False
 return original_export(*args,**kwargs)
torch.onnx.export=legacy_export
m=YOLO(sys.argv[1] if len(sys.argv)>1 else 'yolito.pt')
print('CLASSES',m.names,flush=True)
print(m.export(format='onnx',imgsz=640,opset=17,simplify=False,dynamic=False,device='cpu'),flush=True)
