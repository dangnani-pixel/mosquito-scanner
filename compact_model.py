import onnx,numpy as np,json,os,hashlib,sys,tempfile
from onnx import numpy_helper,helper,TensorProto
m=onnx.load(sys.argv[1] if len(sys.argv)>1 else 'yolito.onnx');nodes=[]
for tensor in m.graph.initializer:
 if tensor.data_type==TensorProto.FLOAT:
  name=tensor.name;arr=numpy_helper.to_array(tensor)
  if arr.size<8:continue
  tensor.CopyFrom(numpy_helper.from_array(arr.astype(np.float16),name=name+'_fp16'))
  nodes.append(helper.make_node('Cast',[name+'_fp16'],[name],to=TensorProto.FLOAT,name=name+'_restore'))
old=list(m.graph.node);del m.graph.node[:];m.graph.node.extend(nodes+old);m.ir_version=10
onnx.checker.check_model(m)
# Serialize in memory (portable: no hard-coded /tmp path, works on Windows too)
b=m.SerializeToString();root=(sys.argv[2] if len(sys.argv)>2 else 'dist/model');os.makedirs(root,exist_ok=True);parts=[]
for old_part in os.listdir(root):
 if old_part.startswith('yolito-') and old_part.endswith('.bin'):os.remove(os.path.join(root,old_part))
for i,start in enumerate(range(0,len(b),8*1024*1024)):
 name=f'yolito-{i:02}.bin';parts.append(name);open(os.path.join(root,name),'wb').write(b[start:start+8*1024*1024])
json.dump({'size':len(b),'parts':parts,'sha256':hashlib.sha256(b).hexdigest(),'input':[1,3,640,640],'output':[1,5,8400],'source':'https://github.com/WildMosquit0/YOLito','class':'mosquito','license':'AGPL-3.0'},open(os.path.join(root,'manifest.json'),'w'))
print('Model bytes',len(b))
