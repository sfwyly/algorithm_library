
import bisect
import numpy as np

import tensortrt as trt


# 预推理

# 创建Logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)

# 创建构建器builder
builder = trt.Builder(logger)

# 预训练网络
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

# 加载onnx解析器
parser = trt.OnnxParser(network, logger)

success = parser.parse_from_file(onnx_path)

for idx in range(parser.num_errors):
  print(parser.get_error(idx))
if not success:
  pass  # Error handling code here
# builder配置
config = builder.create_builder_config()

# 分配显存作为工作区间，一般建议为显存一半的大小
config.max_workspace_size = 1 << 30  # 1 Mi

serialized_engine = builder.build_serialized_network(network, config)

# 序列化生成engine文件
with open(engine_path, "wb") as f:
   f.write(serialized_engine)


# 部署

import tensorrt as trt
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
#创建logger：日志记录器

logger = trt.Logger(trt.Logger.WARNING)
#创建runtime并反序列化生成engine
with open("sample.engine", "rb") as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
#分配CPU锁页内存和GPU显存
h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)
d_input = cuda.mem_alloc(h_input.nbytes)
d_output = cuda.mem_alloc(h_output.nbytes)
#创建cuda流
stream = cuda.Stream()
#创建context并进行推理
with engine.create_execution_context() as context:
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return the host output. 该数据等同于原始模型的输出数据
    return h_output
