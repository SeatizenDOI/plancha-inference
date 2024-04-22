import numpy as np
import tensorrt as trt

from .common_cuda import allocate_buffers, do_inference_v2

logger = trt.Logger(trt.Logger.WARNING)

def build_and_save_engine_from_onnx(path_onnx, path_engine):
    # 1 - Build Phase

    # 1.1 - Logs all message in stdout and create a builder.
    builder = trt.Builder(logger)

    # 1.1.1 - First step in optimizing model is to create network definition.
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    # 1.1.2 - Import a model using ONNX parser.
    parser = trt.OnnxParser(network, logger)

    succes = parser.parse_from_file(path_onnx)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not succes:
        print("1. Failed to imported ONNX model")
        quit()

    # 1.1.3 - Build engine.
    config = builder.create_builder_config()

    # This parameter limits the maximum size that any layer in the network can use.
    # Commented cause doesn't handle batch size feature 
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30) # 1 GiB

    # Serialize engine.
    serialized_engine = builder.build_serialized_network(network, config)

    # Save engine for another use.
    with open(path_engine, 'wb') as f:
        f.write(serialized_engine)


def load_engine(path):    
    # Deserialize the engine using the runtime interface.
    runtime = trt.Runtime(logger)

    # Load engine from file.
    with open(path, 'rb') as f:
        serialized_engine = f.read()

    # Deserialize from a memory buffer.
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # Allocate buffers and create cuda stream.
    inputs, outputs, bindings, stream = allocate_buffers(engine)

    # Context.
    context = engine.create_execution_context()

    return engine, context

class NeuralNetworkGPU:
    def __init__(self, model_path):
        self.engine, self.context = load_engine(model_path)
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(self.engine)

    def detect(self, images): 
        np.copyto(self.inputs[0].host, images.flatten())
        return do_inference_v2(self.context, self.bindings, self.inputs, self.outputs, self.stream)