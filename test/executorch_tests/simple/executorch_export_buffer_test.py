import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
# from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from executorch.runtime import Runtime

# --- Functions and constants --- #

maxerr = lambda x, y: torch.max(torch.abs(x - y))

data_type = torch.float32

INPUT_SIZE = (16, 16) # input feature A size, input feature B size
EMBEDDING_SIZE = (32,) # hidden feature size
OUTPUT_SIZE = (16,) # output feature size
BUFFER_LENGTH = 16
BATCH_SIZE = 1

PARTITIONER = XnnpackPartitioner()
# PARTITIONER = VulkanPartitioner()

# --- Model definition --- #

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1a = torch.nn.Linear(INPUT_SIZE[0], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer1b = torch.nn.Linear(INPUT_SIZE[1], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer2 = torch.nn.Linear(EMBEDDING_SIZE[0], OUTPUT_SIZE[0], dtype=data_type)
        self.register_buffer("cache", torch.zeros(BATCH_SIZE, BUFFER_LENGTH, EMBEDDING_SIZE[0], dtype=data_type), persistent=False)

    def forward(self,
                inputA: torch.Tensor,
                inputB: torch.Tensor,
                cache_len_tensor: torch.Tensor,
                ) -> torch.Tensor:
        
        cache_len = cache_len_tensor.size(0)
        output = self.layer1a(inputA) + self.layer1b(inputB)
        with torch.no_grad():
            self.cache[:, cache_len:cache_len+1] = output.unsqueeze(1)
        output = self.cache[:, :cache_len+1]
        output = self.layer2(output)

        return output, self.cache
    
model = SimpleModel()
model.eval()

# --- Example inputs --- #

example_BS = 1
example_seq_len = 6
max_cache_length = 16
example_inputs = (
    torch.randn(example_BS, INPUT_SIZE[0], dtype=data_type),  # input A
    torch.randn(example_BS, INPUT_SIZE[1], dtype=data_type),  # input B
    torch.ones(example_seq_len, dtype=torch.long),  # cache length tensor
)

# --- Export to executorch --- #

# Dynamic shapes: this will be fundamental to differentiate between prefetch and decode phases in LLM generation
seq_len_dim = Dim("seq_len", max=max_cache_length-1)
dynamic_shapes = {
    "inputA": (Dim.STATIC, Dim.STATIC),
    "inputB": (Dim.STATIC, Dim.STATIC),
    "cache_len_tensor": (seq_len_dim,),
}

# convert to graph
print("Exporting model to graph...")
exported_program = export(model, example_inputs, dynamic_shapes=dynamic_shapes)
print(exported_program)

# convert graph to target
print("Lowering graph to executorch...")
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [PARTITIONER],
).to_executorch()

# save the executorch model
with open("simple_model.pte", "wb") as file:
    file.write(executorch_program.buffer)

# --- Load back and run the executorch model --- #

runtime = Runtime.get()
program = runtime.load_program("simple_model.pte")
method = program.load_method("forward")

for i in range(10):
    BS = 1
    SL = i 
    example_inputs = (
        torch.randn(BS, INPUT_SIZE[0], dtype=data_type),  # x
        torch.randn(BS, INPUT_SIZE[1], dtype=data_type),  # y
        torch.ones(SL, dtype=torch.long),  # cache length tensor
    )

    outputs = method.execute(list(example_inputs))

    print(outputs[1][0, :, 0])
