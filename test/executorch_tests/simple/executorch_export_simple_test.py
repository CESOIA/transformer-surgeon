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

PARTITIONER = XnnpackPartitioner()
# PARTITIONER = VulkanPartitioner()

# --- Model definition --- #

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1a = torch.nn.Linear(INPUT_SIZE[0], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer1b = torch.nn.Linear(INPUT_SIZE[1], EMBEDDING_SIZE[0], dtype=data_type)
        self.layer2 = torch.nn.Linear(EMBEDDING_SIZE[0], OUTPUT_SIZE[0], dtype=data_type)

    def forward(self,
                inputA: torch.Tensor,
                inputB: torch.Tensor,
                ) -> torch.Tensor:
        
        output = self.layer1a(inputA) + self.layer1b(inputB)
        output = self.layer2(output)

        return output
    
model = SimpleModel()
model.eval()

# --- Example inputs --- #

example_BS = 16
example_inputs = (
    torch.randn(example_BS, INPUT_SIZE[0], dtype=data_type),  # input A
    torch.randn(example_BS, INPUT_SIZE[1], dtype=data_type),  # input B
)

# --- Export to executorch --- #

# Dynamic shapes: this will be fundamental to differentiate between prefetch and decode phases in LLM generation
batch = Dim("batch", max=256)
dynamic_shapes = {
    "inputA": (batch, INPUT_SIZE[0]),
    "inputB": (batch, INPUT_SIZE[1]),
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

BS = 4 # Change the batch size to validate dynamic shapes
example_inputs = (
    torch.randn(BS, INPUT_SIZE[0], dtype=data_type),  # x
    torch.randn(BS, INPUT_SIZE[1], dtype=data_type),  # y
)

runtime = Runtime.get()

program = runtime.load_program("simple_model.pte")
method = program.load_method("forward")
outputs = method.execute(list(example_inputs))

# --- Validate outputs --- #

example_outputs = model(*example_inputs)

maxerr = maxerr(example_outputs, outputs[0])
print(f"Max error between original and executorch outputs: {maxerr.item()}")