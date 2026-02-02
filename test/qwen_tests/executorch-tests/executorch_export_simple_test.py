import torch

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.backends.apple.coreml.partition import CoreMLPartitioner
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from executorch.runtime import Runtime

# --- Functions and constants --- #

maxerr = lambda x, y: torch.max(torch.abs(x - y))

data_type = torch.float32

INPUT_SIZE = [16, 16]
OUTPUT_SIZE = [16]

# PARTITIONER = XnnpackPartitioner()
PARTITIONER = CoreMLPartitioner()

# --- Model definition --- #

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        EMBEDDING_SIZE = 32
        self.layer1a = torch.nn.Linear(INPUT_SIZE[0], EMBEDDING_SIZE, dtype=data_type)
        self.layer1b = torch.nn.Linear(INPUT_SIZE[1], EMBEDDING_SIZE, dtype=data_type)
        self.layer2 = torch.nn.Linear(EMBEDDING_SIZE, OUTPUT_SIZE[0], dtype=data_type)

    def forward(self,
                x: torch.Tensor,
                y: torch.Tensor,
                ) -> torch.Tensor:
        
        output = self.layer1a(x) + self.layer1b(y)
        output = self.layer2(output)

        return output
    
model = SimpleModel()
model.eval()

# --- Example inputs --- #

example_inputs = (
    torch.randn(INPUT_SIZE[0], dtype=data_type),  # x
    torch.randn(INPUT_SIZE[1], dtype=data_type),  # y
)

# --- Export to executorch --- #

# Avoid dynamic shapes for now
# batch = Dim("batch", max=256)
# dynamic_shapes = {
#     "x": (batch, None),
#     "y": (batch, None),
# }

# convert to graph
print("Exporting model to graph...")
exported_program = export(model, example_inputs)
print(exported_program)

# convert graph to target
print("Lowering graph to executorch...")
executorch_program = to_edge_transform_and_lower(
    exported_program,
    partitioner = [PARTITIONER],
).to_executorch()

# save the executorch model
with open("model.pte", "wb") as file:
    file.write(executorch_program.buffer)

# --- Load back and run the executorch model --- #

# BS = 1
# example_inputs = (
#     torch.randn(BS, INPUT_SIZE[0], dtype=data_type),  # x
#     torch.randn(BS, INPUT_SIZE[1], dtype=data_type),  # y
# )

runtime = Runtime.get()

program = runtime.load_program("model.pte")
method = program.load_method("forward")
outputs = method.execute(list(example_inputs))

# --- Validate outputs --- #

example_outputs = model(*example_inputs)

maxerr = maxerr(example_outputs, outputs[0])
print(f"Max error between original and executorch outputs: {maxerr.item()}")