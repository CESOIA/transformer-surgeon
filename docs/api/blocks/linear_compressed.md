# LinearCompressed

Drop-in `nn.Linear` replacement used as the compression target. All projection
layers in supported models are replaced with `LinearCompressed` at load time.

In full-rank mode it behaves identically to `nn.Linear`. After `init_lrd(rank)`
it operates in two-matrix factored mode. See [Concepts](../../concepts.md) for the full explanation.

::: transformersurgeon.blocks.linear_compressed.LinearCompressed
