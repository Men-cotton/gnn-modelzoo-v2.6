# GNN Findings

## `cstorch.nn.functional.sparse_matmul` for GCN-PubMed

`cstorch.nn.functional.sparse_matmul` was tested as a possible replacement for
the full-graph GCN propagation `A @ H` on PubMed. The attempt did not work well
for this model shape.

The existing in-repo use case is `TopKExpertsSparseLinear` in
`src/cerebras/modelzoo/layers/SparseMoEBlock.py`. That path uses the operator in
its natural layout: each token has a small `top_k` expert dimension, and the
operator selects from weights shaped like `(hidden_out, num_experts, hidden_in)`.
The compressed sparse dimension is the expert-selection axis, not the node axis
of a graph.

For GCN-PubMed, encoding `A @ H` requires treating graph node ids as the sparse
dimension. The PubMed graph has 19,717 nodes and the normalized adjacency expands
to fixed row slots with fanout 172. On CPU, `sparse_matmul` uses its fallback
implementation: scatter the sparse values into a dense tensor, run
`torch.einsum("...MBN, KBN -> ...MBK", ...)`, then gather the selected slots.
This materializes a dense intermediate over the full node dimension, so the
fallback loses the intended sparsity benefit.

The concrete command:

```bash
uv run cszoo fit \
  src/cerebras/modelzoo/models/gnn/configs/params_gcn_sparse_matmul_pubmed.yaml \
  --target_device CPU \
  -o /tmp/gnn_gcn_sparse_matmul_pubmed_cpu
```

loaded PubMed successfully, then failed on the first training step inside the
CPU fallback `einsum`:

```text
DefaultCPUAllocator: can't allocate memory: you tried to allocate 49761291392 bytes.
```

This is not just slow; the CPU path is structurally impractical for full-graph
GCN-PubMed. It also indicates a mismatch between the operator's intended sparse
expert selection use case and graph adjacency propagation. Keeping a dedicated
GCN sparse-matmul implementation would add an alternate data format and model
registration without a runnable local verification path, so the implementation
was removed.
