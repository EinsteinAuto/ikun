# Online Serving Failures and Fallbacks

Coding guidelines for exception handling in online execution paths
(graph capture, kernel dispatch, decode forward, prefill).

## Principles

- **Prefer the let-it-crash principle for unexpected failures in online
  execution paths.**  Model execution, graph capture/replay, device
  operations, and runtime state updates should let unexpected exceptions
  propagate instead of catching them and continuing the request.

- **Do not introduce `try/catch` or `try/except` casually in online
  execution paths.**  In particular, do not swallow an exception, retry
  the operation, fall back to another execution path, or keep serving
  from state that may already be partially modified.

- **Do not introduce fallback code casually.**  A fallback is
  appropriate only when it is an explicitly supported path selected
  *before* execution through observable conditions, and both paths have
  defined and tested semantics.  It must not be used to mask a
  correctness issue, unsupported state, or failed operation.

- Express expected unsupported conditions through explicit validation,
  return values, or early exits *before* mutating execution state.
  Do not use exceptions as normal capability detection.

- Exception handling remains valid at required process, RPC, C ABI,
  untrusted-input, or third-party API boundaries.  Such handlers must
  catch the narrowest practical exception, preserve the failure signal,
  and must not pretend that a partially failed online operation
  succeeded.

## Examples

```python
# Good: choose a supported path before execution starts.
if not graph_runner.can_execute(input_ids):
    raise RuntimeError("batch exceeds graph capacity")
return graph_runner.replay(input_ids)

# Bad: graph execution may have modified runtime state before failing.
try:
    return graph_runner.replay(input_ids)
except Exception:
    return eager_forward(model, input_ids)
```

```cpp
// Good: validate, then execute.
const bool ok = graph->capture(model, params);
CHECK(ok) << "graph capture failed for bucket " << bucket;

// Bad: catch-and-fallback hides partially mutated graph state.
try {
    graph->capture(model, params);
} catch (...) {
    return forward_eager(model, params);
}
```

## Acceptable exception handling

- `try/finally` for resource cleanup (file handles, temp directories)
  that does not alter the control flow of the online request.
- `try/except` at process-level entry points (worker main loops, RPC
  handlers) where the exception is logged and the *process* is
  terminated or the *request* is failed, never silently retried.
- `try/except` around import statements for optional dependencies.
