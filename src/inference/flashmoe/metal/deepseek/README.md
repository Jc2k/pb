# DeepSeek V4 Flash Metal sources

These files are vendored from [`antirez/ds4`](https://github.com/antirez/ds4) at commit
`80ebbc396aee40eedc1d829222f3362d10fa4c6c` and retain its MIT license. They are compiled as a
family-specific library by the existing FlashMoe Metal execution facade. The host graph, scheduler,
expert streaming, tokenizer, cache, and session state remain pb-owned.

The source set is intentionally pinned. Updating it requires reviewing kernel signatures, updating
the load-time capability manifest, and rerunning DeepSeek plus existing Qwen/GLM parity checks.
