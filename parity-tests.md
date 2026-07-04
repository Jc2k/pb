# FlashMoe Parity Fixtures

The parity tests in `src/inference/flashmoe/mod.rs` use tiny synthetic manifests,
tokenizers, expert packs, and images. They intentionally avoid importing
Qwen3.5-397B or baking in assumptions that only hold for that checkpoint.
They are functional fixtures only: no timing harnesses, throughput assertions, or
benchmark infrastructure belong here.

Refresh the fixture goldens when upstream Qwen tokenizer, chat-template, config,
or Qwen3-VL image-processor behavior changes:

1. Regenerate the tokenizer/chat goldens with the upstream Hugging Face
   tokenizer for the same tiny messages used in the tests. Update only the
   expected rendered strings and token IDs that changed.
2. Recompute visual-token goldens with the upstream Qwen3-VL image processor on
   a small deterministic image whose dimensions are multiples of the patch and
   merge sizes. Keep the fixture image generated in the test, not checked in.
3. Recompute M-RoPE positions from the expanded tiny prompt, preserving the
   `temporal/height/width` triples as explicit golden intermediate values.
4. For routing and expert fixtures, keep synthetic scores and Q4 matrices small
   enough to verify by hand. Update expected top-K indices, softmax weights,
   parsed Q4 records, and MLP intermediates together.
5. Run `cargo test flashmoe_parity --lib` and `cargo test qwen3vl_parity --lib`
   after updating the goldens, then run the broader FlashMoe unit tests before
   committing.

Optional real-model fixtures must stay ignored or feature/env gated and should
document the exact local cache they require. The default CI/test path should
continue to run with only the synthetic tokenizer/config/image fixtures above.
