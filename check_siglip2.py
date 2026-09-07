from embedders.siglip2_embedder import SigLIP2Embedder

emb = SigLIP2Embedder(
    model_id="google/siglip2-base-patch16-naflex",
    max_num_patches=576,
    batch_size=1,
)

print("model_id       :", emb.model_id)
print("OLD detection  :", "naflex" in emb.model_id.lower())
print("NEW detection  :", emb.is_naflex)
print("max_num_patches:", emb.max_num_patches)
print("processor      :", type(emb.processor.image_processor).__name__)
print("input names    :", getattr(emb.processor.image_processor, "model_input_names", None))
