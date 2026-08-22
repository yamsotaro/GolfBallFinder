# Model resource

Place the exported Core ML package here as:

`GolfBall.mlpackage`

Then regenerate/open the Xcode project so the resource is bundled. The app intentionally shows a clear runtime error when the model is absent.

`training/export_coreml.py` also writes `ModelManifest.json` with the checkpoint hash,
tool versions, input size, and export precision. Keep that small manifest with benchmark results;
the generated Core ML package itself remains ignored by default. If a model package is deliberately committed,
the signed Codemagic workflow requires its manifest too.
