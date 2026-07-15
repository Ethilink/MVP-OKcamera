"""RF-DETR ONNX detector component.

Wraps the pre-trained RF-DETR instance-segmentation ONNX export: loads the
model, runs a single image through `preprocess` -> `session.run` ->
`decode_predictions`. See `../../../../docs/detector.md` for the verified
runtime contract.
"""

from pathlib import Path

import numpy as np
import onnxruntime
import supervision as sv

from orc_model.components.detector._rfdetr_postprocess import decode_predictions, preprocess


class Detector:
    def __init__(
        self,
        weights_path: Path | str,
        confidence_threshold: float = 0.5,
        top_k: int = 300,
        providers: list[str] | None = None,
        provider_options: list[dict] | None = None,
    ) -> None:
        """Loads the RF-DETR ONNX model via onnxruntime.InferenceSession."""
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Detector weights not found: {weights_path}")

        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.session = onnxruntime.InferenceSession(
            str(weights_path), providers=providers, provider_options=provider_options
        )
        self._input_name = self.session.get_inputs()[0].name
        self._output_names = ["dets", "labels", "masks"]
        available_outputs = {output.name for output in self.session.get_outputs()}
        missing_outputs = set(self._output_names) - available_outputs
        if missing_outputs:
            missing = ", ".join(sorted(missing_outputs))
            raise ValueError(f"RF-DETR ONNX output contract missing: {missing}")

    def predict(
        self, image: np.ndarray, confidence_threshold: float | None = None
    ) -> sv.Detections:
        """image: BGR np.ndarray (H,W,3) e.g. from cv2.imread or Frame.load_image().
        Returns sv.Detections in the image's own pixel coordinate space.

        ``confidence_threshold`` defaults to ``self.confidence_threshold``. Pass it
        explicitly to pin the exact value used for this call: the dashboard's
        capture loop records the threshold as annotation provenance, and reading
        the mutable attribute once here would leave a race where a concurrent
        ``/confidence`` change makes the recorded value disagree with the value
        that actually filtered the detections.
        """
        height, width = image.shape[:2]
        threshold = (
            self.confidence_threshold if confidence_threshold is None else confidence_threshold
        )

        preprocessed = preprocess(image)

        dets, labels, masks = self.session.run(
            self._output_names,
            {self._input_name: preprocessed},
        )

        return decode_predictions(
            dets,
            labels,
            masks,
            image_width=width,
            image_height=height,
            confidence_threshold=threshold,
            top_k=self.top_k,
        )
