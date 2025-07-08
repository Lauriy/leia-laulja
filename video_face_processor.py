import cv2
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("face_detection.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VideoFaceProcessor:
    def __init__(
        self,
        video_path: str,
        output_path: str = "face_detection_results.json",
        thumbnails_dir: str = "face_thumbnails",
        thumbnail_size: int = 64,
    ):
        """
        Initialize the video face processor.

        Args:
            video_path: Path to the input video file
            output_path: Path for the output JSON file
            thumbnails_dir: Directory to save face thumbnails
            thumbnail_size: Size of thumbnail images (square)
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnail_size = thumbnail_size
        self.face_app = None
        self.results = []

        # Create thumbnails directory
        self.thumbnails_dir.mkdir(exist_ok=True)

    def initialize_face_detector(self) -> None:
        """Initialize InsightFace with GPU acceleration."""
        try:
            # Check available providers
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")

            # Try to use GPU first, fall back to CPU
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                logger.info("Attempting to use GPU (CUDA)")
            else:
                providers = ["CPUExecutionProvider"]
                logger.warning("GPU not available, using CPU")

            self.face_app = FaceAnalysis(providers=providers)
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))

            # Check which provider is actually being used
            if hasattr(self.face_app.det_model, "providers"):
                actual_providers = self.face_app.det_model.providers
                logger.info(f"Face detector using providers: {actual_providers}")

                if "CUDAExecutionProvider" in actual_providers:
                    logger.info("✅ Successfully initialized with GPU acceleration")
                else:
                    logger.warning("⚠️ Running on CPU - GPU acceleration not available")

        except Exception as e:
            logger.error(f"Failed to initialize face detector: {e}")
            raise

    def format_timestamp(self, frame_number: int, fps: float) -> str:
        """Convert frame number to timestamp string."""
        seconds = frame_number / fps
        return str(timedelta(seconds=int(seconds)))

    def extract_face_thumbnail(
        self, frame: np.ndarray, bbox: np.ndarray, frame_number: int, face_id: int
    ) -> Optional[str]:
        """
        Extract and save face thumbnail from frame.

        Args:
            frame: Original video frame
            bbox: Face bounding box coordinates
            frame_number: Current frame number
            face_id: Face ID within the frame

        Returns:
            Path to saved thumbnail file
        """
        try:
            # Extract face region with some padding
            x1, y1, x2, y2 = bbox.astype(int)
            h, w = frame.shape[:2]

            # Add padding (10% of face size)
            face_w, face_h = x2 - x1, y2 - y1
            padding_w, padding_h = int(face_w * 0.1), int(face_h * 0.1)

            # Apply padding with bounds checking
            x1 = max(0, x1 - padding_w)
            y1 = max(0, y1 - padding_h)
            x2 = min(w, x2 + padding_w)
            y2 = min(h, y2 + padding_h)

            # Extract face region
            face_region = frame[y1:y2, x1:x2]

            # Resize to thumbnail size
            thumbnail = cv2.resize(
                face_region, (self.thumbnail_size, self.thumbnail_size)
            )

            # Save thumbnail
            thumbnail_filename = f"face_{frame_number:08d}_{face_id:02d}.jpg"
            thumbnail_path = self.thumbnails_dir / thumbnail_filename
            cv2.imwrite(str(thumbnail_path), thumbnail)

            return str(thumbnail_path)

        except Exception as e:
            logger.error(
                f"Error extracting thumbnail for frame {frame_number}, face {face_id}: {e}"
            )
            return None

    def detect_faces_in_frame(
        self, frame: np.ndarray, frame_number: int, fps: float
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame and extract thumbnails.

        Args:
            frame: The video frame
            frame_number: Current frame number
            fps: Frames per second of the video

        Returns:
            List of face detection results
        """
        try:
            faces = self.face_app.get(frame)
            frame_results = []

            timestamp = self.format_timestamp(frame_number, fps)

            for i, face in enumerate(faces):
                # Extract bounding box coordinates
                bbox = face.bbox.astype(int)

                # Extract thumbnail
                thumbnail_path = self.extract_face_thumbnail(
                    frame, face.bbox, frame_number, i
                )

                face_data = {
                    "face_id": i,
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "bbox": {
                        "x1": int(bbox[0]),
                        "y1": int(bbox[1]),
                        "x2": int(bbox[2]),
                        "y2": int(bbox[3]),
                    },
                    "confidence": float(face.det_score),
                    "landmarks": face.kps.tolist() if hasattr(face, "kps") else None,
                    "age": int(face.age) if hasattr(face, "age") else None,
                    "gender": face.sex if hasattr(face, "sex") else None,
                    "thumbnail_path": thumbnail_path,
                }

                frame_results.append(face_data)

            return frame_results

        except Exception as e:
            logger.error(f"Error detecting faces in frame {frame_number}: {e}")
            return []

    def process_video_chunk(
        self, start_time: float, end_time: float, frame_skip: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of the video between start_time and end_time.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            frame_skip: Number of frames to skip between processing

        Returns:
            List of face detection results for this chunk
        """
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Set starting position
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        chunk_results = []
        frame_count = 0
        processed_frames = 0

        logger.info(
            f"Processing chunk: {timedelta(seconds=int(start_time))} to {timedelta(seconds=int(end_time))}"
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time:
                break

            # Skip frames for efficiency
            if frame_count % frame_skip == 0:
                current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                face_results = self.detect_faces_in_frame(
                    frame, current_frame_number, fps
                )
                chunk_results.extend(face_results)
                processed_frames += 1

                if processed_frames % 10 == 0:
                    logger.info(
                        f"Processed {processed_frames} frames, found {len(chunk_results)} faces so far"
                    )

            frame_count += 1

        cap.release()
        logger.info(f"Chunk complete: {len(chunk_results)} faces detected")
        return chunk_results

    def process_video(self, chunk_duration: int = 600, frame_skip: int = 30) -> None:
        """
        Process the entire video in chunks.

        Args:
            chunk_duration: Duration of each chunk in seconds (default: 10 minutes)
            frame_skip: Number of frames to skip between processing
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        # Initialize face detector
        self.initialize_face_detector()

        # Get video duration
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()

        logger.info(f"Video duration: {timedelta(seconds=int(duration))}")
        logger.info(
            f"Processing with frame skip: {frame_skip} (every {frame_skip / fps:.1f} seconds)"
        )
        logger.info(f"Thumbnails will be saved to: {self.thumbnails_dir}")

        # Process video in chunks
        current_time = 0
        chunk_number = 1

        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)

            try:
                chunk_results = self.process_video_chunk(
                    current_time, end_time, frame_skip
                )
                self.results.extend(chunk_results)

                # Save intermediate results
                self.save_results(f"intermediate_results_chunk_{chunk_number}.json")

                logger.info(
                    f"Chunk {chunk_number} complete. Total faces detected: {len(self.results)}"
                )

            except Exception as e:
                logger.error(f"Error processing chunk {chunk_number}: {e}")

            current_time = end_time
            chunk_number += 1

        # Save final results
        self.save_results()
        logger.info(
            f"Video processing complete! Total faces detected: {len(self.results)}"
        )
        logger.info(f"Thumbnails saved in: {self.thumbnails_dir}")

    def save_results(self, filename: Optional[str] = None) -> None:
        """Save the detection results to JSON file."""
        output_file = Path(filename) if filename else self.output_path

        metadata = {
            "video_path": str(self.video_path),
            "processing_date": datetime.now().isoformat(),
            "total_faces_detected": len(self.results),
            "total_frames_processed": len(
                set(result["frame_number"] for result in self.results)
            ),
            "thumbnails_directory": str(self.thumbnails_dir),
            "thumbnail_size": self.thumbnail_size,
        }

        output_data = {"metadata": metadata, "faces": self.results}

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        logger.info(f"Results saved to {output_file}")


def main():
    """Main function to run the video face processor."""
    video_path = "iseoma.mp4"
    output_path = "estonian_song_festival_faces.json"
    thumbnails_dir = "face_thumbnails"

    processor = VideoFaceProcessor(video_path, output_path, thumbnails_dir)

    # Process video with 10-minute chunks, processing every 30th frame
    processor.process_video(chunk_duration=600, frame_skip=30)


if __name__ == "__main__":
    main()
