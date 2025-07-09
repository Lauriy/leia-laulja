import cv2
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
import threading
import concurrent.futures

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
        batch_size: int = 32,  # Significantly increased for better GPU utilization
        max_parallel_chunks: int = 3,  # Process multiple chunks in parallel
    ):
        """
        Initialize the video face processor.

        Args:
            video_path: Path to the input video file
            output_path: Path for the output JSON file
            thumbnails_dir: Directory to save face thumbnails
            thumbnail_size: Size of thumbnail images (square)
            batch_size: Number of frames to process in each batch for better GPU utilization
            max_parallel_chunks: Number of chunks to process in parallel
        """
        self.video_path = Path(video_path)
        self.output_path = Path(output_path)
        self.thumbnails_dir = Path(thumbnails_dir)
        self.thumbnail_size = thumbnail_size
        self.batch_size = batch_size
        self.max_parallel_chunks = max_parallel_chunks
        self.face_app = None
        self.results = []
        self.results_lock = threading.Lock()

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
                # Configure CUDA provider for better memory usage
                cuda_provider_options = {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                    "gpu_mem_limit": 5 * 1024 * 1024 * 1024,  # 5GB limit
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                }
                providers = [
                    ("CUDAExecutionProvider", cuda_provider_options),
                    "CPUExecutionProvider",
                ]
                logger.info("Attempting to use GPU (CUDA) with optimized settings")
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
        self,
        frame: np.ndarray,
        bbox: np.ndarray,
        frame_number: int,
        face_id: int,
        chunk_number: int,
    ) -> Optional[str]:
        """
        Extract and save face thumbnail from frame.

        Args:
            frame: Original video frame
            bbox: Face bounding box coordinates
            frame_number: Current frame number
            face_id: Face ID within the frame
            chunk_number: Current chunk number for directory organization

        Returns:
            Path to saved thumbnail file
        """
        try:
            # Create chunk subdirectory
            chunk_dir = self.thumbnails_dir / f"chunk_{chunk_number:03d}"
            chunk_dir.mkdir(exist_ok=True)

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

            # Save thumbnail in chunk subdirectory
            thumbnail_filename = f"face_{frame_number:08d}_{face_id:02d}.jpg"
            thumbnail_path = chunk_dir / thumbnail_filename
            cv2.imwrite(str(thumbnail_path), thumbnail)

            return str(thumbnail_path)

        except Exception as e:
            logger.error(
                f"Error extracting thumbnail for frame {frame_number}, face {face_id}: {e}"
            )
            return None

    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normalize face embedding for consistent comparison.

        Args:
            embedding: Raw face embedding vector

        Returns:
            Normalized embedding vector
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def detect_faces_in_frame(
        self, frame: np.ndarray, frame_number: int, fps: float, chunk_number: int
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame and extract thumbnails.

        Args:
            frame: The video frame
            frame_number: Current frame number
            fps: Frames per second of the video
            chunk_number: Current chunk number for directory organization

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
                    frame, face.bbox, frame_number, i, chunk_number
                )

                # Extract and normalize face embedding
                embedding = None
                if hasattr(face, "normed_embedding"):
                    embedding = face.normed_embedding.tolist()
                elif hasattr(face, "embedding"):
                    # Normalize the embedding manually if it's not already normalized
                    normalized_embedding = self.normalize_embedding(face.embedding)
                    embedding = normalized_embedding.tolist()

                face_data = {
                    "face_id": i,
                    "timestamp": timestamp,
                    "frame_number": frame_number,
                    "chunk_number": chunk_number,
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
                    "embedding": embedding,  # This is the crucial addition for face grouping
                    "thumbnail_path": thumbnail_path,
                }

                frame_results.append(face_data)

            return frame_results

        except Exception as e:
            logger.error(f"Error detecting faces in frame {frame_number}: {e}")
            return []

    def detect_faces_in_batch(
        self, frames_batch: List[tuple], chunk_number: int
    ) -> List[Dict[str, Any]]:
        """
        Process multiple frames in a batch for better GPU utilization.

        Args:
            frames_batch: List of (frame, frame_number, fps) tuples
            chunk_number: Current chunk number for directory organization

        Returns:
            List of face detection results for the batch
        """
        batch_results = []

        # Process frames in smaller sub-batches to avoid memory issues
        sub_batch_size = min(8, len(frames_batch))

        for i in range(0, len(frames_batch), sub_batch_size):
            sub_batch = frames_batch[i : i + sub_batch_size]

            for frame, frame_number, fps in sub_batch:
                # Process each frame - InsightFace doesn't support true batching
                frame_results = self.detect_faces_in_frame(
                    frame, frame_number, fps, chunk_number
                )
                batch_results.extend(frame_results)

        return batch_results

    def process_video_chunk_pipeline(
        self,
        start_time: float,
        end_time: float,
        frame_skip: int = 30,
        chunk_number: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of the video with pipeline approach for better GPU utilization.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            frame_skip: Number of frames to skip between processing
            chunk_number: Current chunk number for directory organization

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
        frames_batch = []

        logger.info(
            f"Processing chunk {chunk_number}: {timedelta(seconds=int(start_time))} to {timedelta(seconds=int(end_time))}"
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
                frames_batch.append((frame.copy(), current_frame_number, fps))
                processed_frames += 1

                # Process batch when it's full
                if len(frames_batch) >= self.batch_size:
                    batch_results = self.detect_faces_in_batch(
                        frames_batch, chunk_number
                    )
                    chunk_results.extend(batch_results)
                    frames_batch = []

                    if processed_frames % 50 == 0:
                        logger.info(
                            f"Chunk {chunk_number}: Processed {processed_frames} frames, found {len(chunk_results)} faces so far"
                        )

            frame_count += 1

        # Process remaining frames in batch
        if frames_batch:
            batch_results = self.detect_faces_in_batch(frames_batch, chunk_number)
            chunk_results.extend(batch_results)

        cap.release()
        logger.info(
            f"Chunk {chunk_number} complete: {len(chunk_results)} faces detected"
        )
        return chunk_results

    def process_video_chunk(
        self,
        start_time: float,
        end_time: float,
        frame_skip: int = 30,
        chunk_number: int = 1,
    ) -> List[Dict[str, Any]]:
        """
        Process a chunk of the video between start_time and end_time.

        Args:
            start_time: Start time in seconds
            end_time: End time in seconds
            frame_skip: Number of frames to skip between processing
            chunk_number: Current chunk number for directory organization

        Returns:
            List of face detection results for this chunk
        """
        # Initialize face detector for this thread
        self.initialize_face_detector()

        # Use pipeline approach for better GPU utilization
        return self.process_video_chunk_pipeline(
            start_time, end_time, frame_skip, chunk_number
        )

    def process_video_parallel(
        self, chunk_duration: int = 600, frame_skip: int = 30
    ) -> None:
        """
        Process the entire video in parallel chunks for maximum GPU utilization.

        Args:
            chunk_duration: Duration of each chunk in seconds (default: 10 minutes)
            frame_skip: Number of frames to skip between processing
        """
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

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
        logger.info(f"Batch size: {self.batch_size} frames")
        logger.info(f"Max parallel chunks: {self.max_parallel_chunks}")
        logger.info(f"Thumbnails will be saved to: {self.thumbnails_dir}")

        # Calculate all chunks
        chunks = []
        current_time = 0
        chunk_number = 1

        while current_time < duration:
            end_time = min(current_time + chunk_duration, duration)
            chunks.append((current_time, end_time, chunk_number))
            current_time = end_time
            chunk_number += 1

        logger.info(f"Total chunks to process: {len(chunks)}")

        # Process chunks in parallel
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_parallel_chunks
        ) as executor:
            futures = []

            for start_time, end_time, chunk_num in chunks:
                future = executor.submit(
                    self.process_video_chunk,
                    start_time,
                    end_time,
                    frame_skip,
                    chunk_num,
                )
                futures.append((future, chunk_num))

            # Collect results as they complete
            for future, chunk_num in futures:
                try:
                    chunk_results = future.result()

                    # Thread-safe results addition
                    with self.results_lock:
                        self.results.extend(chunk_results)

                    # Save intermediate results
                    self.save_results(f"intermediate_results_chunk_{chunk_num}.json")

                    logger.info(
                        f"Chunk {chunk_num} complete. Total faces detected: {len(self.results)}"
                    )

                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_num}: {e}")

        # Save final results
        self.save_results()
        logger.info(
            f"Video processing complete! Total faces detected: {len(self.results)}"
        )
        logger.info(f"Thumbnails saved in: {self.thumbnails_dir}")

    def process_video(self, chunk_duration: int = 600, frame_skip: int = 30) -> None:
        """
        Process the entire video in chunks.

        Args:
            chunk_duration: Duration of each chunk in seconds (default: 10 minutes)
            frame_skip: Number of frames to skip between processing
        """
        # Use parallel processing for maximum GPU utilization
        self.process_video_parallel(chunk_duration, frame_skip)

    def save_results(self, filename: Optional[str] = None) -> None:
        """Save the detection results to JSON file."""
        output_file = Path(filename) if filename else self.output_path

        with self.results_lock:
            metadata = {
                "video_path": str(self.video_path),
                "processing_date": datetime.now().isoformat(),
                "total_faces_detected": len(self.results),
                "total_frames_processed": len(
                    set(result["frame_number"] for result in self.results)
                ),
                "thumbnails_directory": str(self.thumbnails_dir),
                "thumbnail_size": self.thumbnail_size,
                "batch_size": self.batch_size,
                "max_parallel_chunks": self.max_parallel_chunks,
                "embedding_dimension": 512,  # InsightFace typically uses 512-dimensional embeddings
                "chunks_processed": len(
                    set(result["chunk_number"] for result in self.results)
                ),
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

    # Use aggressive settings for maximum GPU utilization
    processor = VideoFaceProcessor(
        video_path,
        output_path,
        thumbnails_dir,
        batch_size=32,  # Increased batch size
        max_parallel_chunks=3,  # Process 3 chunks in parallel
    )

    # Process video with 10-minute chunks, processing every 30th frame
    processor.process_video(chunk_duration=600, frame_skip=30)


if __name__ == "__main__":
    main()
