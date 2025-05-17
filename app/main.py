from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fer import FER
import cv2
import numpy as np
import asyncio
from typing import Set
import logging
import os
from datetime import datetime

# Настройка окружения OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_JASPER"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

class DetectorPool:
    def __init__(self, pool_size: int = 4):
        logger.info(f"Initializing DetectorPool with {pool_size} detectors")
        self.detectors = asyncio.Queue()
        for i in range(pool_size):
            logger.debug(f"Creating detector {i+1}/{pool_size}")
            self.detectors.put_nowait(FER(mtcnn=False))
        logger.info("All detectors initialized successfully")

    async def detect(self, frame: np.ndarray) -> list:
        logger.debug("Acquiring detector from pool...")
        start_time = datetime.now()
        detector = await self.detectors.get()
        acquire_time = (datetime.now() - start_time).total_seconds()
        logger.debug(f"Detector acquired in {acquire_time:.3f}s. Queue size: {self.detectors.qsize()}")

        try:
            logger.debug("Starting emotion detection...")
            detect_start = datetime.now()
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                self._safe_detect,
                detector,
                frame
            )
            detect_time = (datetime.now() - detect_start).total_seconds()
            logger.debug(f"Detection completed in {detect_time:.3f}s")
            return result
        finally:
            await self.detectors.put(detector)
            logger.debug("Detector returned to pool")

    def _safe_detect(self, detector: FER, frame: np.ndarray) -> list:
        try:
            logger.debug(f"Detecting emotions on frame (shape: {frame.shape})")
            results = detector.detect_emotions(frame)
            logger.debug(f"Detection results: {results}")
            return results
        except Exception as e:
            logger.error(f"Detection error: {str(e)}", exc_info=True)
            return []

detector_pool = DetectorPool(pool_size=4)
ALLOWED_EMOTIONS: Set[str] = {'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust'}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_ip = websocket.client.host if websocket.client else "unknown"
    logger.info(f"New WebSocket connection from {client_ip}")
    
    try:
        await websocket.accept()
        logger.info(f"WebSocket connection accepted with {client_ip}")

        # Получаем начальные данные
        logger.debug("Waiting for initial data...")
        init_data = await websocket.receive_json()
        logger.info(f"Initial data received: {init_data}")
        
        target_emotion = init_data.get("emotion")
        confidence = min(max(init_data.get("confidence", 0.1), 0.0), 1.0)
        logger.info(f"Target emotion: {target_emotion}, confidence threshold: {confidence}")

        if target_emotion not in ALLOWED_EMOTIONS:
            logger.warning(f"Invalid emotion requested: {target_emotion}")
            await websocket.close(code=1008)
            return

        frame_counter = 0
        while True:
            frame_counter += 1
            logger.debug(f"Waiting for frame #{frame_counter}...")
            
            # Получаем кадр
            frame_start = datetime.now()
            frame_data = await websocket.receive_bytes()
            receive_time = (datetime.now() - frame_start).total_seconds()
            logger.debug(f"Frame #{frame_counter} received in {receive_time:.3f}s, size: {len(frame_data)} bytes")
            logger.info("Get Frame")


            # Декодируем кадр
            decode_start = datetime.now()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            decode_time = (datetime.now() - decode_start).total_seconds()
            
            if frame is None or frame.size == 0:
                logger.warning(f"Invalid frame #{frame_counter} received")
                continue
                
            logger.debug(f"Frame #{frame_counter} decoded in {decode_time:.3f}s, shape: {frame.shape}")

            # Детекция эмоций
            detect_start = datetime.now()
            emotions_data = await detector_pool.detect(frame)
            detect_time = (datetime.now() - detect_start).total_seconds()
            logger.debug(f"Frame #{frame_counter} processed in {detect_time:.3f}s. Results: {len(emotions_data)} faces detected")

            # Проверка результатов
            detected = False
            for i, face in enumerate(emotions_data):
                logger.info("In FOR")


                current_confidence = face.get('emotions', {}).get(target_emotion, 0)
                logger.debug(f"Face #{i+1}: {target_emotion} confidence = {current_confidence:.2f}")
                
                if current_confidence >= confidence:
                    detected = True
                    logger.info(f"Target emotion detected on face #{i+1} with confidence {current_confidence:.2f}")
                    break

            # Отправка результата
            if detected:
                response = {
                    "status": "detected",
                    "emotion": target_emotion,
                    "confidence": float(current_confidence),
                    "processing_time": detect_time,
                    "frame_number": frame_counter
                }
                logger.info(f"Sending detection response: {response}")
                
                try:
                    await websocket.send_json(response)
                    logger.info("Detection notification sent successfully")
                except Exception as e:
                    logger.error(f"Failed to send notification: {str(e)}", exc_info=True)
                finally:
                    await websocket.close()
                    logger.info(f"WebSocket connection with {client_ip} closed after successful detection")
                return

    except WebSocketDisconnect:
        logger.info(f"Client {client_ip} disconnected")
    except Exception as e:
        logger.error(f"Connection error with {client_ip}: {str(e)}", exc_info=True)
        await websocket.close(code=1011)

@app.get("/health")
async def health_check():
    logger.info("Health check request received")
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server...")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )