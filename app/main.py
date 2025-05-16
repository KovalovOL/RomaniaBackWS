from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fer import FER
import cv2
import numpy as np
import asyncio
from typing import Set
import logging


import os
# Отключаем ненужные функции OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
os.environ["OPENCV_IO_ENABLE_JASPER"] = "0"
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

# Log settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class DetectorPool:
    def __init__(self, pool_size: int = 4):
        self.detectors = asyncio.Queue()
        for _ in range(pool_size):
            self.detectors.put_nowait(FER(mtcnn=True))

    async def detect(self, frame: np.ndarray) -> list:
        detector = await self.detectors.get()
        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._safe_detect,
                detector,
                frame
            )
        finally:
            await self.detectors.put(detector)

    def _safe_detect(self, detector: FER, frame: np.ndarray) -> list:
        try:
            return detector.detect_emotions(frame)
        except Exception as e:
            logger.error(f"Detection error: {str(e)}")
            return []

detector_pool = DetectorPool(pool_size=4)
ALLOWED_EMOTIONS: Set[str] = {'happy', 'sad', 'angry', 'surprise', 'neutral', 'fear', 'disgust'}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        init_data = await websocket.receive_json()
        target_emotion = init_data.get("emotion")
        
        # Set confidence 0.1 if it didn't set
        confidence = min(max(
            init_data.get("confidence", 0.1),  # Default 10% 
            0.0
        ), 1.0)

        if target_emotion not in ALLOWED_EMOTIONS:
            await websocket.close(code=1008)
            return

        while True:

            frame_data = await websocket.receive_bytes()
            logger.debug("Frame received")
            
            # Decode frame
            frame = cv2.imdecode(
                np.frombuffer(frame_data, np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if frame is None or frame.size == 0:
                logger.warning("Invalid frame received")
                continue

            # Detect emotion
            emotions_data = await detector_pool.detect(frame)
            logger.debug(f"Detection results: {emotions_data}")

            # Check res
            detected = False
            for face in emotions_data:
                current_confidence = face.get('emotions', {}).get(target_emotion, 0)
                logger.debug(f"{target_emotion} confidence: {current_confidence}")
                
                if current_confidence >= confidence:
                    detected = True
                    break

            # send res
            if detected:
                logger.info(f"Emotion detected: {target_emotion}")
                try:
                    await websocket.send_json({
                        "status": "detected",
                        "emotion": target_emotion,
                        "confidence": float(current_confidence)
                    })
                    logger.info("Notification sent to client")
                except Exception as e:
                    logger.error(f"Failed to send notification: {str(e)}")
                finally:
                    await websocket.close()
                return

    except WebSocketDisconnect:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Connection error: {str(e)}")
        await websocket.close(code=1011)

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,
        log_level="info"
    )