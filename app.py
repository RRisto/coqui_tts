import os
import sys
import logging
import asyncio
import json
import base64
import io
import wave
from pathlib import Path
import numpy as np
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import Coqui TTS
try:
    from TTS.api import TTS
    from TTS.tts.configs.xtts_config import XttsConfig
    import torch
    from torch.serialization import add_safe_globals

    add_safe_globals([XttsConfig])
    TTS_AVAILABLE = True
except ImportError as e:
    logger.error(f"TTS not available: {e}")
    TTS_AVAILABLE = False

# Import gRPC dependencies
try:
    from tts_pb2 import (
        SynthesizeRequest,
        SynthesizeResponse,
        HealthCheckRequest,
        HealthCheckResponse,
    )
    from tts_pb2_grpc import add_TTSServiceServicer_to_server, TTSServiceServicer

    GRPC_AVAILABLE = True
except ImportError as e:
    logger.error(f"gRPC generated code not found: {e}")
    GRPC_AVAILABLE = False

# Import WebSocket dependencies
try:
    from fastapi import FastAPI, WebSocket
    import uvicorn

    WEBSOCKET_AVAILABLE = True
except ImportError:
    logger.warning("FastAPI/uvicorn not available for WebSocket support")
    WEBSOCKET_AVAILABLE = False

# Import Ray Serve
try:
    from ray import serve
    from ray.serve.config import gRPCOptions

    RAY_AVAILABLE = True
except ImportError:
    logger.error("Ray Serve not available")
    RAY_AVAILABLE = False


class CoquiTTSService:
    """Coqui TTS Service for Estonian text-to-speech"""

    def __init__(self, model_path: str = None, config_path: str = None):
        # Keep your defaults
        self.model_path = "/app/model/model_file.pth.tar"
        self.config_path = "model/config.json"
        self.model = None
        self.tts = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Coqui TTS model"""
        try:
            logger.info("=== INITIALIZING COQUI TTS MODEL ===")

            model_path = Path(self.model_path)
            config_path = Path(self.config_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model path not found: {model_path}")
            if not config_path.exists():
                raise FileNotFoundError(f"Config path not found: {config_path}")

            # IMPORTANT: don't shadow model_path variable; force CPU
            self.tts = TTS(model_path=str(model_path), config_path=str(config_path), gpu=False)

            # --- keep your monkey patch exactly as requested ---
            import types
            def _patched_check_arguments(self, **kwargs):
                # Skip checks; assume single-speaker, single-language
                return
            self.tts._check_arguments = types.MethodType(_patched_check_arguments, self.tts)
            # ---------------------------------------------------

            logger.info(f"TTS model loaded from: {model_path}")
            logger.info("=== MODEL INITIALIZATION COMPLETE ===")

        except Exception as e:
            logger.exception("Failed to initialize TTS model")
            raise

    def _detect_sample_rate(self) -> int:
        """Try to detect the real sample rate from the synthesizer/config."""
        sr = None
        try:
            syn = getattr(self.tts, "synthesizer", None)
            if syn is not None:
                # Common in newer versions
                sr = getattr(syn, "output_sample_rate", None)
                if not sr:
                    cfg = getattr(syn, "tts_config", None)
                    if cfg is not None:
                        audio = getattr(cfg, "audio", None)
                        if isinstance(audio, dict):
                            sr = audio.get("sample_rate")
                        elif audio is not None:
                            sr = getattr(audio, "sample_rate", None)
        except Exception:
            pass
        return int(sr or 22050)

    def synthesize(self, text: str) -> dict:
        """Synthesize text to speech using Coqui TTS, return WAV bytes."""
        try:
            if not self.tts:
                return {
                    "success": False,
                    "message": "TTS model not initialized",
                    "audio_data": b"",
                    "sampling_rate": 0,
                    "duration": 0.0,
                    "format": "wav",
                }

            text = (text or "").strip()
            if not text:
                return {
                    "success": False,
                    "message": "No text provided",
                    "audio_data": b"",
                    "sampling_rate": 0,
                    "duration": 0.0,
                    "format": "wav",
                }

            logger.info(f"Synthesizing text: '{text[:50]}{'...' if len(text) > 50 else ''}'")

            # Generate audio as float32 -1..1
            wav = self.tts.tts(text=text)
            if hasattr(wav, "numpy"):
                wav = wav.numpy()
            wav = np.asarray(wav, dtype=np.float32)

            # Get correct sample rate
            sampling_rate = self._detect_sample_rate()

            # Convert to int16 PCM
            wav = np.clip(wav, -1.0, 1.0)
            pcm16 = (wav * 32767.0).astype("<i2")

            # Make a proper WAV container
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)     # mono (change if stereo)
                wf.setsampwidth(2)     # 2 bytes → int16
                wf.setframerate(sampling_rate)
                wf.writeframes(pcm16.tobytes())
            audio_bytes = buf.getvalue()

            duration = float(len(pcm16)) / float(sampling_rate) if sampling_rate > 0 else 0.0

            logger.info(f"Synthesis completed: {len(audio_bytes)} bytes WAV, {duration:.2f}s @ {sampling_rate} Hz")

            return {
                "success": True,
                "message": "Synthesis completed successfully",
                "audio_data": audio_bytes,   # WAV bytes
                "sampling_rate": sampling_rate,
                "duration": duration,
                "format": "wav",
            }

        except Exception as e:
            logger.exception("Synthesis failed")
            return {
                "success": False,
                "message": f"Synthesis failed: {str(e)}",
                "audio_data": b"",
                "sampling_rate": 0,
                "duration": 0.0,
                "format": "wav",
            }

    def health_check(self) -> dict:
        """Check if the service is healthy"""
        try:
            if self.tts:
                return {"status": "SERVING", "message": "TTS service is healthy"}
            else:
                return {"status": "NOT_SERVING", "message": "TTS service not initialized"}
        except Exception as e:
            logger.exception("Health check failed")
            return {"status": "NOT_SERVING", "message": f"Health check failed: {str(e)}"}


# Global TTS service instance
tts_service = None

def initialize_tts_service():
    """Initialize the global TTS service"""
    global tts_service
    if tts_service is None:
        model_path = os.getenv("TTS_MODEL_PATH", "/app/model")
        tts_service = CoquiTTSService(model_path)
    return tts_service


# gRPC Service Implementation
if GRPC_AVAILABLE and RAY_AVAILABLE:
    @serve.deployment(ray_actor_options={"num_cpus": 1}, name="tts")
    class TTSGrpcServicer:
        def __init__(self):
            self.tts_service = initialize_tts_service()

        def Synthesize(self, request: SynthesizeRequest) -> SynthesizeResponse:
            """Synthesize text to speech via gRPC"""
            result = self.tts_service.synthesize(request.text)
            return SynthesizeResponse(
                success=result["success"],
                message=result["message"],
                audio_data=result["audio_data"],        # raw WAV bytes on gRPC
                sampling_rate=result["sampling_rate"],
                duration=result["duration"]
            )

        def HealthCheck(self, request: HealthCheckRequest) -> HealthCheckResponse:
            """Health check via gRPC"""
            result = self.tts_service.health_check()
            status_map = {
                "SERVING": HealthCheckResponse.SERVING,
                "NOT_SERVING": HealthCheckResponse.NOT_SERVING
            }
            return HealthCheckResponse(
                status=status_map.get(result["status"], HealthCheckResponse.UNKNOWN),
                message=result["message"]
            )

# WebSocket Service Implementation with Ray Serve
if WEBSOCKET_AVAILABLE and RAY_AVAILABLE:
    @serve.deployment(ray_actor_options={"num_cpus": 1}, name="tts-websocket")
    class TTSWebSocketService:
        def __init__(self):
            self.tts_service = initialize_tts_service()

        async def synthesize_websocket(self, text: str) -> dict:
            """Synthesize text to speech for WebSocket"""
            result = self.tts_service.synthesize(text)
            # Base64 WAV for JSON transport
            if result["success"] and result["audio_data"]:
                result["audio_data"] = base64.b64encode(result["audio_data"]).decode('utf-8')
            return result

        async def health_check_websocket(self) -> dict:
            """Health check for WebSocket"""
            return self.tts_service.health_check()

    async def serve_websocket(port: int = 8080):
        """Start WebSocket server for TTS"""
        app = FastAPI()

        # Wait for Ray Serve deployment to be ready
        if not wait_for_deployment("tts-websocket", "tts-websocket"):
            logger.error("WebSocket deployment not ready, exiting")
            return

        handle = serve.get_deployment_handle("tts-websocket", app_name="tts-websocket")

        @app.websocket("/synthesize")
        async def websocket_synthesize(websocket: WebSocket):
            await websocket.accept()
            try:
                data = await websocket.receive_text()
                request_data = json.loads(data)
                text = request_data.get("text", "")
                if not text:
                    await websocket.send_json({"success": False, "message": "No text provided"})
                    return

                logger.info(f"WebSocket synthesis request: text='{text[:50]}{'...' if len(text) > 50 else ''}'")
                result = await handle.synthesize_websocket.remote(text)
                await websocket.send_json(result)
                logger.info(f"WebSocket synthesis completed: success={result.get('success', False)}")

            except Exception as e:
                logger.exception(f"WebSocket synthesis error: {e}")
                await websocket.send_json({"success": False, "message": f"WebSocket error: {str(e)}"})

        @app.websocket("/health")
        async def websocket_health(websocket: WebSocket):
            await websocket.accept()
            try:
                result = await handle.health_check_websocket.remote()
                await websocket.send_json(result)
            except Exception as e:
                logger.exception(f"WebSocket health check error: {e}")
                await websocket.send_json({"status": "NOT_SERVING", "message": f"Health check error: {str(e)}"})

        logger.info(f"WebSocket server listening on port {port}")
        config = uvicorn.Config(app, host="0.0.0.0", port=port)
        server = uvicorn.Server(config)
        await server.serve()


def wait_for_deployment(name="tts", app_name="tts", timeout=300):
    """Wait for Ray Serve deployment to be ready"""
    import time
    start = time.time()
    while time.time() - start < timeout:
        try:
            serve.get_deployment_handle(name, app_name=app_name)
            logger.info(f"Deployment '{name}' is ready!")
            return True
        except Exception:
            logger.info(f"Waiting for deployment '{name}'...")
            time.sleep(5)
    return False


def main():
    """Main entry point"""
    if not TTS_AVAILABLE:
        logger.error("TTS not available. Please install Coqui TTS.")
        sys.exit(1)

    if not RAY_AVAILABLE:
        logger.error("Ray not available. Please install Ray.")
        sys.exit(1)

    port = int(os.getenv("PORT", "8080"))
    server_type = os.getenv("SERVER_TYPE", "websocket").lower()

    logger.info(f"Starting TTS server in {server_type.upper()} mode on port {port}")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")

    import ray
    ray.init()
    if server_type == "websocket":
        if not WEBSOCKET_AVAILABLE:
            logger.error("WebSocket dependencies not available. Install fastapi and uvicorn.")
            sys.exit(1)

        serve.start()
        serve.run(TTSWebSocketService.bind(), name="tts-websocket")
        logger.info("Ray Serve WebSocket deployment started")
        asyncio.run(serve_websocket(port=port))

    elif server_type == "grpc":
        if not GRPC_AVAILABLE:
            logger.error("gRPC dependencies not available.")
            sys.exit(1)

        serve.start(
            grpc_options=gRPCOptions(
                port=port,
                grpc_servicer_functions=["tts_pb2_grpc.add_TTSServiceServicer_to_server"],
            )
        )
        serve.run(TTSGrpcServicer.bind(), name="tts")
        logger.info(f"gRPC server ready on port {port}")

        import threading
        threading.Event().wait()

    else:
        logger.error(f"Unknown SERVER_TYPE: {server_type}. Use 'grpc' or 'websocket'")
        sys.exit(1)


if __name__ == "__main__":
    main()
