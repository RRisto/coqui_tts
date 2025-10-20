#!/usr/bin/env python3
"""
Example client for the Coqui TTS service
Demonstrates both WebSocket and gRPC usage
"""

import asyncio
import json
import base64
import websockets
import grpc
#from tts_pb2 import SynthesizeRequest, HealthCheckRequest
#from tts_pb2_grpc import TTSServiceStub


async def test_websocket_client(host="localhost", port=8080):
    """Test WebSocket client"""
    print(f"Testing WebSocket client on {host}:{port}")
    
    try:
        async with websockets.connect(f"ws://{host}:{port}/synthesize") as websocket:
            # Test synthesis
            request = {
                "text": "Tere maailm! See on test Coqui TTS teenusega."
            }
            
            print(f"Sending request: {request}")
            await websocket.send(json.dumps(request))
            
            response = await websocket.recv()
            result = json.loads(response)
            
            print(f"Response: {result}")
            
            if result.get("success"):
                print(f"Audio data length: {len(result.get('audio_data', ''))}")
                print(f"Sampling rate: {result.get('sampling_rate')}")
                print(f"Duration: {result.get('duration')} seconds")
                
                # Save audio to file
                if result.get('audio_data'):
                    audio_data = base64.b64decode(result['audio_data'])
                    with open("output_websocket.wav", "wb") as f:
                        f.write(audio_data)
                    print("Audio saved to output_websocket.wav")
            else:
                print(f"Error: {result.get('message')}")
                
    except Exception as e:
        print(f"WebSocket error: {e}")

import base64
import io
import json
import wave
import numpy as np

async def test_websocket_client(host="localhost", port=8080):
    """Test WebSocket client with robust audio handling (WAV or raw PCM)."""
    import websockets

    print(f"Testing WebSocket client on {host}:{port}")
    try:
        async with websockets.connect(f"ws://{host}:{port}/synthesize") as websocket:
            request = {
                "text": "Tere maailm! See on test Coqui TTS teenusega."
                # If your server supports it, you can also send:
                # "format": "wav" or "encoding": "pcm_f32le"
            }
            print(f"Sending request: {request}")
            await websocket.send(json.dumps(request))

            response = await websocket.recv()
            result = json.loads(response)
            print(f"Response keys: {list(result.keys())}")

            if not result.get("success"):
                print(f"Error: {result.get('message')}")
                return

            b64 = result.get("audio_data")
            if not b64:
                print("No audio_data in response.")
                return

            audio_bytes = base64.b64decode(b64)
            sr = int(result.get("sampling_rate", 22050))
            encoding = (result.get("encoding") or result.get("format") or "").lower()

            # Heuristic: check for WAV header
            is_wav_bytes = audio_bytes[:4] == b"RIFF"
            print(f"encoding hint: {encoding!r} | RIFF header found: {is_wav_bytes}")

            if is_wav_bytes or encoding == "wav":
                out_path = "output_websocket.wav"
                with open(out_path, "wb") as f:
                    f.write(audio_bytes)
                print(f"Saved WAV to {out_path}")
            else:
                # Assume raw PCM float32 little-endian unless server specifies otherwise
                # If your server sends int16, change dtype to '<i2' and skip scaling
                dtype = "<f4"
                if "pcm_f32" in encoding:
                    dtype = "<f4"
                elif "pcm_s16" in encoding or "pcm_16" in encoding or "int16" in encoding:
                    dtype = "<i2"
                else:
                    # No clear hint → try float32 (common with TTS servers)
                    dtype = "<f4"

                samples = np.frombuffer(audio_bytes, dtype=dtype)

                # Convert to int16 WAV for best player compatibility
                if samples.dtype == np.float32:
                    samples = np.clip(samples, -1.0, 1.0)
                    samples = (samples * 32767.0).astype("<i2")  # int16 little endian

                out_path = "output_websocket.wav"
                with wave.open(out_path, "wb") as wf:
                    wf.setnchannels(1)        # mono (change to 2 if server sends stereo)
                    wf.setsampwidth(2)        # 2 bytes for int16
                    wf.setframerate(sr)
                    wf.writeframes(samples.tobytes())
                print(f"Wrapped raw PCM into WAV → {out_path}")

            # Optional: quick playback on Windows (requires a valid WAV)
            try:
                import winsound
                winsound.PlaySound("output_websocket.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
                print("Playing output_websocket.wav via winsound…")
            except Exception as e:
                print(f"winsound playback skipped: {e}")

    except Exception as e:
        print(f"WebSocket error: {e}")



async def test_websocket_health(host="localhost", port=8080):
    """Test WebSocket health check"""
    print(f"Testing WebSocket health check on {host}:{port}")
    
    try:
        async with websockets.connect(f"ws://{host}:{port}/health") as websocket:
            response = await websocket.recv()
            result = json.loads(response)
            print(f"Health status: {result}")
            
    except Exception as e:
        print(f"WebSocket health check error: {e}")


def test_grpc_client(host="localhost", port=8080):
    """Test gRPC client"""
    print(f"Testing gRPC client on {host}:{port}")
    
    try:
        # Create gRPC channel
        channel = grpc.insecure_channel(f"{host}:{port}")
        stub = TTSServiceStub(channel)
        
        # Test health check
        print("Testing health check...")
        health_response = stub.HealthCheck(HealthCheckRequest(service="tts"))
        print(f"Health status: {health_response.status}")
        print(f"Health message: {health_response.message}")
        
        # Test synthesis
        print("Testing synthesis...")
        request = SynthesizeRequest(text="Tere maailm! See on test Coqui TTS teenusega.")
        response = stub.Synthesize(request)
        
        print(f"Synthesis success: {response.success}")
        print(f"Synthesis message: {response.message}")
        print(f"Audio data length: {len(response.audio_data)}")
        print(f"Sampling rate: {response.sampling_rate}")
        print(f"Duration: {response.duration} seconds")
        
        # Save audio to file
        if response.success and response.audio_data:
            with open("output_grpc.wav", "wb") as f:
                f.write(response.audio_data)
            print("Audio saved to output_grpc.wav")
            
    except Exception as e:
        print(f"gRPC error: {e}")


async def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test TTS service client")
    parser.add_argument("--host", default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=8080, help="Server port")
    parser.add_argument("--mode", choices=["websocket", "grpc", "both"], default="websocket", help="Test mode")
    
    args = parser.parse_args()
    
    print("=== Coqui TTS Service Client Test ===")
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Mode: {args.mode}")
    print()
    
    if args.mode in ["websocket", "both"]:
        print("=== WebSocket Tests ===")
        await test_websocket_health(args.host, args.port)
        await test_websocket_client(args.host, args.port)
        print()
    
    if args.mode in ["grpc", "both"]:
        print("=== gRPC Tests ===")
        test_grpc_client(args.host, args.port)
        print()
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
