#!/usr/bin/env python3
"""
TesseractCSMHelper - CSM-1B Text-to-Speech Server
Runs on localhost:5050 and provides REST API for speech synthesis.
"""

import os
import sys
import signal
import tempfile
import threading
import time
from pathlib import Path

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin for local development

# Global state
model = None
processor = None
device = None
model_loaded = False
loading_progress = {"status": "idle", "progress": 0, "message": ""}

# Configuration
MODEL_ID = "sesame/csm-1b"
DEFAULT_PORT = 5050
HOST = "127.0.0.1"  # Only accept local connections


def get_device():
    """Determine best available device."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model_async():
    """Load the model in a background thread."""
    global model, processor, device, model_loaded, loading_progress
    
    try:
        loading_progress = {"status": "loading", "progress": 0.1, "message": "Importing libraries..."}
        
        import torch
        from transformers import CsmForConditionalGeneration, AutoProcessor
        
        device = get_device()
        loading_progress = {"status": "loading", "progress": 0.2, "message": f"Using device: {device}"}
        print(f"[CSMHelper] Using device: {device}")
        
        loading_progress = {"status": "loading", "progress": 0.3, "message": "Loading processor..."}
        print("[CSMHelper] Loading processor...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        
        loading_progress = {"status": "loading", "progress": 0.5, "message": "Loading model (this may take a while)..."}
        print("[CSMHelper] Loading model...")
        
        dtype = torch.float16 if device != "cpu" else torch.float32
        model = CsmForConditionalGeneration.from_pretrained(
            MODEL_ID,
            device_map=device if device != "mps" else None,
            torch_dtype=dtype
        )
        
        if device == "mps":
            model = model.to(device)
        
        loading_progress = {"status": "ready", "progress": 1.0, "message": "Model loaded successfully!"}
        model_loaded = True
        print("[CSMHelper] Model loaded successfully!")
        
    except Exception as e:
        loading_progress = {"status": "error", "progress": 0, "message": str(e)}
        print(f"[CSMHelper] Error loading model: {e}")


# ============== API Routes ==============

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'model_loaded': model_loaded,
        'device': device
    })


@app.route('/status')
def status():
    """Detailed status including loading progress."""
    return jsonify({
        'model_loaded': model_loaded,
        'device': device,
        'model_id': MODEL_ID,
        'loading': loading_progress
    })


@app.route('/synthesize', methods=['POST'])
def synthesize():
    """Synthesize speech from text."""
    global model, processor
    
    if not model_loaded:
        return jsonify({'error': 'Model not loaded yet'}), 503
    
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data provided'}), 400
    
    text = data.get('text', '').strip()
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    speaker_id = data.get('speaker_id', 0)
    
    try:
        import torch
        
        # Prepare input
        conversation = [{
            "role": str(speaker_id),
            "content": [{"type": "text", "text": text}]
        }]
        
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True
        )
        inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}
        
        # Generate audio
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True)
        
        # Save to temp file
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        processor.save_audio(audio, tmp.name)
        
        return send_file(
            tmp.name,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='speech.wav'
        )
        
    except Exception as e:
        print(f"[CSMHelper] Synthesis error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/voices')
def voices():
    """List available voices/speakers."""
    return jsonify({
        'voices': [
            {'id': 0, 'name': 'Default'},
            {'id': 1, 'name': 'Speaker 1'},
            {'id': 2, 'name': 'Speaker 2'},
            {'id': 3, 'name': 'Speaker 3'},
        ]
    })


@app.route('/shutdown', methods=['POST'])
def shutdown():
    """Gracefully shutdown the server."""
    print("[CSMHelper] Shutdown requested...")
    os._exit(0)


@app.route('/reload', methods=['POST'])
def reload_model():
    """Reload the model."""
    global model_loaded
    model_loaded = False
    thread = threading.Thread(target=load_model_async)
    thread.start()
    return jsonify({'status': 'reloading'})


# ============== Main ==============

def signal_handler(sig, frame):
    print("\n[CSMHelper] Shutting down...")
    sys.exit(0)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='TesseractCSMHelper - CSM TTS Server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'Port to listen on (default: {DEFAULT_PORT})')
    parser.add_argument('--no-load', action='store_true', help='Start without loading model')
    args = parser.parse_args()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print(f"[CSMHelper] Starting TesseractCSMHelper on {HOST}:{args.port}")
    
    # Load model in background thread
    if not args.no_load:
        thread = threading.Thread(target=load_model_async)
        thread.start()
    
    # Start Flask server
    app.run(
        host=HOST,
        port=args.port,
        debug=False,
        threaded=True,
        use_reloader=False
    )


if __name__ == '__main__':
    main()
