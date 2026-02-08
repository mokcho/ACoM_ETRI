import opuslib
import numpy as np
import soundfile as sf
import os
from pathlib import Path
import argparse

def encode_decode_opus(audio_np, sample_rate=16000, bitrate=12000, frame_size=320):
    """
    Encode and decode audio with Opus
    
    Args:
        audio_np: numpy array of audio samples (float32, range -1 to 1)
        sample_rate: sample rate in Hz
        bitrate: target bitrate in bps
        frame_size: frame size in samples (20ms at 16kHz = 320 samples)
    
    Returns:
        decoded audio as numpy array
    """
    # Ensure mono
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    
    # Pad to multiple of frame_size
    orig_length = len(audio_np)
    pad_length = (frame_size - (orig_length % frame_size)) % frame_size
    
    if pad_length > 0:
        audio_np = np.pad(audio_np, (0, pad_length), mode='constant')
    
    # Convert to int16 for Opus
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)
    
    # Create encoder and decoder
    encoder = opuslib.Encoder(
        fs=sample_rate,
        channels=1,
        application=opuslib.APPLICATION_AUDIO
    )
    encoder.bitrate = bitrate
    
    decoder = opuslib.Decoder(
        fs=sample_rate,
        channels=1
    )
    
    # Encode frame by frame
    encoded_frames = []
    for i in range(0, len(audio_int16), frame_size):
        frame = audio_int16[i:i+frame_size]
        if len(frame) == frame_size:
            encoded = encoder.encode(frame.tobytes(), frame_size)
            encoded_frames.append(encoded)
    
    # Decode
    decoded_audio = []
    for encoded in encoded_frames:
        decoded = decoder.decode(encoded, frame_size)
        decoded_int16 = np.frombuffer(decoded, dtype=np.int16)
        decoded_audio.append(decoded_int16)
    
    # Concatenate and convert back to float
    decoded_np = np.concatenate(decoded_audio).astype(np.float32) / 32767.0
    
    # Remove padding
    return decoded_np[:orig_length]

def process_audio_file(input_path, output_path, bitrate=12000, sample_rate=14000):
    """Process a single audio file"""
    # Read audio
    audio, sr = sf.read(input_path)
    
    if os.path.exists(output_path) :
        return
    
    # Resample if needed (you might want to use librosa for better resampling)
    if sr != sample_rate:
        print(f"Warning: {input_path} has sample rate {sr}, expected {sample_rate}")
        # Simple resampling - consider using librosa.resample for better quality
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)
    
    # Encode/decode with Opus
    frame_size = int(sample_rate * 0.02)  # 20ms frame
    decoded = encode_decode_opus(audio, sample_rate=sample_rate, bitrate=bitrate, frame_size=frame_size)
    
    # Save
    sf.write(output_path, decoded, sample_rate)
    print(f"Processed: {input_path} -> {output_path} ({bitrate} bps)")

def main():
    parser = argparse.ArgumentParser(description='Encode audio files with Opus codec')
    parser.add_argument('input_dir', type=str, help='Input directory with audio files')
    parser.add_argument('output_dir', type=str, help='Output directory for encoded files')
    parser.add_argument('--bitrate', type=float, default=12.0, 
                       help='Bitrate in kbps (default: 12.0)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                       help='Sample rate in Hz (default: 16000)')
    parser.add_argument('--extensions', nargs='+', default=['.wav', '.flac', '.mp3'],
                       help='Audio file extensions to process')
    
    args = parser.parse_args()
    
    # Convert kbps to bps
    bitrate_bps = int(args.bitrate * 1000)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get all audio files
    input_path = Path(args.input_dir)
    audio_files = []
    for ext in args.extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Opus bitrate: {args.bitrate} kbps ({bitrate_bps} bps)")
    print(f"Sample rate: {args.sample_rate} Hz")
    
    # Process each file
    for audio_file in audio_files:
        # Preserve directory structure
        relative_path = audio_file.relative_to(input_path)
        output_path = Path(args.output_dir) / relative_path
        
        # Create subdirectories if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Change extension to .wav
        output_path = output_path.with_suffix('.wav')
        
        try:
            process_audio_file(str(audio_file), str(output_path), 
                             bitrate=bitrate_bps, sample_rate=args.sample_rate)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
    
    print("Done!")

if __name__ == "__main__":
    main()