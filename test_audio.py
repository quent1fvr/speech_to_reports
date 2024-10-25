from audio_processor import AudioProcessor, TranscriptManager
import os
from pathlib import Path

def test_pipeline(audio_path: str, hf_token: str, mistral_api_key: str):
    """
    Test the full pipeline with a given audio file
    
    Args:
        audio_path: Path to the audio file
        hf_token: HuggingFace token for speaker diarization
        mistral_api_key: Mistral API key for content analysis
    """
    print("Starting audio processing pipeline...")
    print(f"Processing file: {audio_path}")
    
    try:
        # Initialize processor
        processor = AudioProcessor()
        print(f"Using device: {processor.device}")
        
        # Process audio
        print("\nProcessing audio file...")
        results = processor.process_audio(audio_path, hf_token)
        
        # Save results in a directory we're sure to have access to
        output_dir = os.path.join(str(Path.home()), 'speech_reports')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transcript.json")
        
        # Save and get the actual path where the file was saved
        saved_path = processor.save_results(results, output_path)
        print(f"\nResults saved to: {saved_path}")
        
        # Initialize transcript manager and load results
        transcript_manager = TranscriptManager(mistral_api_key)
        transcript_manager.load_transcript(saved_path)
        
        # Print detected speakers
        print("\nDetected speakers:")
        for speaker in results["metadata"]["speakers"]:
            print(f"- {speaker}")
        
        # Generate and print transcript
        print("\nGenerated Transcript:")
        print("-" * 50)
        print(transcript_manager.generate_transcript())
        
        # Analyze discussion using Mistral
        print("\nAnalyzing discussion content...")
        transcript_manager.analyze_discussion()
        
        # Generate and print complete report
        print("\nGenerated Report:")
        print("-" * 50)
        print(transcript_manager.generate_report())
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise

if __name__ == "__main__":
    AUDIO_FILE = "/Users/quent1/Documents/speech_report/M_1023_10y10m_1.wav"
    HF_TOKEN = "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"
    MISTRAL_API_KEY = "d8FQhxmLJ3M1kyv84yhkbtEY78HSGQqI"  # Replace with your actual Mistral API key
    
    test_pipeline(AUDIO_FILE, HF_TOKEN, MISTRAL_API_KEY)