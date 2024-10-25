from audio_processor import AudioProcessor, TranscriptManager
import os
from pathlib import Path
import json

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
        
        # Verify results
        print("\nVerifying results structure...")
        if not results:
            raise ValueError("No results returned from processing")
        print(f"Results keys: {results.keys()}")
        print(f"Number of segments: {len(results.get('segments', []))}")
        print(f"Speakers detected: {results.get('metadata', {}).get('speakers', [])}")
        
        # Save results in a directory we're sure to have access to
        output_dir = os.path.join(str(Path.home()), 'speech_reports')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "transcript.json")
        
        # Save and get the actual path where the file was saved
        print(f"\nSaving results to: {output_path}")
        saved_path = processor.save_results(results, output_path)
        print(f"Results saved to: {saved_path}")
        
        # Verify the saved file exists and can be read
        print("\nVerifying saved file...")
        if not os.path.exists(saved_path):
            raise FileNotFoundError(f"Failed to find saved file at: {saved_path}")
            
        # Try reading the saved file
        with open(saved_path, 'r', encoding='utf-8') as f:
            saved_data = json.load(f)
            print(f"Successfully read saved file with keys: {saved_data.keys()}")
        
        # Initialize transcript manager and load results
        print("\nInitializing transcript manager...")
        transcript_manager = TranscriptManager(mistral_api_key)
        transcript_manager.load_transcript(saved_path)
        
        # Print detected speakers
        print("\nDetected speakers:")
        for speaker in results["metadata"]["speakers"]:
            print(f"- {speaker}")
        
        # Generate and print transcript
        print("\nGenerating transcript...")
        transcript_text = transcript_manager.generate_transcript()
        print("-" * 50)
        print(transcript_text)
        
        # Analyze discussion using Mistral
        print("\nAnalyzing discussion content...")
        transcript_manager.analyze_discussion()
        
        # Generate and print complete report
        print("\nGenerating report...")
        report_text = transcript_manager.generate_report()
        print("-" * 50)
        print(report_text)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    AUDIO_FILE = "/Users/quent1/Documents/speech_report/cadeau-anniversaire-ironie__=1.mp3"
    HF_TOKEN = "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"
    MISTRAL_API_KEY = "d8FQhxmLJ3M1kyv84yhkbtEY78HSGQqI"  # Replace with your actual Mistral API key
    
    test_pipeline(AUDIO_FILE, HF_TOKEN, MISTRAL_API_KEY)