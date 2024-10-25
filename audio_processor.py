import os
import json
import whisperx
import torch
import datetime
import requests
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from pathlib import Path

@dataclass
class Speaker:
    id: str
    name: str = ""
    surname: str = ""
    
    @property
    def full_name(self) -> str:
        return f"{self.name} {self.surname}" if self.name and self.surname else self.id

@dataclass 
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str
    words: List[Dict]

class MistralAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.mistral.ai/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
    
    def analyze_discussion(self, transcript: str) -> Tuple[str, str]:
        """
        Analyze the discussion to extract themes and summary
        Returns: Tuple of (themes, summary)
        """
        messages = [
            {
                "role": "system",
                "content": "You are an expert at analyzing conversations and identifying key themes and creating summaries."
            },
            {
                "role": "user",
                "content": f"""Please analyze this transcript and provide two sections:
                1. THEMES: List the main themes or topics of discussion
                2. SUMMARY: Provide a concise summary of the key points discussed

                Transcript:
                {transcript}
                
                Format your response in FRENCH with the headers 'THEMES:' and 'SUMMARY:' on separate lines."""
            }
        ]
        
        try:
            response = requests.post(
                self.api_url,
                headers=self.headers,
                json={
                    "model": "mistral-small-latest",
                    "messages": messages,
                    "temperature": 0,
                    "max_tokens": 1000
                }
            )
            response.raise_for_status()
            analysis = response.json()["choices"][0]["message"]["content"]
            
            parts = analysis.split("SUMMARY:")
            themes = parts[0].replace("THEMES:", "").strip()
            summary = parts[1].strip() if len(parts) > 1 else ""
            
            return themes, summary
            
        except Exception as e:
            raise RuntimeError(f"Error analyzing transcript with Mistral: {str(e)}")

class AudioProcessor:
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.batch_size = 16 if self.device == "cuda" else 4
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
    def process_audio(self, audio_path: str, hf_token: Optional[str] = None) -> Dict:
        """
        Process audio file through the full pipeline
        """
        try:
            # 1. Load Audio
            audio = whisperx.load_audio(audio_path)
            
            # 2. Load ASR Model
            model = whisperx.load_model(
                "small", 
                self.device,
                compute_type=self.compute_type
            )
            
            # 3. Transcribe with updated parameters
            result = model.transcribe(
                audio, 
                batch_size=self.batch_size,
                language='en'  # Specify language to avoid auto-detection
            )
            
            # 4. Align Whisper output
            model_a, metadata = whisperx.load_align_model(
                language_code=result["language"],
                device=self.device
            )
            result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # 5. Diarize with safety checks
            diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=hf_token,
                device=self.device
            )
            diarize_segments = diarize_model(
                audio,
                min_speakers=1,
                max_speakers=10
            )
            
            # 6. Assign word speakers
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            return self._process_results(result)
            
        except Exception as e:
            raise RuntimeError(f"Error processing audio: {str(e)}")
            
    def _process_results(self, whisperx_output: Dict) -> Dict:
        """Process WhisperX output into our format"""
        speakers = set()
        segments = []
        
        for segment in whisperx_output["segments"]:
            speaker_id = segment.get("speaker", "UNKNOWN")
            speakers.add(speaker_id)
            
            segments.append(TranscriptSegment(
                start=segment["start"],
                end=segment["end"],
                text=segment["text"],
                speaker=speaker_id,
                words=segment.get("words", [])
            ))
            
        return {
            "metadata": {
                "processed_at": datetime.datetime.now().isoformat(),
                "speakers": list(speakers)
            },
            "segments": [asdict(s) for s in segments]
        }
    
    def save_results(self, results: Dict, output_path: str) -> str:
        """
        Save results to JSON file
        Returns the actual path where the file was saved
        """
        try:
            # Convert to absolute path and ensure directory exists
            output_path = os.path.abspath(output_path)
            output_dir = os.path.dirname(output_path)
            
            print(f"Attempting to save to: {output_path}")
            print(f"Creating directory if needed: {output_dir}")
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Verify the results dictionary has content
            if not results or not isinstance(results, dict):
                raise ValueError(f"Invalid results format: {type(results)}")
                
            if "metadata" not in results or "segments" not in results:
                raise ValueError(f"Missing required keys in results. Keys present: {results.keys()}")
            
            # Try to save in the specified location
            print(f"Writing JSON file...")
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved to: {output_path}")
            return output_path
                
        except PermissionError as pe:
            print(f"Permission error when saving to {output_path}")
            # If permission denied, save in user's home directory
            home_dir = str(Path.home())
            new_path = os.path.join(home_dir, 'speech_report_transcript.json')
            print(f"Attempting to save to alternate location: {new_path}")
            with open(new_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved to alternate location: {new_path}")
            return new_path
        except Exception as e:
            print(f"Error while saving results: {str(e)}")
            raise

class TranscriptManager:
    def __init__(self, mistral_api_key: Optional[str] = None):
        self.speakers = {}
        self.segments = []
        self.mistral_analyzer = MistralAnalyzer(mistral_api_key) if mistral_api_key else None
        self.themes = ""
        self.summary = ""
        
    def load_transcript(self, transcript_path: str):
        """Load transcript from JSON file"""
        with open(transcript_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Initialize speakers
        for speaker_id in data["metadata"]["speakers"]:
            self.speakers[speaker_id] = Speaker(id=speaker_id)
            
        self.segments = data["segments"]
        
    def assign_speaker_identity(self, speaker_id: str, name: str, surname: str):
        """Assign name to speaker ID"""
        if speaker_id in self.speakers:
            self.speakers[speaker_id].name = name
            self.speakers[speaker_id].surname = surname
            
    def generate_transcript(self) -> str:
        """Generate formatted transcript"""
        transcript = []
        transcript.append("Transcript")
        transcript.append(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        for segment in self.segments:
            speaker = self.speakers.get(segment["speaker"])
            speaker_name = speaker.full_name if speaker else segment["speaker"]
            
            start = datetime.timedelta(seconds=int(segment["start"]))
            end = datetime.timedelta(seconds=int(segment["end"]))
            
            transcript.append(f"[{start} - {end}] {speaker_name}:")
            transcript.append(f"{segment['text']}\n")
            
        return "\n".join(transcript)
    
    def analyze_discussion(self) -> None:
        """Analyze the transcript using Mistral AI"""
        if not self.mistral_analyzer:
            raise RuntimeError("Mistral API key not provided. Cannot analyze discussion.")
        
        transcript = self.generate_transcript()
        self.themes, self.summary = self.mistral_analyzer.analyze_discussion(transcript)
    
    def generate_report(self) -> str:
        """Generate comprehensive meeting report"""
        total_duration = max(s["end"] for s in self.segments)
        speaker_times = {}
        
        for segment in self.segments:
            speaker = segment["speaker"]
            duration = segment["end"] - segment["start"]
            speaker_times[speaker] = speaker_times.get(speaker, 0) + duration
            
        report = []
        report.append("Meeting Report")
        report.append(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"Duration: {int(total_duration // 60)} minutes\n")
        
        report.append("Participants:")
        for speaker_id, time in speaker_times.items():
            speaker = self.speakers.get(speaker_id)
            name = speaker.full_name if speaker else speaker_id
            report.append(f"- {name} (Speaking time: {int(time // 60)} minutes)")
        
        if self.themes or self.summary:
            report.append("\nDiscussion Analysis:")
            if self.themes:
                report.append("\nMain Themes:")
                report.append(self.themes)
            if self.summary:
                report.append("\nSummary:")
                report.append(self.summary)
            
        return "\n".join(report)