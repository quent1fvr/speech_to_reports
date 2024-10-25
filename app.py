# app.py
import streamlit as st
import os
from audio_processor import AudioProcessor, TranscriptManager
from tempfile import NamedTemporaryFile

def main():
    st.title("Audio Transcription & Report Generator")
    
    # Initialize session state
    if "transcript_manager" not in st.session_state:
        st.session_state.transcript_manager = None
    
    # File upload
    uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
    
    if uploaded_file:
        # HuggingFace token input
        hf_token = st.text_input(
            "Enter HuggingFace Token (required for speaker diarization)",
            type="password"
        )
        
        if st.button("Process Audio"):
            with st.spinner("Processing audio file..."):
                # Save uploaded file temporarily
                with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                try:
                    # Process audio
                    processor = AudioProcessor()
                    results = processor.process_audio(audio_path, hf_token)
                    
                    # Save results
                    output_path = "transcript.json"
                    processor.save_results(results, output_path)
                    
                    # Initialize transcript manager
                    st.session_state.transcript_manager = TranscriptManager()
                    st.session_state.transcript_manager.load_transcript(output_path)
                    
                    st.success("Audio processed successfully!")
                    
                finally:
                    # Cleanup
                    os.unlink(audio_path)
                    
    # Speaker Assignment section
    if st.session_state.transcript_manager:
        st.subheader("Assign Speakers")
        
        for speaker_id in st.session_state.transcript_manager.speakers:
            col1, col2 = st.columns(2)
            with col1:
                name = st.text_input(f"Name for {speaker_id}", key=f"name_{speaker_id}")
            with col2:
                surname = st.text_input(f"Surname for {speaker_id}", key=f"surname_{speaker_id}")
                
            if name and surname:
                st.session_state.transcript_manager.assign_speaker_identity(
                    speaker_id, name, surname
                )
        
        # Generate outputs
        if st.button("Generate Transcript and Report"):
            # Transcript
            transcript = st.session_state.transcript_manager.generate_transcript()
            st.subheader("Transcript")
            st.text_area("Generated Transcript", transcript, height=300)
            
            st.download_button(
                label="Download Transcript",
                data=transcript,
                file_name="transcript.txt",
                mime="text/plain"
            )
            
            # Report
            report = st.session_state.transcript_manager.generate_report()
            st.subheader("Report")
            st.text_area("Generated Report", report, height=200)
            
            st.download_button(
                label="Download Report",
                data=report,
                file_name="report.txt",
                mime="text/plain"
            )

if __name__ == "__main__":
    main()