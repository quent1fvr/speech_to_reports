# app.py
from utils import patch_streamlit_watcher
patch_streamlit_watcher()

import streamlit as st
import os
from audio_processor import AudioProcessor, TranscriptManager
from tempfile import NamedTemporaryFile
from config import MISTRAL_API_KEY, HUGGING_FACE_TOKEN
def main():
    st.title("Audio Transcription & Report Generator BY HEXAMIND")
    
    # Initialize session state
    if "transcript_manager" not in st.session_state:
        st.session_state.transcript_manager = None
    
    # Language selection
    st.subheader("Language Settings")
    language = st.selectbox(
        "Select Audio Language (for transcription and analysis)",
        options=[
            ("English", "en"),
            ("French", "fr"),
        ],
        help="This will be used for both speech recognition and report generation",
        format_func=lambda x: x[0],
        index=0
    )[1]  # Get the language code
    
    # File upload
    uploaded_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'])
    
    if uploaded_file:
        # HuggingFace token input
        if st.button("Process Audio"):
            with st.spinner("Processing audio file..."):
                # Save uploaded file temporarily
                with NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    audio_path = tmp_file.name
                
                try:
                    # Process audio
                    processor = AudioProcessor(language=language)
                    results = processor.process_audio(audio_path, hf_token=HUGGING_FACE_TOKEN)
                    
                    # Save results
                    output_path = "transcript.json"
                    processor.save_results(results, output_path)
                    
                    # Initialize transcript manager with language
                    st.session_state.transcript_manager = TranscriptManager(
                        mistral_api_key=MISTRAL_API_KEY,
                        language=language
                    )
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


if __name__ == "__main__":
    main()