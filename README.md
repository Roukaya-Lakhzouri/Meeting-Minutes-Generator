
# Meeting Minutes Generator (WIP)

## **Project Overview**

This project aims to automate meeting minute generation from multi-speaker audio recordings. It combines **speech recognition**, **speaker diarization**, **text summarization**, and **action-item extraction**.

**Current Status:** Work in progress. Core components developed; pipeline not yet finalized.

**Implemented Features:**

* Data collection from the **AMI Meeting Corpus**
* Voice preprocessing: noise reduction, silence trimming, mono conversion
* Speech-to-text conversion with **Whisper** and **Wav2Vec2**
* LLM-based summarization setup using deepseek-r1
  

**Planned Features:**

* Full action-item extraction
* Dashboard for structured output and visualization
* Real-time processing capabilities

---

## **System Architecture**

The system currently follows a **four-step AI pipeline**:

1. **Data Collection**

   * Using AMI Meeting Corpus: multi-speaker audio with transcripts and annotations.

2. **Voice Preprocessing**

   * Noise reduction and normalization
   * Silence trimming
   * Conversion to mono `.wav`
   * speech diarization

3. **Speech-to-Text Conversion**

   * OpenAI Whisper for transcription
   * Generates speaker-labeled transcripts

4. **Minutes Generation**

   * LLM-based summarization (T5 / GPT-4)
   * Planned: structured action-item extraction

---

## **Installation**

```bash
git clone https://github.com/yourusername/meeting-minutes-generator.git
cd meeting-minutes-generator
pip install -r requirements.txt
```

**Dependencies include:** Python 3.10+, librosa, noisereduce, pyannote.audio, transformers, openai, pandas, numpy

---



## **Next Steps**

* Implement **action-item extraction**
* Integrate a **dashboard** to visualize structured output
* Improve summarization quality and speaker diarization accuracy

---

## **Notes**

This is a **development version**. Results are preliminary, and some components are placeholders.

---

If you want, I can **also make a shorter “LinkedIn-ready version”** highlighting your progress and skills for sharing while the project is still in progress. Do you want me to do that?
