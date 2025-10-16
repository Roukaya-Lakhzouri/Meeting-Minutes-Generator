#%%
import os
from pathlib import Path


# Path to the parent folder
parent_folder = "Data\\amicorpus"

# List all directories in the folder
audio_folders = [name for name in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, name))]

# Path to the transcreption  folder
annotation_folder_path = Path("Data\Annotations\words")
files = [f for f in os.listdir(annotation_folder_path) if os.path.isfile(os.path.join(annotation_folder_path, f))]
annotation_name_files=[f for f in files if f.split(".")[0] in audio_folders]


# %%
import xml.etree.ElementTree as ET
import glob
transcript = []
for name_file in annotation_name_files:
    xml_path = f"Data\Annotations\words\{name_file}"
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for elem in root.findall(".//w"):  # 'w' tags store words
        if elem.attrib.get('starttime')!= None: 
            word = elem.text
            start = float(elem.attrib.get('starttime'))
            end = float(elem.attrib.get('endtime'))
            speaker=xml_path.split('.')[1]
            audio_name=name_file.split(".")[0]
            transcript.append((word, start, end,speaker,audio_name))

from collections import defaultdict
def build_segments(words, silence_threshold=0.8):
    segments = []
    current = {"speaker": None, "start": None, "end": None, "words": []}
    punct = {".", ",", "?", "!", ";", ":"}

    for word, start, end, spk in sorted(words, key=lambda x: x[1]):
        if word in punct:
            if current["words"]:
                current["words"][-1] += word
            current["end"] = end
            continue

        if (
            spk != current["speaker"]
            or (current["end"] and start - current["end"] > silence_threshold)
        ):
            if current["words"]:
                current["text"] = " ".join(current["words"])
                segments.append(current)
            current = {"speaker": spk, "start": start, "end": end, "words": [word]}
        else:
            current["words"].append(word)
            current["end"] = end

    if current["words"]:
        current["text"] = " ".join(current["words"])
        segments.append(current)

    return segments

# Group by audio
audio_groups = defaultdict(list)
for word, start, end, spk, name in transcript:
    audio_groups[name].append((word, start, end, spk))
final_transcripts = {}

for name, words in audio_groups.items():
    words.sort(key=lambda x: x[1])  # sort by start time
    # apply your segmentation + punctuation + overlap logic
    segments = build_segments(words)
    final_transcripts[name] = segments
#%%
audiou=final_transcripts['ES2008b']

text=''
for dict in audiou:
    text+=f"{dict['speaker']} : {dict['text']} \n "
print(text)

# %%
