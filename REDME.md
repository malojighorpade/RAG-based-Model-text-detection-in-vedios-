## Convert the mp4 into mp3
useing ffmpg model convert the format
## convert audio into text(chunks)
using the openAI whisper crete the json file where all text of file is saves model=large-v2

## read the chunks and and create embedding of chunks
do the embedding of all chunks and stored in embeding.joblib model=bge-m3

## take input as text and do the embeding on it
use cosine simalrity to mathch the text
## crete the prompt for ollama 
use lamma for output generation

