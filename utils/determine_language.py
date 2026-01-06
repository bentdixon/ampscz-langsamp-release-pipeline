from typing import cast
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline
from utils.transcripts import Transcript
from data.langs import Language

nlp = Pipeline(lang="multilingual", processors="langid", use_gpu=True)

def determine_language(transcript: Transcript) -> Language:
    if len(transcript.lines) > 450:  # prevents strange cuDNN error
        text = "\n".join(line.text for line in transcript.lines[:450])
        print("Reducing text size...")
    else:
        text = "\n".join(line.text for line in transcript.lines)

    if len(text) == 0:
        print(f"Transcript has length of 0: {transcript.filename}, returning LANGUAGE.UNKNOWN")
        return Language.UNKNOWN

    doc = cast(Document, nlp(text))
    try:
        if doc.lang:
            if doc.lang in ["yue", "zh", "zh-hans", "zh-hant"]:  # Part of temporary fix - returns "Chinese" for all possible languages under Chinese umbrella
                return Language.cn
            return Language[doc.lang]
        else:
            print(f"Language for {transcript.patient_id} could not be identified")
            return Language.UNKNOWN
    except KeyError as e:
        print(f"Error {e}")
        return Language.UNKNOWN

