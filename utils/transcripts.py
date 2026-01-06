import os
import re
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import NamedTuple
from data.langs import Language, SITE_CODE_TO_LANGUAGES

_TIMESTAMP_PATTERN = re.compile(r'(\d{2}:\d{2}:\d{2}\.\d{3})')


class ClinicalGroup(Enum):
    CHR = 'CHR'
    HC = 'HC'
    UNKNOWN = 'UNKNOWN'


class TranscriptLine(NamedTuple):
    line_index: int
    speaker: str
    timestamp: str
    text: str


class Transcript:
    directory_path: Path | None = None
    _warning_shown: bool = False

    @classmethod
    def set_directory_path(cls, path: str | Path) -> None:
        cls.directory_path = Path(path)

    @classmethod
    def list_transcripts(cls, text_type: str | None = None) -> list:
        transcripts: list = []

        if cls.directory_path is None:
            raise ValueError("directory_path is not set, call Transcript.set_directory_path() first and point towards the transcript directory.")
        else:
            for transcript in list(cls.directory_path.rglob('*.txt')):
                transcripts.append(Transcript(transcript))

        return transcripts

    def __init__(self, filename: str | Path, language: Language | None = None):
        if Transcript.directory_path is None:
            if not Transcript._warning_shown:
                print("directory_path is not set, call Transcript.set_directory_path() first and point towards the transcript directory. (This warning will only display once.)")
                Transcript._warning_shown = True
            self.filename: str | Path = filename
            self.full_path: Path = Path(filename) # identical
            self.group_status: ClinicalGroup = self._get_clinical_status()
            self.patient_id = self._get_id()
            self.lines = self._get_text()
            self.site = self._get_site()
            self.language = self._get_language()
            self.session = self._get_session()
            self.day = self._get_day()
            self.transcript_type = self._get_transcript_type()
        else:
            self.filename: str | Path = filename
            self.full_path: Path = Transcript.directory_path / filename
            self.group_status: ClinicalGroup = self._get_clinical_status()
            self.patient_id = self._get_id()
            self.lines = self._get_text()
            self.site = self._get_site()
            self.language = self._get_language()
            self.session = self._get_session()
            self.day = self._get_day()
            self.transcript_type = self._get_transcript_type()

    @cached_property
    def interviewer_lines(self) -> list[TranscriptLine]:
        return [line for line in self.lines if line.speaker == "INTERVIEWER"]

    @cached_property
    def participant_lines(self) -> list[TranscriptLine]:
        return [line for line in self.lines if line.speaker == "PARTICIPANT"]

    def _get_site(self) -> str | None:
        name: str = Path(self.filename).name
        network_site: str | None = name.split("_")[0]
        sites = SITE_CODE_TO_LANGUAGES.keys()
        for site in sites:
            if site in network_site:
                return site
        return None

    def _get_clinical_status(self) -> ClinicalGroup:
        for parent in self.full_path.parents:
            if 'CHR' in parent.name.upper():
                return ClinicalGroup.CHR
            elif 'HC' in parent.name.upper():
                return ClinicalGroup.HC
        # print(f"ClinicalGroup not found for {self.full_path}, defaulting to UNKNOWN")
        return ClinicalGroup.UNKNOWN

    def _get_id(self) -> str | None:
        try:
            patient_id = os.path.basename(self.full_path).split("_")[1]
            return patient_id
        except IndexError as e:
            print(f"Error: {e}\nTranscript: {self.filename}")
            return None

    def _get_text(self) -> list[TranscriptLine]:
        with open(self.full_path, 'r', encoding='utf-8') as f:
            content = f.read()
        index: int = 0
        lines = []
        for line in content.split('\n'):
            line = line.strip()
            if line == "":
                continue
            index += 1
            if line.startswith('PARTICIPANT'):
                speaker = "PARTICIPANT"
            elif line.startswith('INTERVIEWER'):
                speaker = "INTERVIEWER"
            else:
                speaker = "UNKNOWN"
            match = _TIMESTAMP_PATTERN.search(line)
            if match:
                timestamp = match.group(1)
                text = line[match.end():].strip()
            else:
                timestamp = ""
                text = line.split(':', 1)[-1].strip()
            lines.append(TranscriptLine(index, speaker, timestamp, text))
        return lines

    def _get_session(self) -> str | None:
        try:
            session = os.path.basename(self.full_path).split("_")[5]
            return session
        except IndexError as e:
            print(f"Error: {e}\nTranscript: {self.filename}")
            return None

    def _get_day(self) -> str | None:
        try:
            day = os.path.basename(self.full_path).split("_")[4]
            return day
        except IndexError as e:
            print(f"Error: {e}\nTranscript: {self.filename}")
            return None

    def _get_transcript_type(self) -> str | None:
        try:
            transcript_type = os.path.basename(self.full_path).split("_")[3]
            return transcript_type
        except IndexError as e:
            print(f"Error: {e}\nTranscript: {self.filename}")
            return None

    def _get_language(self) -> Language:
        try:
            for parent in self.full_path.parents:
                for lang in Language:
                    if f"Language.{lang.name}" == parent.name:
                        return lang
            return Language.UNKNOWN
        except IndexError as e:
            print(f"Error: {e}\nTranscript: {self.filename}")
            return Language.UNKNOWN
