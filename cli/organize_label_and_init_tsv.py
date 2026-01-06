"""
Pipeline entrypoint: Organize, label, and initialize TSV for transcript processing.

This script:
1. Organizes transcripts by language and clinical status
2. Uses LLM to assign PARTICIPANT/INTERVIEWER roles to speaker labels (S1, S2, SP, SI, etc.)
3. Writes labeled transcripts to organized directory structure
4. Creates preliminary TSV with metadata and empty feature columns

Output TSV will have 2 rows per interview file (PARTICIPANT + INTERVIEWER) or 1 row per diary.
"""

import os
import sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set GPU before any CUDA imports
if '--gpu' in sys.argv:
    gpu_idx = sys.argv.index('--gpu')
    if gpu_idx + 1 < len(sys.argv):
        os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[gpu_idx + 1]

import re
import csv
import shutil
import argparse
from pathlib import Path
from rich.console import Console
from rich.tree import Tree
from rich.prompt import Confirm

from vllm import LLM, SamplingParams
from utils.transcripts import Transcript, ClinicalGroup
from utils.determine_language import determine_language
from data.langs import Language, SITE_CODE_TO_LANGUAGES

console = Console()


def load_clinical_status_csv(csv_path: Path) -> dict[str, ClinicalGroup]:
    """Load CSV and return mapping of patient_id to ClinicalGroup."""
    status_map: dict[str, ClinicalGroup] = {}

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        if 'patient_id' not in reader.fieldnames or 'clinical_status' not in reader.fieldnames:
            raise ValueError("CSV must contain 'patient_id' and 'clinical_status' columns")

        for row in reader:
            patient_id = row['patient_id'].strip()
            status_str = row['clinical_status'].strip().upper()

            if status_str == 'CHR':
                status_map[patient_id] = ClinicalGroup.CHR
            elif status_str == 'HC':
                status_map[patient_id] = ClinicalGroup.HC
            else:
                status_map[patient_id] = ClinicalGroup.UNKNOWN

    return status_map


def set_clinical_status(transcript: Transcript, status_map: dict[str, ClinicalGroup] | None = None) -> None:
    """Set clinical status from CSV if available, otherwise infer from path."""
    if status_map is not None and transcript.patient_id in status_map:
        transcript.group_status = status_map[transcript.patient_id]
        return

    if status_map is not None:
        console.print(f"[yellow]Warning:[/yellow] Patient {transcript.patient_id} not found in CSV, inferring from path.")

    for path in transcript.full_path.parents:
        if "CHR" in path.name.upper():
            transcript.group_status = ClinicalGroup.CHR
            return
        elif "HC" in path.name.upper():
            transcript.group_status = ClinicalGroup.HC
            return

    transcript.group_status = ClinicalGroup.UNKNOWN


def set_language(transcript: Transcript) -> None:
    """Determine and set language for transcript."""
    language = determine_language(transcript)

    if transcript.site is None:
        console.print(f"[yellow]Warning:[/yellow] Transcript {transcript.filename} has no site code.")
        transcript.language = language
    else:
        langs = SITE_CODE_TO_LANGUAGES.get(transcript.site, (Language.UNKNOWN,))
        if language in langs:
            transcript.language = language
        else:
            console.print(f"[yellow]Warning:[/yellow] Language {language} not in site {transcript.site} languages {langs}")
            transcript.language = Language.UNKNOWN


def is_diary(filepath: Path) -> bool:
    """Check if a file is a diary/audio journal based on filename."""
    return "audioJournal" in str(filepath) or "diary" in str(filepath).lower()


def normalize_speaker_labels(content: str) -> tuple[str, dict[str, str]]:
    """
    Normalize speaker labels to S1, S2, S3 format for LLM processing.

    Returns:
        Tuple of (normalized_content, label_mapping)
        label_mapping maps normalized labels (S1, S2) back to original labels (SI, SP)
    """
    # Find all unique speaker labels at start of lines
    speaker_pattern = r'^(S[IP\d]+)(?::|(?=\s))'
    speakers = set(re.findall(speaker_pattern, content, re.MULTILINE))

    # If all speakers are already in S1, S2, S3 format, no normalization needed
    if all(re.match(r'^S[123]$', s) for s in speakers):
        return content, {}

    # Create mapping: SI -> S1, SP -> S2, or other speakers to S1, S2, S3
    mapping = {}
    reverse_mapping = {}
    normalized_labels = ['S1', 'S2', 'S3']

    # Sort speakers for consistent mapping
    sorted_speakers = sorted(speakers)

    for i, original in enumerate(sorted_speakers):
        if i < len(normalized_labels):
            normalized = normalized_labels[i]
            mapping[original] = normalized
            reverse_mapping[normalized] = original

    # Replace speaker labels in content
    normalized_content = content
    for original, normalized in mapping.items():
        normalized_content = re.sub(
            rf'^{re.escape(original)}(?::|(?=\s))',
            f'{normalized}:',
            normalized_content,
            flags=re.MULTILINE
        )

    return normalized_content, reverse_mapping


def load_llm_model(
    model_name: str = "openai/gpt-oss-120b",
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
) -> LLM:
    """Load LLM model using vLLM."""
    console.print(f"Loading LLM model: {model_name}...")

    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=80000,
        dtype="auto",
        trust_remote_code=True,
    )

    console.print(f"[green]Model {model_name} loaded successfully[/green]")
    return llm


def build_llm_messages(transcript_content: str) -> list[dict[str, str]]:
    """Build messages for LLM speaker role classification."""
    system_prompt = "You are a trained clinical annotator. Only respond in the format prescribed. Keep your reasoning succinct."

    user_prompt = f"""Analyze the following transcript excerpt and determine which speaker is the INTERVIEWER and which is the PARTICIPANT.

The INTERVIEWER typically:
- Asks questions about the participant's experiences, thoughts, or feelings
- Guides the conversation with prompts or follow-up questions
- Uses phrases like "Can you tell me about...", "How did that make you feel?"

The PARTICIPANT typically:
- Responds to questions with personal experiences or opinions
- Provides longer narrative responses
- Shares information about themselves

Transcript:
{transcript_content}

Based on the conversation pattern, classify the speakers. If there are three speakers (S1, S2, S3), only label the INTERVIEWER and PARTICIPANT—leave the third speaker unlabeled.

Respond with exactly two lines:
<speaker>: INTERVIEWER
<speaker>: PARTICIPANT

For example:
S1: INTERVIEWER
S2: PARTICIPANT

Or if S1 is an unlabeled third party:
S2: INTERVIEWER
S3: PARTICIPANT"""

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_llm_roles(response: str) -> dict[str, str] | None:
    """
    Parse speaker roles from LLM response.

    Returns:
        Dictionary mapping speaker labels (S1/S2/S3) to roles, or None on failure.
    """
    roles = {}

    matches = re.findall(r"(S[123]):\s*(INTERVIEWER|PARTICIPANT)", response, re.IGNORECASE)

    for speaker, role in matches:
        roles[speaker] = role.upper()

    # Validate: must have exactly one INTERVIEWER and one PARTICIPANT
    role_values = list(roles.values())
    if role_values.count("INTERVIEWER") != 1 or role_values.count("PARTICIPANT") != 1:
        return None

    if len(roles) != 2:
        return None

    return roles


def classify_speaker_roles_batch(
    transcripts: list[Transcript],
    llm: LLM,
    sampling_params: SamplingParams,
    chars: int = 5000,
) -> list[tuple[Transcript, dict[str, str] | None, dict[str, str]]]:
    """
    Classify speaker roles for a batch of transcripts using LLM.

    Returns:
        List of (transcript, roles, label_mapping) tuples
        roles maps normalized labels (S1, S2) to roles (INTERVIEWER, PARTICIPANT)
        label_mapping maps normalized labels back to original labels (SI, SP)
    """
    all_messages = []
    all_mappings = []

    for transcript in transcripts:
        with open(transcript.full_path, "r", encoding="utf-8") as f:
            content = f.read()[:chars]

        normalized_content, label_mapping = normalize_speaker_labels(content)
        messages = build_llm_messages(normalized_content)
        all_messages.append(messages)
        all_mappings.append(label_mapping)

    outputs = llm.chat(
        messages=all_messages,
        sampling_params=sampling_params,
        use_tqdm=True,
    )

    results = []
    for transcript, output, label_mapping in zip(transcripts, outputs, all_mappings):
        response = output.outputs[0].text.strip()
        console.print(f"\n{transcript.filename} →\n{response}\n")

        roles = parse_llm_roles(response)

        if roles is None:
            console.print(f"[red]Failed to parse roles for {transcript.filename}[/red]")

        results.append((transcript, roles, label_mapping))

    return results


def get_original_speaker_label(filepath: Path) -> dict[str, str]:
    """
    Get original speaker labels from transcript file.
    Returns dict mapping 'PARTICIPANT' and/or 'INTERVIEWER' to original labels.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    labels = {}

    # Find first occurrence of PARTICIPANT
    match = re.search(r'^PARTICIPANT(?::|(?=\s))', content, re.MULTILINE)
    if match:
        labels['PARTICIPANT'] = 'PARTICIPANT'

    # Find first occurrence of INTERVIEWER
    match = re.search(r'^INTERVIEWER(?::|(?=\s))', content, re.MULTILINE)
    if match:
        labels['INTERVIEWER'] = 'INTERVIEWER'

    # If not labeled yet, find S# labels
    if not labels:
        speaker_pattern = r'^(S[IP\d]+)(?::|(?=\s))'
        speakers = re.findall(speaker_pattern, content, re.MULTILINE)
        if speakers:
            # For now, just store the first speaker as PARTICIPANT
            labels['PARTICIPANT'] = speakers[0]

    return labels


def write_labeled_transcript(
    transcript: Transcript,
    roles: dict[str, str] | None,
    label_mapping: dict[str, str],
    output_path: Path,
) -> None:
    """
    Write transcript with PARTICIPANT/INTERVIEWER labels.
    For diaries, keeps original labels intact.
    """
    with open(transcript.full_path, "r", encoding="utf-8") as f:
        content = f.read()

    if roles is not None and not is_diary(transcript.full_path):
        # Map normalized labels to roles
        for normalized_label, role in roles.items():
            original_label = label_mapping.get(normalized_label, normalized_label)
            content = re.sub(
                rf"^{re.escape(original_label)}(?::|(?=\s))",
                f"{role}:",
                content,
                flags=re.MULTILINE
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)


def build_feature_list(feature_file: Path) -> list[str]:
    """Load feature names from file."""
    features = []
    with open(feature_file, 'r') as f:
        for line in f:
            feature = line.strip()
            if feature:
                # Rename problematic feature names
                if feature == '1':
                    feature = 'p1'
                elif feature == '2':
                    feature = 'p2'
                elif feature == '3':
                    feature = 'p3'
                elif feature == 'Yes':
                    feature = 'pronoun_possession'
                features.append(feature)
    return features


def initialize_tsv(
    transcripts_with_roles: list[tuple[Transcript, dict[str, str] | None, dict[str, str], Path]],
    output_tsv: Path,
    feature_file: Path,
) -> None:
    """
    Create preliminary TSV with metadata and empty feature columns.

    Args:
        transcripts_with_roles: List of (transcript, roles, label_mapping, output_path) tuples
        output_tsv: Path to output TSV file
        feature_file: Path to feature list file
    """
    features = build_feature_list(feature_file)

    # Build header
    header = [
        'network', 'language', 'src_subject_id', 'interview_type',
        'day', 'interview_number', 'transcript_speaker_label', 'speaker_role'
    ] + features + ['num_sent', 'word_freq', 'file_name.txt']

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    with open(output_tsv, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)

        for transcript, roles, label_mapping, labeled_path in transcripts_with_roles:
            site = transcript.site or 'UNKNOWN'
            patient_id = transcript.patient_id or 'UNKNOWN'
            language = transcript.language.value if transcript.language else 'UNKNOWN'
            interview_type = transcript.transcript_type or 'UNKNOWN'
            day = transcript.day or 'UNKNOWN'
            session = transcript.session or 'UNKNOWN'
            filename = labeled_path.name

            # Get relative path from organized directory
            # (will be used for reading later)

            is_diary_file = is_diary(transcript.full_path)

            if is_diary_file:
                # Diaries: one row with original speaker label
                original_labels = get_original_speaker_label(transcript.full_path)
                transcript_speaker_label = original_labels.get('PARTICIPANT', 'S0')

                row = [
                    site, language, patient_id, interview_type,
                    day, session, transcript_speaker_label, 'Participant'
                ] + [''] * len(features) + ['', '', filename]
                writer.writerow(row)

            elif roles is not None:
                # Interviews: two rows (PARTICIPANT, INTERVIEWER)
                for normalized_label, role in roles.items():
                    # Get original label (SP, SI, S1, S2, etc.)
                    transcript_speaker_label = label_mapping.get(normalized_label, normalized_label)
                    speaker_role = 'Participant' if role == 'PARTICIPANT' else 'Interviewer'

                    row = [
                        site, language, patient_id, interview_type,
                        day, session, transcript_speaker_label, speaker_role
                    ] + [''] * len(features) + ['', '', filename]
                    writer.writerow(row)
            else:
                # Failed to classify: create rows with unknown labels
                console.print(f"[yellow]Warning: No roles for {filename}, creating placeholder rows[/yellow]")
                for role in ['Participant', 'Interviewer']:
                    row = [
                        site, language, patient_id, interview_type,
                        day, session, '', role
                    ] + [''] * len(features) + ['', '', filename]
                    writer.writerow(row)

    console.print(f"[green]Initialized TSV: {output_tsv}[/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Organize, label, and initialize TSV for transcript processing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--i", type=str, required=True, help="Input directory with unlabeled transcripts")
    parser.add_argument("--o", type=str, required=True, help="Output directory for organized/labeled transcripts")
    parser.add_argument("--tsv", type=str, required=True, help="Output TSV file path")
    parser.add_argument("--feats", type=str, required=True, help="Feature list file (tags_upos_xpos.txt)")
    parser.add_argument("--csv", type=str, required=False, help="CSV with patient_id and clinical_status columns")
    parser.add_argument("--text-type", type=str, required=True, help="Transcript text type (psychs, open, diaries)")
    parser.add_argument("--gpu", type=int, required=True, help="GPU device ID")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for LLM inference")
    parser.add_argument("--skip-labeling", action="store_true", help="Skip LLM labeling (use existing labels)")
    args = parser.parse_args()

    input_dir = Path(args.i)
    output_dir = Path(args.o)
    output_tsv = Path(args.tsv)
    feature_file = Path(args.feats)
    text_type = args.text_type

    if not input_dir.exists():
        console.print(f"[red]Error: Input directory {input_dir} does not exist[/red]")
        return

    if not feature_file.exists():
        console.print(f"[red]Error: Feature file {feature_file} does not exist[/red]")
        return

    # Load clinical status CSV if provided
    status_map = None
    if args.csv:
        csv_path = Path(args.csv)
        if csv_path.exists():
            status_map = load_clinical_status_csv(csv_path)
        else:
            console.print(f"[red]Error: CSV {csv_path} does not exist[/red]")
            return

    # Collect and process transcripts
    console.print(f"[bold]Scanning transcripts in {input_dir}[/bold]")
    txt_files = list(input_dir.rglob('*.txt'))
    console.print(f"Found {len(txt_files)} transcript files")

    transcripts = []
    for txt_file in txt_files:
        transcript = Transcript(txt_file)
        set_clinical_status(transcript, status_map)
        set_language(transcript)
        transcripts.append(transcript)

    # Separate diaries from interviews
    diaries = [t for t in transcripts if is_diary(t.full_path)]
    interviews = [t for t in transcripts if not is_diary(t.full_path)]

    console.print(f"\nCategorized:")
    console.print(f"  Interviews: {len(interviews)}")
    console.print(f"  Diaries: {len(diaries)}")

    # Load LLM for interviews (skip if --skip-labeling)
    interview_results = []
    if not args.skip_labeling and interviews:
        llm = load_llm_model(tensor_parallel_size=args.tp)
        sampling_params = SamplingParams(max_tokens=700, temperature=0.0)

        console.print(f"\n[bold]Classifying speaker roles for {len(interviews)} interviews[/bold]")

        # Process in batches
        for i in range(0, len(interviews), args.batch_size):
            batch = interviews[i:i + args.batch_size]
            console.print(f"Processing batch {i//args.batch_size + 1}/{(len(interviews) + args.batch_size - 1)//args.batch_size}")
            results = classify_speaker_roles_batch(batch, llm, sampling_params)
            interview_results.extend(results)
    else:
        # Skip labeling: use existing labels or None
        interview_results = [(t, None, {}) for t in interviews]

    # Write labeled transcripts to organized directory
    console.print(f"\n[bold]Writing transcripts to {output_dir}[/bold]")

    transcripts_with_roles = []

    # Process interviews
    for transcript, roles, label_mapping in interview_results:
        lang_name = transcript.language.name if transcript.language else 'UNKNOWN'
        group_name = transcript.group_status.name if transcript.group_status else 'UNKNOWN'

        output_path = output_dir / text_type / lang_name / group_name / transcript.full_path.name
        write_labeled_transcript(transcript, roles, label_mapping, output_path)
        transcripts_with_roles.append((transcript, roles, label_mapping, output_path))
        console.print(f"  Wrote: {output_path.relative_to(output_dir)}")

    # Process diaries (no labeling)
    for transcript in diaries:
        lang_name = transcript.language.name if transcript.language else 'UNKNOWN'
        group_name = transcript.group_status.name if transcript.group_status else 'UNKNOWN'

        output_path = output_dir / text_type / lang_name / group_name / transcript.full_path.name
        write_labeled_transcript(transcript, None, {}, output_path)  # Keep original labels
        transcripts_with_roles.append((transcript, None, {}, output_path))
        console.print(f"  Wrote: {output_path.relative_to(output_dir)}")

    # Initialize TSV
    console.print(f"\n[bold]Initializing TSV with metadata[/bold]")
    initialize_tsv(transcripts_with_roles, output_tsv, feature_file)

    console.print(f"\n[green]✓ Pipeline complete![/green]")
    console.print(f"  Organized transcripts: {output_dir}")
    console.print(f"  Preliminary TSV: {output_tsv}")
    console.print(f"\nNext step: Run tag_grammatical_feats.py to fill feature columns")


if __name__ == "__main__":
    main()
