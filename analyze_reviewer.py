# step 1: run egu_harvester.py to download the PDFs and text files into the egusphere_harvest directory
# step 2: pip install pandas openpyxl pypdf
# step 3: python analyze_reviewers.py
import re
import json
import hashlib
from pathlib import Path

import pandas as pd
from pypdf import PdfReader


# ============================================================
# 1. CONFIGURATION
# ============================================================

BASE_DIR = Path("egusphere_harvest")
PDF_DIR = BASE_DIR / "pdfs"
TEXT_DIR = BASE_DIR / "text"

OUTPUT_DIR = BASE_DIR / "analysis"
EXTRACTED_DIR = OUTPUT_DIR / "extracted_text"

for directory in [OUTPUT_DIR, EXTRACTED_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

OUTPUT_JSON = OUTPUT_DIR / "reviewer_analysis.json"
OUTPUT_CSV = OUTPUT_DIR / "reviewer_analysis.csv"
OUTPUT_EXCEL = OUTPUT_DIR / "reviewer_analysis.xlsx"


# ============================================================
# 2. LINGUISTIC INDICATORS
# ============================================================

POSITIVE_TERMS = {
    "excellent", "good", "great", "interesting",
    "important", "valuable", "useful", "clear",
    "strong", "innovative", "impressive",
    "appreciate", "appreciated", "thank",
    "thanks", "helpful", "convincing",
    "robust", "relevant", "satisfactory",
    "appropriate"
}

NEGATIVE_TERMS = {
    "poor", "weak", "unclear", "incorrect",
    "inadequate", "insufficient", "wrong",
    "problematic", "unconvincing", "flawed",
    "misleading", "confusing", "disappointing",
    "unacceptable", "inappropriate",
    "inconsistent", "unjustified",
    "unsupported", "missing", "lack",
    "lacks", "failed", "failure"
}

CONSTRUCTIVE_TERMS = {
    "suggest", "suggestion", "recommend",
    "recommendation", "consider",
    "clarify", "clarification", "explain",
    "explanation", "provide", "include", "add",
    "revise", "revision", "improve",
    "improvement", "expand", "elaborate",
    "discuss", "address", "specify",
    "compare", "comparison", "correct",
    "correction", "justify", "justification",
    "reorganize", "rewrite", "please",
    "could", "should", "would", "propose",
    "resolve"
}

DISCOURTEOUS_TERMS = {
    "ridiculous", "absurd", "incompetent",
    "nonsense", "stupid", "careless",
    "unprofessional", "lazy", "terrible"
}

GENDER_PATTERNS = {
    "woman": [
        r"\bi identify as a woman\b",
        r"\bi am a woman\b",
        r"\bi'm a woman\b"
    ],
    "man": [
        r"\bi identify as a man\b",
        r"\bi am a man\b",
        r"\bi'm a man\b"
    ],
    "nonbinary": [
        r"\bi identify as nonbinary\b",
        r"\bi identify as non-binary\b",
        r"\bi am nonbinary\b",
        r"\bi am non-binary\b"
    ],
    "self_described_other": [
        r"\bi identify my gender as\b"
    ]
}


# ============================================================
# 3. FILE UTILITIES
# ============================================================

def sha256_file(path):
    digest = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(
            lambda: file.read(1024 * 1024), b""
        ):
            digest.update(chunk)

    return digest.hexdigest()


def clean_text(text):
    text = text.replace("\x00", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ============================================================
# 4. PDF EXTRACTION
# ============================================================

def extract_pdf_text(path):
    try:
        reader = PdfReader(str(path))
        pages = []

        for page_number, page in enumerate(
            reader.pages, start=1
        ):
            page_text = page.extract_text() or ""
            pages.append(
                f"\n[PAGE {page_number}]\n{page_text}"
            )

        return clean_text("\n".join(pages)), "success"

    except Exception as exc:
        print(f"PDF extraction failed: {path}: {exc}")
        return "", f"failed: {exc}"


# ============================================================
# 5. TEXT FILE EXTRACTION
# ============================================================

def extract_txt_text(path):
    for encoding in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(path, "r", encoding=encoding) as file:
                return clean_text(file.read()), "success"

        except UnicodeDecodeError:
            continue

    return "", "failed: unable to decode text"


# ============================================================
# 6. IDENTIFY REVIEWER
# ============================================================

def identify_reviewer(path):
    match = re.search(
        r"(RC\d+|AC\d+|CC\d+|EC\d+|CEC\d+)",
        path.stem,
        re.I
    )

    return match.group(1).upper() if match else path.stem


# ============================================================
# 7. SPLIT PARAGRAPHS
# ============================================================

def split_paragraphs(text):
    paragraphs = re.split(r"\n\s*\n", text)

    return [
        p.strip()
        for p in paragraphs
        if len(p.strip()) > 30
    ]


# ============================================================
# 8. SENTIMENT ANALYSIS
# ============================================================

def count_terms(text, vocabulary):
    words = re.findall(
        r"\b[\w'-]+\b",
        text.lower()
    )

    return sum(1 for word in words if word in vocabulary)


def calculate_sentiment(text):
    positive = count_terms(text, POSITIVE_TERMS)
    negative = count_terms(text, NEGATIVE_TERMS)
    discourteous = count_terms(text, DISCOURTEOUS_TERMS)

    total = positive + negative

    if total == 0:
        sentiment = "neutral"
    elif positive > negative:
        sentiment = "positive"
    elif negative > positive:
        sentiment = "negative"
    else:
        sentiment = "mixed"

    score = (
        (positive - negative) / total
        if total else 0
    )

    return {
        "positive_count": positive,
        "negative_count": negative,
        "discourteous_count": discourteous,
        "sentiment_score": round(score, 3),
        "sentiment": sentiment
    }


# ============================================================
# 9. CONSTRUCTIVENESS ANALYSIS
# ============================================================

def calculate_constructiveness(text):
    suggestions = count_terms(
        text, CONSTRUCTIVE_TERMS
    )

    specific_references = len(re.findall(
        r"\b(figure|table|section|paragraph|line|"
        r"equation|page|reference|citation|"
        r"supplement|appendix)\s*\d*",
        text,
        re.I
    ))

    questions = text.count("?")

    actionable_phrases = len(re.findall(
        r"\b(please clarify|it would be helpful|"
        r"i recommend|i suggest|the authors should|"
        r"the authors could|please provide|"
        r"would benefit from|should be addressed|"
        r"needs to be clarified)\b",
        text,
        re.I
    ))

    raw_score = (
        suggestions
        + specific_references * 2
        + questions
        + actionable_phrases * 3
    )

    word_count = max(len(text.split()), 1)

    score = min(
        100,
        round(raw_score / word_count * 100, 2)
    )

    if score >= 8:
        classification = "high"
    elif score >= 3:
        classification = "moderate"
    else:
        classification = "low"

    return {
        "constructive_term_count": suggestions,
        "specific_reference_count": specific_references,
        "question_count": questions,
        "actionable_phrase_count": actionable_phrases,
        "constructiveness_score": score,
        "constructiveness": classification
    }


# ============================================================
# 10. GENDER ANALYSIS
# ============================================================

def detect_self_disclosed_gender(text):
    detected = []

    for gender, patterns in GENDER_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.I):
                detected.append(gender)
                break

    detected = sorted(set(detected))

    if len(detected) == 1:
        return {
            "gender": detected[0],
            "gender_source": "explicit self-disclosure"
        }

    if len(detected) > 1:
        return {
            "gender": "multiple or ambiguous disclosures",
            "gender_source": "explicit self-disclosure"
        }

    return {
        "gender": "not disclosed",
        "gender_source": "not available"
    }


# ============================================================
# 11. OVERALL TONE
# ============================================================

def determine_tone(sentiment, constructiveness):
    if sentiment["discourteous_count"] >= 2:
        return "potentially discourteous"

    if (
        sentiment["sentiment"] == "positive"
        and constructiveness["constructiveness"]
        in ["high", "moderate"]
    ):
        return "supportive and constructive"

    if (
        sentiment["sentiment"] == "negative"
        and constructiveness["constructiveness"]
        in ["high", "moderate"]
    ):
        return "critical but constructive"

    if sentiment["sentiment"] == "negative":
        return "predominantly critical"

    if sentiment["sentiment"] == "positive":
        return "predominantly positive"

    return "neutral or mixed"


# ============================================================
# 12. ANALYZE INDIVIDUAL REVIEWER
# ============================================================

def analyze_reviewer(path):
    print(f"Analyzing: {path.name}")

    if path.suffix.lower() == ".pdf":
        text, status = extract_pdf_text(path)

    elif path.suffix.lower() == ".txt":
        text, status = extract_txt_text(path)

    else:
        return None

    reviewer_id = identify_reviewer(path)

    extracted_path = EXTRACTED_DIR / f"{path.stem}.txt"

    extracted_path.write_text(
        text,
        encoding="utf-8"
    )

    if not text.strip():
        return {
            "reviewer_id": reviewer_id,
            "filename": path.name,
            "extraction_status": status,
            "word_count": 0,
            "gender": "not disclosed",
            "overall_tone": "not assessed"
        }

    paragraphs = split_paragraphs(text)

    sentiment = calculate_sentiment(text)
    constructiveness = calculate_constructiveness(text)
    gender = detect_self_disclosed_gender(text)

    tone = determine_tone(
        sentiment,
        constructiveness
    )

    paragraph_results = []

    for index, paragraph in enumerate(
        paragraphs, start=1
    ):
        p_sentiment = calculate_sentiment(paragraph)

        p_constructiveness = calculate_constructiveness(
            paragraph
        )

        paragraph_results.append({
            "reviewer_id": reviewer_id,
            "paragraph_number": index,
            "paragraph_text": paragraph,
            **p_sentiment,
            **p_constructiveness
        })

    return {
        "reviewer_id": reviewer_id,
        "filename": path.name,
        "sha256": sha256_file(path),
        "extraction_status": status,
        "word_count": len(text.split()),
        "paragraph_count": len(paragraphs),
        **sentiment,
        **constructiveness,
        **gender,
        "overall_tone": tone,
        "extracted_text": str(extracted_path),
        "paragraph_analysis": paragraph_results
    }


# ============================================================
# 13. COLLECT FILES
# ============================================================

def collect_files():
    files = []

    if PDF_DIR.exists():
        files.extend(PDF_DIR.glob("*.pdf"))

    if TEXT_DIR.exists():
        files.extend(TEXT_DIR.glob("*.txt"))

    return sorted(files)


def process_all_reviewers():
    files = collect_files()

    print("=" * 60)
    print("REVIEWER LANGUAGE ANALYSIS")
    print("=" * 60)

    print(f"Files discovered: {len(files)}")

    results = []

    for path in files:
        result = analyze_reviewer(path)

        if result is not None:
            results.append(result)

    return results


# ============================================================
# 14. EXPORT RESULTS
# ============================================================

def export_results(results):
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    summary_rows = []
    paragraph_rows = []

    for reviewer in results:
        summary_rows.append({
            key: value
            for key, value in reviewer.items()
            if key != "paragraph_analysis"
        })

        paragraph_rows.extend(
            reviewer.get("paragraph_analysis", [])
        )

    summary_df = pd.DataFrame(summary_rows)
    paragraph_df = pd.DataFrame(paragraph_rows)

    summary_df.to_csv(
        OUTPUT_CSV,
        index=False,
        encoding="utf-8-sig"
    )

    with pd.ExcelWriter(
        OUTPUT_EXCEL,
        engine="openpyxl"
    ) as writer:

        summary_df.to_excel(
            writer,
            sheet_name="Reviewer Summary",
            index=False
        )

        paragraph_df.to_excel(
            writer,
            sheet_name="Paragraph Analysis",
            index=False
        )

        if not summary_df.empty:

            indicators = summary_df[[
                "reviewer_id",
                "positive_count",
                "negative_count",
                "discourteous_count",
                "constructive_term_count",
                "specific_reference_count",
                "actionable_phrase_count",
                "constructiveness_score",
                "sentiment_score"
            ]]

            indicators.to_excel(
                writer,
                sheet_name="Indicators",
                index=False
            )

            summary_df[[
                "reviewer_id",
                "gender",
                "gender_source"
            ]].to_excel(
                writer,
                sheet_name="Gender",
                index=False
            )

    print("\nResults exported successfully.")


# ============================================================
# 15. MAIN
# ============================================================

def main():
    results = process_all_reviewers()

    if not results:
        print("\nNo reviewer files found.")
        return

    export_results(results)

    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)

    for reviewer in results:
        print(f"\n{reviewer['reviewer_id']}")

        print(
            "Sentiment:",
            reviewer.get("sentiment", "N/A")
        )

        print(
            "Sentiment score:",
            reviewer.get("sentiment_score", "N/A")
        )

        print(
            "Constructiveness:",
            reviewer.get("constructiveness", "N/A")
        )

        print(
            "Constructiveness score:",
            reviewer.get("constructiveness_score", "N/A")
        )

        print(
            "Gender:",
            reviewer.get("gender", "N/A")
        )

        print(
            "Tone:",
            reviewer.get("overall_tone", "N/A")
        )

    print("\nOutput directory:")
    print(OUTPUT_DIR.resolve())

    print("\nCompleted.")


if __name__ == "__main__":
    main()
