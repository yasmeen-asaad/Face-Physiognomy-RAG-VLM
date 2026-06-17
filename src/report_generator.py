"""
=============================================================
  Face Physiognomy Project — Report Generator
=============================================================

WHAT THIS FILE DOES:
  Takes all evidence collected from:
    - FaceDescriber  (visual observations per region)
    - RAGRetriever   (book passages per region)
  Combines them into one structured prompt
  Sends to Gemini and gets back a final physiognomy report

THIS IS THE LAST STEP IN THE PIPELINE:

  Part 1: FaceDetectorValidator    → face_crop + landmarks
  Part 2: FacePartExtractor        → region crops
  Part 3: FaceDescriber            → visual JSON per region
  Part 4: RAGRetriever             → book passages per region
  Part 5: ReportGenerator  ← HERE → final personality report

Install:
  pip install google-generativeai
"""

import json
import time
from dataclasses import dataclass
from typing import Dict, Optional

import google.generativeai as genai


# =============================================================
#  Result Data Class
# =============================================================

@dataclass
class FinalReport:
    """
    The complete physiognomy reading for one person.

    Fields:
        report_text  : the full written report from the LLM
        evidence_used: the structured evidence that was passed in
        success      : True if generation succeeded
        error        : error message if something went wrong
        tokens_used  : API token count
    """
    report_text   : str
    evidence_used : Dict
    success       : bool
    error         : Optional[str] = None
    tokens_used   : int           = 0


# =============================================================
#  Evidence Formatter
# =============================================================

def format_evidence(all_evidence: Dict) -> str:
    """
    Convert the evidence dict into a clean text block for the prompt.

    WHY format it carefully?
      The LLM reads this as context. The clearer and more structured
      it is, the better the report will be.
      Garbage in → garbage out.

    Input structure:
      {
        "nose": {
          "query"  : "wide nose with prominent ridge",
          "region" : "nose",
          "results": [
            {"page": 29, "score": 0.82, "content": "A wide nose..."},
            ...
          ]
        },
        ...
      }

    Output (text block):
      ═══ NOSE ═══
      Visual observation: wide nose with prominent ridge and rounded tip

      Book evidence (page 29, score: 0.82):
      A wide nose indicates generosity and...

      Book evidence (page 30, score: 0.74):
      The nose ridge, when prominent...

      ═══ EYES ═══
      ...
    """
    sections = []

    for region, evidence in all_evidence.items():
        region_title = region.upper().replace("_", " ")
        lines = [f"{'═'*3} {region_title} {'═'*3}"]

        # Visual observation (the query = natural language summary)
        lines.append(f"Visual observation: {evidence.get('query', 'N/A')}")
        lines.append("")

        # Book passages
        results = evidence.get("results", [])
        if results:
            for r in results:
                lines.append(
                    f"Book evidence (page {r['page']}, "
                    f"relevance: {r['score']:.0%}):"
                )
                lines.append(r["content"])
                lines.append("")
        else:
            lines.append("No relevant book passages found for this region.")
            lines.append("")

        sections.append("\n".join(lines))

    return "\n".join(sections)


# =============================================================
#  Report Generator Class
# =============================================================

class ReportGenerator:
    """
    Generates the final physiognomy reading report.

    Combines visual evidence + book knowledge into one
    coherent personality analysis using Gemini.

    Usage:
        generator = ReportGenerator()

        report = generator.generate(
            all_evidence = retriever.search_all_parts(descriptions),
        )

        if report.success:
            print(report.report_text)
    """

    

    # Base prompt — tells the LLM its role and rules
    # This is the prompt you designed — clean and focused.
    BASE_PROMPT = """You are a face reading report writer based on physiognomy.

Below are visual observations extracted from the face analysis,
and supporting excerpts retrieved from the physiognomy book.

Write a structured personality report based ONLY on the evidence provided.

STRICT RULES:
1. Do NOT invent information not present in the evidence.
2. Use ONLY the supplied visual observations and book excerpts.
3. If evidence for a region is weak or missing, say so briefly.
4. Write in clear, professional English.
5. Structure the report by facial region, then add an overall summary.
6. Do not repeat the raw evidence — synthesize it into insights.

EVIDENCE:
{evidence}

Write the physiognomy report now:"""

    def __init__(self, api_key: str = None, model_name="gemini-1.5-flash"):
        """
        Initialize Gemini client.
        Reads GEMINI_API_KEY from Kaggle secrets if not provided.
        """
        self.model_name = model_name#"gemini-1.5-flash"
        key = api_key or self._get_api_key()
        if not key:
            raise ValueError(
                "No Gemini API key found.\n"
                "Set GEMINI_API_KEY in Kaggle Secrets."
            )

        genai.configure(api_key=key)
        self.model = genai.GenerativeModel(self.model_name)
        print(f"ReportGenerator ready — model: {self.model_name}")

    def _get_api_key(self) -> Optional[str]:
        import os
        key = os.environ.get("GEMINI_API_KEY")
        if key:
            return key
        try:
            from kaggle_secrets import UserSecretsClient
            return UserSecretsClient().get_secret("GEMINI_API_KEY")
        except Exception:
            pass
        return None

    # ----------------------------------------------------------
    #  Main Method
    # ----------------------------------------------------------

    def generate(self, all_evidence: Dict) -> FinalReport:
        """
        Generate the final report from all collected evidence.

        HOW IT WORKS:
          1. Format evidence into readable text block
          2. Insert into prompt template
          3. Send to Gemini
          4. Return FinalReport

        Args:
            all_evidence : output of RAGRetriever.search_all_parts()
                           {region: {"query": ..., "results": [...]}}

        Returns:
            FinalReport with .report_text if successful
        """
        # Step 1: Format evidence into readable text
        evidence_text = format_evidence(all_evidence)

        # Step 2: Build full prompt
        prompt = self.BASE_PROMPT.format(evidence=evidence_text)

        # Step 3: Call Gemini
        try:
            response    = self.model.generate_content(prompt)
            report_text = response.text
            tokens_used = response.usage_metadata.total_token_count \
                          if hasattr(response, "usage_metadata") else 0

            return FinalReport(
                report_text   = report_text,
                evidence_used = all_evidence,
                success       = True,
                tokens_used   = tokens_used,
            )

        except Exception as e:
            return FinalReport(
                report_text   = "",
                evidence_used = all_evidence,
                success       = False,
                error         = str(e),
            )

    # ----------------------------------------------------------
    #  Save Report
    # ----------------------------------------------------------

    def save(self, report: FinalReport, output_path: str = "report.txt"):
        """
        Save the final report to a text file.

        Args:
            report      : FinalReport object
            output_path : where to save
        """
        if not report.success:
            print(f"Report failed — nothing to save: {report.error}")
            return

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report.report_text)

        print(f"Report saved → {output_path}")


# =============================================================
#  Kaggle Usage — Full Pipeline in One Place
# =============================================================
#
# from report_generator import ReportGenerator
# from rag_retriever     import PhysiognomyRetriever
#
# # Load retriever
# retriever = PhysiognomyRetriever(
#     index_path  = "/kaggle/input/rag-index/index.faiss",
#     chunks_path = "/kaggle/input/rag-index/chunks.pkl",
# )
#
# # Search all parts
# all_evidence = retriever.search_all_parts(descriptions, top_k=3)
#
# # Generate report
# generator = ReportGenerator()
# report    = generator.generate(all_evidence)
#
# if report.success:
#     print(report.report_text)
#     generator.save(report, "/kaggle/working/physiognomy_report.txt")
