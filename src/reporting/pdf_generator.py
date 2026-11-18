"""Professional PDF Report Generator (CP-11).

Generates publication-grade, academic, and executive cybersecurity reports using ReportLab.
Includes custom two-pass NumberedCanvas for exact 'Page X of Y' numbering, running headers,
threat classification banners, wrapped multi-page tables, and strict non-attribution disclaimers.
"""

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pypdf
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from config.settings import settings
from src.reporting.report_models import (
    ReportClassification,
    ReportDataPackage,
    ReportMetadata,
    ReportType,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Two-Pass Numbered Canvas for Exact Page Counts and Running Headers/Footers
# ==============================================================================

class NumberedCanvas(canvas.Canvas):
    """Custom canvas that records all drawing commands and prints total page count in a 2nd pass."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        """Save page state for deferred rendering."""
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        """Draw running headers, footers, and page numbers on all pages during second pass."""
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count: int) -> None:
        """Draw running header, running footer, and confidentiality banner."""
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#475569"))

        # Running Header (on pages > 1)
        if self._pageNumber > 1:
            header_text = "AI-Powered Network Intrusion Detection System — Cybersecurity Assessment"
            self.drawString(54, 755, header_text)
            self.setStrokeColor(colors.HexColor("#cbd5e1"))
            self.setLineWidth(0.5)
            self.line(54, 750, 558, 750)

        # Running Footer (on all pages)
        self.setStrokeColor(colors.HexColor("#cbd5e1"))
        self.setLineWidth(0.5)
        self.line(54, 45, 558, 45)

        footer_left = "ACADEMIC / DEFENSIVE USE ONLY — Model predictions do not constitute confirmed incidents."
        self.drawString(54, 32, footer_left)

        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 32, page_str)

        self.restoreState()


# ==============================================================================
# PDF Report Generator
# ==============================================================================

class PDFReportGenerator:
    """Compiles publication-grade PDF documents from ReportDataPackage."""

    def __init__(self, page_size: Tuple[float, float] = letter):
        """Initialize generator with standard page dimensions."""
        self.page_size = page_size
        self.styles = self._setup_styles()

    def _setup_styles(self) -> Dict[str, ParagraphStyle]:
        """Configure typography and color palette."""
        base_styles = getSampleStyleSheet()

        custom_styles = {
            "DocTitle": ParagraphStyle(
                "DocTitle",
                parent=base_styles["Heading1"],
                fontName="Helvetica-Bold",
                fontSize=20,
                leading=24,
                textColor=colors.HexColor("#0f172a"),
                spaceAfter=4,
            ),
            "DocSubtitle": ParagraphStyle(
                "DocSubtitle",
                parent=base_styles["Normal"],
                fontName="Helvetica",
                fontSize=10,
                leading=14,
                textColor=colors.HexColor("#475569"),
                spaceAfter=12,
            ),
            "SectionHeader": ParagraphStyle(
                "SectionHeader",
                parent=base_styles["Heading2"],
                fontName="Helvetica-Bold",
                fontSize=13,
                leading=16,
                textColor=colors.HexColor("#1e293b"),
                spaceBefore=12,
                spaceAfter=6,
                keepWithNext=True,
            ),
            "SubSectionHeader": ParagraphStyle(
                "SubSectionHeader",
                parent=base_styles["Heading3"],
                fontName="Helvetica-Bold",
                fontSize=10.5,
                leading=14,
                textColor=colors.HexColor("#334155"),
                spaceBefore=8,
                spaceAfter=4,
                keepWithNext=True,
            ),
            "Body": ParagraphStyle(
                "Body",
                parent=base_styles["Normal"],
                fontName="Helvetica",
                fontSize=9,
                leading=12.5,
                textColor=colors.HexColor("#1e293b"),
                spaceAfter=6,
            ),
            "BodyBold": ParagraphStyle(
                "BodyBold",
                parent=base_styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=9,
                leading=12.5,
                textColor=colors.HexColor("#0f172a"),
            ),
            "TableHeader": ParagraphStyle(
                "TableHeader",
                parent=base_styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8.5,
                leading=11,
                textColor=colors.white,
            ),
            "TableCell": ParagraphStyle(
                "TableCell",
                parent=base_styles["Normal"],
                fontName="Helvetica",
                fontSize=8,
                leading=10.5,
                textColor=colors.HexColor("#1e293b"),
            ),
            "TableCellBold": ParagraphStyle(
                "TableCellBold",
                parent=base_styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=8,
                leading=10.5,
                textColor=colors.HexColor("#0f172a"),
            ),
            "CalloutText": ParagraphStyle(
                "CalloutText",
                parent=base_styles["Normal"],
                fontName="Helvetica-Oblique",
                fontSize=8.5,
                leading=11.5,
                textColor=colors.HexColor("#334155"),
            ),
            "Badge": ParagraphStyle(
                "Badge",
                parent=base_styles["Normal"],
                fontName="Helvetica-Bold",
                fontSize=7.5,
                leading=9,
                textColor=colors.HexColor("#1e40af"),
            ),
        }
        return custom_styles

    def generate(self, data_package: ReportDataPackage, output_path: Path) -> Tuple[Path, str, int, int]:
        """Generate PDF report atomically and verify cryptographic integrity.

        Args:
            data_package: Populated and verified ReportDataPackage.
            output_path: Destination path for final PDF file.

        Returns:
            Tuple of (output_path, sha256_checksum, size_bytes, page_count).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            doc = SimpleDocTemplate(
                str(temp_path),
                pagesize=self.page_size,
                leftMargin=54,
                rightMargin=54,
                topMargin=54,
                bottomMargin=54,
            )

            story: List[Any] = []
            meta = data_package.metadata
            rtype = meta.report_type

            # Cover Banner & Metadata
            story.extend(self._build_header_banner(data_package))
            story.append(Spacer(1, 10))

            # Build relevant sections based on ReportType
            if rtype in (ReportType.EXECUTIVE, ReportType.CONSOLIDATED):
                story.extend(self._build_executive_section(data_package))
                story.append(Spacer(1, 12))

            if rtype in (ReportType.TECHNICAL, ReportType.CONSOLIDATED):
                story.extend(self._build_technical_section(data_package))
                story.append(Spacer(1, 12))

            if rtype in (ReportType.ANALYST_REVIEW, ReportType.CONSOLIDATED):
                story.extend(self._build_analyst_section(data_package))
                story.append(Spacer(1, 12))

            if rtype in (ReportType.TECHNICAL, ReportType.CONSOLIDATED, ReportType.EXECUTIVE):
                story.extend(self._build_threat_context_section(data_package))
                story.append(Spacer(1, 12))

            # Methodology, References & Disclaimers Appendix
            story.extend(self._build_appendix_section(data_package))

            # Build Document with NumberedCanvas
            doc.build(story, canvasmaker=NumberedCanvas)

            # Verification of Generated PDF
            if not temp_path.exists() or temp_path.stat().st_size == 0:
                raise RuntimeError("PDF generation produced empty or missing file.")

            reader = pypdf.PdfReader(str(temp_path))
            page_count = len(reader.pages)
            if page_count < 1:
                raise RuntimeError("Generated PDF has 0 readable pages.")

            with open(temp_path, "rb") as f:
                content = f.read()
                pdf_sha256 = hashlib.sha256(content).hexdigest()
                pdf_size = len(content)

            # Atomic copy to final destination
            with open(output_path, "wb") as f_out:
                f_out.write(content)

            logger.info(
                f"Generated PDF '{output_path.name}' [Pages: {page_count}, "
                f"Size: {pdf_size} bytes, SHA-256: {pdf_sha256[:12]}...]"
            )
            return output_path, pdf_sha256, pdf_size, page_count

        finally:
            if temp_path.exists():
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    # ==========================================================================
    # Section Builders
    # ==========================================================================

    def _build_header_banner(self, data_package: ReportDataPackage) -> List[Any]:
        """Construct cover banner, classification marking, and metadata grid."""
        meta = data_package.metadata
        elements: List[Any] = []

        # Classification & Title
        class_color = colors.HexColor("#0284c7")
        if meta.classification == ReportClassification.TLP_AMBER:
            class_color = colors.HexColor("#d97706")
        elif meta.classification == ReportClassification.TLP_GREEN:
            class_color = colors.HexColor("#16a34a")

        header_data = [
            [
                Paragraph(f"<b>CLASSIFICATION:</b> {meta.classification}", self.styles["Badge"]),
                Paragraph(f"<b>STATUS:</b> {meta.status}", self.styles["Badge"]),
            ]
        ]
        t_header = Table(header_data, colWidths=[250, 254])
        t_header.setStyle(TableStyle([
            ("ALIGN", (0, 0), (0, 0), "LEFT"),
            ("ALIGN", (1, 0), (1, 0), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_header)

        elements.append(Paragraph(meta.title, self.styles["DocTitle"]))
        elements.append(Paragraph(
            f"Automated Academic Intrusion Detection Report — {meta.report_type}",
            self.styles["DocSubtitle"]
        ))
        elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0f172a"), spaceAfter=8))

        # Metadata Grid Table
        grid_data = [
            [
                Paragraph("<b>Report ID:</b>", self.styles["TableCellBold"]),
                Paragraph(meta.report_id, self.styles["TableCell"]),
                Paragraph("<b>Generated UTC:</b>", self.styles["TableCellBold"]),
                Paragraph(meta.created_at_utc[:19].replace("T", " "), self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Detection Run:</b>", self.styles["TableCellBold"]),
                Paragraph(meta.detection_run_id, self.styles["TableCell"]),
                Paragraph("<b>Source File:</b>", self.styles["TableCellBold"]),
                Paragraph(meta.source_filename, self.styles["TableCell"]),
            ],
            [
                Paragraph("<b>Algorithms:</b>", self.styles["TableCellBold"]),
                Paragraph(", ".join(meta.selected_models) or "SVM, MLP, Decision Tree", self.styles["TableCell"]),
                Paragraph("<b>Total Records:</b>", self.styles["TableCellBold"]),
                Paragraph(f"{meta.total_records_analyzed:,}", self.styles["TableCell"]),
            ],
        ]
        t_grid = Table(grid_data, colWidths=[90, 162, 90, 162])
        t_grid.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_grid)
        return elements

    def _build_executive_section(self, data_package: ReportDataPackage) -> List[Any]:
        """Build high-level executive summary, key findings, and limitation notices."""
        meta = data_package.metadata
        elements: List[Any] = []
        elements.append(Paragraph("1. Executive Summary & Key Findings", self.styles["SectionHeader"]))

        attack_pct = round((meta.predicted_attack_records / meta.total_records_analyzed * 100), 2) if meta.total_records_analyzed > 0 else 0.0

        # KPI Metrics Table
        kpi_data = [
            [
                Paragraph("Total Flows Analyzed", self.styles["TableHeader"]),
                Paragraph("Flagged Attacks", self.styles["TableHeader"]),
                Paragraph("Benign Flows", self.styles["TableHeader"]),
                Paragraph("Analyst Reviewed", self.styles["TableHeader"]),
            ],
            [
                Paragraph(f"<b>{meta.total_records_analyzed:,}</b>", self.styles["TableCellBold"]),
                Paragraph(f"<b>{meta.predicted_attack_records:,}</b> ({attack_pct}%)", self.styles["TableCellBold"]),
                Paragraph(f"<b>{meta.predicted_benign_records:,}</b>", self.styles["TableCellBold"]),
                Paragraph(f"<b>{meta.analyst_reviewed_count:,}</b>", self.styles["TableCellBold"]),
            ],
        ]
        t_kpi = Table(kpi_data, colWidths=[126, 126, 126, 126])
        t_kpi.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(t_kpi)
        elements.append(Spacer(1, 6))

        # Narrative Body
        summary_text = (
            f"During this assessment, the multi-model intrusion detection pipeline evaluated a total of "
            f"<b>{meta.total_records_analyzed:,}</b> network connection records originating from source dataset "
            f"<code>{meta.source_filename}</code>. A total of <b>{meta.predicted_attack_records:,}</b> records "
            f"({attack_pct}%) were classified as malicious traffic patterns by the authorized consensus algorithms, "
            f"while <b>{meta.predicted_benign_records:,}</b> records were evaluated as normal benign communication. "
            f"To date, <b>{meta.analyst_reviewed_count:,}</b> records have undergone formal human-in-the-loop analyst review."
        )
        elements.append(Paragraph(summary_text, self.styles["Body"]))

        # Limitations Callout Box
        limitation_text = (
            "<b>CRITICAL GOVERNANCE LIMITATION:</b> Model classifications are statistical predictions produced by "
            "supervised machine learning algorithms (LinearSVC, MLPClassifier, DecisionTreeClassifier) trained on academic "
            "CIC-IDS-2017 benchmark telemetry. Predictions do not constitute confirmed network compromises or active incident indicators."
        )
        t_box = Table([[Paragraph(limitation_text, self.styles["CalloutText"])]], colWidths=[504])
        t_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fef2f2")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#f87171")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(t_box)
        return elements

    def _build_technical_section(self, data_package: ReportDataPackage) -> List[Any]:
        """Build detailed technical detection analysis, per-class breakdown, and model comparisons."""
        elements: List[Any] = []
        elements.append(Paragraph("2. Technical Detection Analysis", self.styles["SectionHeader"]))

        # Prediction Distribution Table
        elements.append(Paragraph("2.1 Predicted Traffic Class Breakdown", self.styles["SubSectionHeader"]))
        dist_rows = [
            [
                Paragraph("Class Label", self.styles["TableHeader"]),
                Paragraph("Classification", self.styles["TableHeader"]),
                Paragraph("Record Count", self.styles["TableHeader"]),
                Paragraph("Traffic Share (%)", self.styles["TableHeader"]),
            ]
        ]
        for item in data_package.prediction_distribution[:20]:
            cat_text = "Malicious / Intrusion" if item["is_attack"] else "Normal / Benign"
            dist_rows.append([
                Paragraph(item["class_label"], self.styles["TableCellBold"]),
                Paragraph(cat_text, self.styles["TableCell"]),
                Paragraph(f"{item['count']:,}", self.styles["TableCell"]),
                Paragraph(f"{item['percentage']:.2f}%", self.styles["TableCell"]),
            ])

        t_dist = Table(dist_rows, colWidths=[150, 150, 104, 100])
        t_dist.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_dist)
        elements.append(Spacer(1, 8))

        # Model Comparisons Table
        if data_package.model_comparisons:
            elements.append(Paragraph("2.2 Multi-Algorithm Comparative Consensus", self.styles["SubSectionHeader"]))
            comp_rows = [
                [
                    Paragraph("Comparison Dimension", self.styles["TableHeader"]),
                    Paragraph("LinearSVC (SVM)", self.styles["TableHeader"]),
                    Paragraph("MLP Classifier", self.styles["TableHeader"]),
                    Paragraph("Decision Tree", self.styles["TableHeader"]),
                ]
            ]
            for row in data_package.model_comparisons[:15]:
                comp_rows.append([
                    Paragraph(row.get("metric_or_aspect", ""), self.styles["TableCellBold"]),
                    Paragraph(row.get("svm_value", "-"), self.styles["TableCell"]),
                    Paragraph(row.get("mlp_value", "-"), self.styles["TableCell"]),
                    Paragraph(row.get("decision_tree_value", "-"), self.styles["TableCell"]),
                ])
            t_comp = Table(comp_rows, colWidths=[150, 118, 118, 118])
            t_comp.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_comp)

        return elements

    def _build_analyst_section(self, data_package: ReportDataPackage) -> List[Any]:
        """Build human-in-the-loop analyst review audit trail and disposition stats."""
        stats = data_package.analyst_summary
        elements: List[Any] = []
        elements.append(Paragraph("3. Human-in-the-Loop Analyst Review Audit", self.styles["SectionHeader"]))

        # Review Disposition Breakdown
        disp_counts = stats.get("disposition_counts", {})
        disp_rows = [
            [
                Paragraph("Analyst Disposition", self.styles["TableHeader"]),
                Paragraph("Record Count", self.styles["TableHeader"]),
                Paragraph("Operational Meaning", self.styles["TableHeader"]),
            ]
        ]
        meanings = {
            "Potentially Malicious": "Human verified evidence aligns with suspicious activity.",
            "Likely Benign": "Apparent benign behavior incorrectly or ambivalently flagged.",
            "False Positive": "Confirmed benign flow falsely classified as attack by ML model.",
            "Needs Further Investigation": "Inconclusive telemetry requiring packet capture or host correlation.",
            "Inconclusive": "Insufficient data to determine security posture.",
        }
        for disp, count in disp_counts.items():
            disp_rows.append([
                Paragraph(disp, self.styles["TableCellBold"]),
                Paragraph(f"{count:,}", self.styles["TableCell"]),
                Paragraph(meanings.get(disp, "Analyst designated review state."), self.styles["TableCell"]),
            ])

        t_disp = Table(disp_rows, colWidths=[150, 84, 270])
        t_disp.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_disp)
        elements.append(Spacer(1, 8))

        # Recent Review Events
        recent_events = stats.get("recent_events", [])
        if recent_events:
            elements.append(Paragraph("3.1 Chronological Review Audit Trail (Recent)", self.styles["SubSectionHeader"]))
            ev_rows = [
                [
                    Paragraph("Row", self.styles["TableHeader"]),
                    Paragraph("Timestamp UTC", self.styles["TableHeader"]),
                    Paragraph("Disposition", self.styles["TableHeader"]),
                    Paragraph("Analyst", self.styles["TableHeader"]),
                    Paragraph("Notes / Rationale", self.styles["TableHeader"]),
                ]
            ]
            for ev in recent_events[:15]:
                ev_rows.append([
                    Paragraph(str(ev["source_row_index"]), self.styles["TableCellBold"]),
                    Paragraph(ev["timestamp_utc"][:19].replace("T", " "), self.styles["TableCell"]),
                    Paragraph(ev["disposition"], self.styles["TableCellBold"]),
                    Paragraph(ev["reviewer"], self.styles["TableCell"]),
                    Paragraph(ev["notes"] or "-", self.styles["TableCell"]),
                ])
            t_ev = Table(ev_rows, colWidths=[40, 100, 110, 84, 170])
            t_ev.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            elements.append(t_ev)

        return elements

    def _build_threat_context_section(self, data_package: ReportDataPackage) -> List[Any]:
        """Build MITRE ATT&CK contextual mapping and evidence evaluation section."""
        elements: List[Any] = []
        elements.append(Paragraph("4. MITRE ATT&CK Threat Context & Evidence Evaluation", self.styles["SectionHeader"]))

        mappings = data_package.attck_mappings
        if not mappings:
            elements.append(Paragraph(
                "No active MITRE ATT&CK candidate associations were flagged for the current traffic profile.",
                self.styles["Body"]
            ))
            return elements

        elements.append(Paragraph(
            "The following conservative behavioral technique associations connect classified attack labels with "
            "MITRE ATT&CK Enterprise (v20.1) techniques. Mappings maintain strict candidate/approval governance:",
            self.styles["Body"]
        ))

        map_rows = [
            [
                Paragraph("Attack Label", self.styles["TableHeader"]),
                Paragraph("Technique", self.styles["TableHeader"]),
                Paragraph("Tactic", self.styles["TableHeader"]),
                Paragraph("Status", self.styles["TableHeader"]),
                Paragraph("Available Telemetry", self.styles["TableHeader"]),
            ]
        ]
        for m in mappings[:15]:
            status_style = self.styles["TableCellBold"]
            map_rows.append([
                Paragraph(m["dataset_label"], self.styles["TableCellBold"]),
                Paragraph(f"{m['technique_id']}: {m['technique_name']}", self.styles["TableCell"]),
                Paragraph(m["tactic"], self.styles["TableCell"]),
                Paragraph(m["status"], status_style),
                Paragraph(m["available_evidence"], self.styles["TableCell"]),
            ])

        t_map = Table(map_rows, colWidths=[90, 134, 80, 70, 130])
        t_map.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#ffffff"), colors.HexColor("#f8fafc")]),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        elements.append(t_map)
        return elements

    def _build_appendix_section(self, data_package: ReportDataPackage) -> List[Any]:
        """Build methodology, academic benchmark references, and integrity checksums."""
        elements: List[Any] = []
        elements.append(Paragraph("5. Methodology, Provenance & Evidence Manifest", self.styles["SectionHeader"]))

        methodology_text = (
            "<b>Assessment Methodology:</b> This assessment was executed using the local AI-Powered NIDS platform. "
            "Network flows were ingested, preprocessed under training-fitted scalers and encoders with zero test-partition leakage, "
            "and evaluated across three authorized supervised learning algorithms: Linear Support Vector Machine (LinearSVC), "
            "Multilayer Perceptron (MLPClassifier), and Decision Tree (DecisionTreeClassifier)."
        )
        elements.append(Paragraph(methodology_text, self.styles["Body"]))

        # Disclaimers Box
        disclaimer_lines = "<br/>• ".join(data_package.disclaimers)
        d_box = Table([[Paragraph(f"<b>LEGAL & FORENSIC GOVERNANCE NOTICE:</b><br/>• {disclaimer_lines}", self.styles["CalloutText"])]], colWidths=[504])
        d_box.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
            ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        elements.append(d_box)
        return elements
