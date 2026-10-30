"""
AI Clinical Risk Report Generator Module.
Formats patient risk predictions, SHAP attributions, and clinical recommendations
into standardized text reports and publication-grade PDFs.
"""

import os
import datetime
from typing import Dict, Any, List, Tuple, Optional

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

NORMAL_RANGES = {
    "Glucose": "70 - 100 mg/dL",
    "BloodPressure": "60 - 80 mmHg",
    "SkinThickness": "10 - 30 mm",
    "Insulin": "15 - 160 mIU/L",
    "BMI": "18.5 - 24.9 kg/m²",
    "trestbps": "90 - 120 mmHg",
    "chol": "125 - 200 mg/dL",
    "thalach": "100 - 170 bpm",
    "oldpeak": "< 1.0 mm",
    "bgr": "70 - 125 mg/dL",
    "bu": "7 - 20 mg/dL",
    "sc": "0.6 - 1.2 mg/dL",
    "sod": "135 - 145 mEq/L",
    "pot": "3.5 - 5.0 mEq/L",
    "hemo": "13.5 - 17.5 g/dL",
    "pcv": "38 - 50 %",
    "wc": "4500 - 11000 /µL",
    "rc": "4.5 - 5.9 M/µL",
    "Total_Bilirubin": "0.1 - 1.2 mg/dL",
    "Direct_Bilirubin": "0.0 - 0.3 mg/dL",
    "Alkaline_Phosphotase": "44 - 147 IU/L",
    "Alamine_Aminotransferase": "7 - 56 IU/L",
    "Aspartate_Aminotransferase": "10 - 40 IU/L",
    "Total_Protiens": "6.0 - 8.3 g/dL",
    "Albumin": "3.5 - 5.0 g/dL",
    "Albumin_and_Globulin_Ratio": "0.8 - 2.0"
}

class ClinicalReportGenerator:
    """
    Generates standardized, physician-facing clinical risk reports.
    """
    def __init__(self, disease_name: str):
        self.disease_name = disease_name

    def assign_risk_tier(self, probability: float) -> Tuple[str, str, Tuple[float, float, float]]:
        """
        Assigns a clinical risk tier and corresponding color (RGB) based on probability.
        Returns: (Tier Title, Description, Color RGB Tuple)
        """
        if probability < 0.25:
            # Low Risk: Green
            return "TIER I - LOW RISK", "Biomarkers within normal boundaries.", (0.1, 0.6, 0.2)
        elif probability < 0.50:
            # Moderate Risk: Blue
            return "TIER II - MODERATE RISK", "Early sub-clinical biomarker elevations detected.", (0.1, 0.4, 0.8)
        elif probability < 0.75:
            # High Risk: Orange
            return "TIER III - HIGH RISK", "Pathological thresholds exceeded; early intervention recommended.", (0.9, 0.5, 0.1)
        else:
            # Critical Risk: Red
            return "TIER IV - CRITICAL RISK", "Severe pathological indicators present; immediate specialist referral required.", (0.8, 0.1, 0.1)

    def generate_text_report(
        self,
        patient_id: str,
        age: int,
        gender: str,
        probability: float,
        confidence: float,
        shap_attributions: Dict[str, float],
        patient_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        tier_title, tier_desc, _ = self.assign_risk_tier(probability)
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        sorted_attributions = sorted(shap_attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        top_positive = [(f, v) for f, v in sorted_attributions if v > 0][:4]
        top_negative = [(f, v) for f, v in sorted_attributions if v < 0][:2]

        report = []
        report.append("=" * 80)
        report.append(f"            EXPLAINABLE CLINICAL RISK REPORT — {self.disease_name.upper()}")
        report.append("=" * 80)
        report.append(f"Patient ID : {patient_id:<20} Age / Gender : {age} Yrs / {gender}")
        report.append(f"Report Date: {now_str:<20} Target Condition: {self.disease_name}")
        report.append("-" * 80)
        report.append("DIAGNOSTIC RISK STRATIFICATION")
        report.append(f"Predicted Pathological Probability: {probability:.4f} ({probability * 100:.1f}%)")
        report.append(f"Assigned Clinical Risk Tier       : {tier_title}")
        report.append(f"Model Prediction Confidence       : {confidence * 100:.1f}%")
        report.append(f"Clinical Context Summary          : {tier_desc}")
        report.append("-" * 80)
        
        if patient_metrics:
            report.append("PATIENT LABORATORY METRICS & NORMAL RANGES")
            report.append(f"  {'Biomarker':<25} {'Raw Value':<15} {'Normal Range':<20}")
            for key, val in patient_metrics.items():
                ref = NORMAL_RANGES.get(key, "N/A")
                # Format floats cleanly
                val_str = f"{val:.2f}" if isinstance(val, (float, int)) else str(val)
                report.append(f"  {key:<25} {val_str:<15} {ref:<20}")
            report.append("-" * 80)

        report.append("TOP BIOMARKER ATTRIBUTIONS (SHAP VALUES)")
        report.append("Risk Accelerators (Pushing Risk Higher):")
        for feature, val in top_positive:
            report.append(f"  + {feature:<25} SHAP Attribution: +{val:.4f}")
        if top_negative:
            report.append("Protective Factors (Lowering Risk):")
            for feature, val in top_negative:
                report.append(f"  - {feature:<25} SHAP Attribution: {val:.4f}")
        report.append("-" * 80)
        report.append("PHYSICIAN DECISION SUPPORT GUIDANCE")
        if probability >= 0.50:
            report.append("  1. Order secondary confirmatory laboratory panels.")
            report.append("  2. Schedule immediate specialist consultation.")
            report.append("  3. Review patient blood pressure and metabolic markers.")
        else:
            report.append("  1. Continue routine preventative screening protocol.")
            report.append("  2. Re-assess patient biomarker levels in 6 to 12 months.")
        report.append("-" * 80)
        report.append("NOTICE: This AI decision support report is intended solely for licensed medical")
        report.append("professionals. It does not replace independent clinical judgment.")
        report.append("=" * 80)

        return "\n".join(report)

    def generate_pdf_report(
        self,
        output_path: str,
        patient_id: str,
        age: int,
        gender: str,
        probability: float,
        confidence: float,
        shap_attributions: Dict[str, float],
        patient_metrics: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generates a beautifully typeset, professional 2-page PDF report.
        """
        if not REPORTLAB_AVAILABLE:
            # Fallback if ReportLab is not present (even though we verified it is)
            txt_report = self.generate_text_report(patient_id, age, gender, probability, confidence, shap_attributions, patient_metrics)
            with open(output_path.replace(".pdf", ".txt"), "w") as f:
                f.write(txt_report)
            return output_path.replace(".pdf", ".txt")

        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )
        
        styles = getSampleStyleSheet()
        
        # Define Custom Styles
        title_style = ParagraphStyle(
            name='TitleStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#1A365D'),
            alignment=1, # Centered
            spaceAfter=15
        )

        header_style = ParagraphStyle(
            name='HeaderStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#2C3E50'),
            spaceBefore=10,
            spaceAfter=5
        )

        normal_style = ParagraphStyle(
            name='NormalText',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#333333')
        )

        bold_style = ParagraphStyle(
            name='BoldText',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=10,
            leading=14,
            textColor=colors.HexColor('#333333')
        )

        disclaimer_style = ParagraphStyle(
            name='DisclaimerStyle',
            parent=styles['Normal'],
            fontName='Helvetica-Oblique',
            fontSize=8,
            leading=11,
            textColor=colors.HexColor('#7F8C8D'),
            spaceBefore=15
        )

        tier_title, tier_desc, tier_color_rgb = self.assign_risk_tier(probability)
        tier_color = colors.Color(*tier_color_rgb)

        story = []

        # 1. Main Title
        story.append(Paragraph(f"EXPLAINABLE CLINICAL RISK REPORT — {self.disease_name.upper()}", title_style))
        story.append(Spacer(1, 10))

        # 2. Patient Metadata Table
        meta_data = [
            [Paragraph("<b>Patient ID:</b>", normal_style), Paragraph(patient_id, normal_style), 
             Paragraph("<b>Age / Gender:</b>", normal_style), Paragraph(f"{age} Yrs / {gender}", normal_style)],
            [Paragraph("<b>Report Date:</b>", normal_style), Paragraph(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), normal_style),
             Paragraph("<b>Diagnostic Target:</b>", normal_style), Paragraph(self.disease_name, normal_style)]
        ]
        meta_table = Table(meta_data, colWidths=[90, 180, 90, 180])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('RIGHTPADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BDC3C7')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0'))
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 15))

        # 3. Stratification Box
        strat_data = [
            [
                Paragraph("<font color='white'><b>RISK STRATIFICATION SUMMARY</b></font>", ParagraphStyle('StratTitle', parent=bold_style, textColor=colors.white)),
                ""
            ],
            [Paragraph("Predicted Pathological Probability:", bold_style), Paragraph(f"<b>{probability:.4f} ({probability * 100:.1f}%)</b>", normal_style)],
            [Paragraph("Assigned Clinical Risk Tier:", bold_style), Paragraph(f"<font color='{tier_color.hexval()}'><b>{tier_title}</b></font>", bold_style)],
            [Paragraph("Model Prediction Confidence:", bold_style), Paragraph(f"{confidence * 100:.1f}%", normal_style)],
            [Paragraph("Clinical Interpretation:", bold_style), Paragraph(tier_desc, normal_style)]
        ]
        strat_table = Table(strat_data, colWidths=[200, 340])
        strat_table.setStyle(TableStyle([
            ('SPAN', (0,0), (1,0)),
            ('BACKGROUND', (0,0), (-1,0), tier_color),
            ('BACKGROUND', (0,1), (-1,-1), colors.HexColor('#FFFDF0') if probability >= 0.5 else colors.HexColor('#F4FBF7')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('BOX', (0,0), (-1,-1), 1.5, tier_color),
            ('INNERGRID', (0,1), (-1,-1), 0.5, colors.HexColor('#E2E8F0'))
        ]))
        story.append(strat_table)
        story.append(Spacer(1, 15))

        # 4. Patient Laboratory Metrics (If available)
        if patient_metrics:
            story.append(Paragraph("PATIENT CLINICAL METRICS & NORMAL REFERENCE RANGES", header_style))
            metric_rows = [[Paragraph("<b>Biomarker</b>", bold_style), Paragraph("<b>Observed Raw Value</b>", bold_style), Paragraph("<b>Normal Reference Range</b>", bold_style)]]
            for key, val in patient_metrics.items():
                ref = NORMAL_RANGES.get(key, "N/A")
                val_str = f"{val:.2f}" if isinstance(val, (float, int)) else str(val)
                metric_rows.append([Paragraph(key, normal_style), Paragraph(val_str, normal_style), Paragraph(ref, normal_style)])
            
            metric_table = Table(metric_rows, colWidths=[180, 180, 180])
            metric_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECEFF1')),
                ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('BOTTOMPADDING', (0,0), (-1,-1), 4),
                ('TOPPADDING', (0,0), (-1,-1), 4),
                ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BDC3C7')),
                ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CFD8DC'))
            ]))
            story.append(metric_table)
            story.append(Spacer(1, 15))

        # 5. SHAP Attributions Table
        story.append(Paragraph("EXPLAINABLE AI ATTRIBUTIONS (SHAP BREAKDOWN)", header_style))
        sorted_attributions = sorted(shap_attributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        shap_rows = [[Paragraph("<b>Biomarker Factor</b>", bold_style), Paragraph("<b>Risk Influence Category</b>", bold_style), Paragraph("<b>SHAP Value (Marginal Impact)</b>", bold_style)]]
        for feature, val in sorted_attributions[:6]:
            status = "<font color='red'>Risk Accelerator (+)</font>" if val > 0 else "<font color='green'>Protective Factor (-)</font>"
            val_str = f"+{val:.4f}" if val > 0 else f"{val:.4f}"
            shap_rows.append([Paragraph(feature, normal_style), Paragraph(status, normal_style), Paragraph(val_str, normal_style)])
            
        shap_table = Table(shap_rows, colWidths=[180, 180, 180])
        shap_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#ECEFF1')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 4),
            ('TOPPADDING', (0,0), (-1,-1), 4),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#BDC3C7')),
            ('INNERGRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CFD8DC'))
        ]))
        story.append(shap_table)
        story.append(Spacer(1, 15))

        # 6. Physician Decision Support & Recommendations
        story.append(Paragraph("RECOMMENDED FOLLOW-UP CLINICAL PROTOCOLS", header_style))
        if probability >= 0.50:
            recs = [
                "<b>1. Confirmatory Testing:</b> Order immediate secondary diagnostic confirmation (e.g. Stress ECG, renal ultrasound, secondary panels).",
                "<b>2. Specialist Referral:</b> Initiate immediate specialist consultation matching the pathological disease profile.",
                "<b>3. Monitoring Frequency:</b> Re-evaluate critical biomarker parameters within 48-72 hours."
            ]
        else:
            recs = [
                "<b>1. Routine Screening:</b> Maintain standard yearly diagnostic screening schedule.",
                "<b>2. Lifestyle Monitoring:</b> Review primary metabolic indices during normal checkups.",
                "<b>3. Re-assessment:</b> Repeat this screening evaluation in 6 to 12 months."
            ]
            
        rec_data = [[Paragraph(rec, normal_style)] for rec in recs]
        rec_table = Table(rec_data, colWidths=[540])
        rec_table.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8F9FA')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ('TOPPADDING', (0,0), (-1,-1), 6),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
            ('BOX', (0,0), (-1,-1), 1, colors.HexColor('#E2E8F0'))
        ]))
        story.append(rec_table)
        story.append(Spacer(1, 15))

        # 7. Disclaimer
        story.append(Paragraph(
            "<b>LEGAL & ETHICAL NOTICE:</b> This document is generated by an artificial intelligence decision support system "
            "intended strictly to assist licensed healthcare providers. It does not replace independent clinical diagnosis, "
            "and final medical decisions remain the sole responsibility of the attending physician.",
            disclaimer_style
        ))

        doc.build(story)
        return output_path
