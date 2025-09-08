"""
Pharmaceutical Document Analyzer
Specialized implementation for pharmaceutical and clinical research applications
"""

from typing import Dict, List, Optional, Any, Union
import logging
import re
from datetime import datetime
from .apertus_core import ApertusCore

logger = logging.getLogger(__name__)


class PharmaDocumentAnalyzer:
    """
    Pharmaceutical document analyzer for clinical trials, safety reports,
    and regulatory compliance using Apertus Swiss AI
    
    Provides specialized analysis for pharmaceutical industry with focus on
    safety, efficacy, regulatory compliance, and transparency.
    """
    
    def __init__(self, apertus_core: Optional[ApertusCore] = None):
        """
        Initialize pharmaceutical analyzer
        
        Args:
            apertus_core: Initialized ApertusCore instance, or None to create new
        """
        if apertus_core is None:
            self.apertus = ApertusCore()
        else:
            self.apertus = apertus_core
        
        self.analysis_history = []
        
        # Pharmaceutical-specific system message
        self.pharma_system = """You are a pharmaceutical AI specialist with expertise in:
        - Clinical trial protocols and results analysis
        - Drug safety and pharmacovigilance
        - Regulatory compliance (FDA, EMA, Swissmedic)
        - Medical literature review and synthesis
        - Quality assurance documentation
        - Post-market surveillance
        
        Always maintain scientific accuracy, cite specific data points when available,
        and note any limitations in your analysis. Follow ICH guidelines and
        regulatory standards in your assessments."""
        
        # Analysis templates for different document types
        self.analysis_templates = {
            "safety": self._get_safety_template(),
            "efficacy": self._get_efficacy_template(),
            "regulatory": self._get_regulatory_template(),
            "pharmacokinetics": self._get_pk_template(),
            "adverse_events": self._get_ae_template(),
            "drug_interactions": self._get_interaction_template(),
            "quality": self._get_quality_template()
        }
        
        logger.info("💊 Pharmaceutical Document Analyzer initialized")
    
    def analyze_clinical_document(
        self,
        document_text: str,
        analysis_type: str = "safety",
        document_type: str = "clinical_study",
        language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Comprehensive analysis of clinical/pharmaceutical documents
        
        Args:
            document_text: Full text of the document to analyze
            analysis_type: Type of analysis (safety, efficacy, regulatory, etc.)
            document_type: Type of document (clinical_study, protocol, csr, etc.)
            language: Language for analysis output
            
        Returns:
            Structured analysis results
        """
        logger.info(f"📄 Analyzing {document_type} document ({analysis_type} focus)")
        
        if analysis_type not in self.analysis_templates:
            raise ValueError(f"Unsupported analysis type: {analysis_type}")
        
        # Prepare document for analysis
        processed_text = self._preprocess_document(document_text)
        
        # Get analysis template
        template = self.analysis_templates[analysis_type]
        prompt = template.format(
            document_text=processed_text,
            document_type=document_type
        )
        
        # Generate analysis
        response = self.apertus.generate_response(
            prompt,
            max_new_tokens=800,
            temperature=0.3,  # Lower temperature for factual analysis
            system_message=self.pharma_system
        )
        
        # Structure the results
        analysis_result = {
            "analysis_type": analysis_type,
            "document_type": document_type,
            "timestamp": datetime.now().isoformat(),
            "raw_analysis": response,
            "structured_findings": self._structure_analysis(response, analysis_type),
            "document_stats": self._get_document_stats(processed_text)
        }
        
        # Store in history
        self.analysis_history.append(analysis_result)
        
        return analysis_result
    
    def extract_adverse_events(
        self,
        document_text: str,
        severity_classification: bool = True
    ) -> Dict[str, Any]:
        """
        Extract and classify adverse events from clinical documents
        
        Args:
            document_text: Clinical document text
            severity_classification: Whether to classify severity
            
        Returns:
            Structured adverse events data
        """
        ae_prompt = f"""Extract all adverse events (AEs) from this clinical document.
        For each adverse event, provide:
        
        1. EVENT DETAILS:
           - Event name/description
           - Frequency/incidence if mentioned
           - Time to onset if available
           - Duration if mentioned
           
        2. SEVERITY ASSESSMENT:
           - Grade/severity (1-5 or mild/moderate/severe)
           - Serious adverse event (SAE) classification
           - Relationship to study drug (related/unrelated/possibly related)
           
        3. PATIENT INFORMATION:
           - Demographics if available
           - Dose/treatment information
           - Outcome (resolved/ongoing/fatal/etc.)
           
        4. REGULATORY CLASSIFICATION:
           - Expected vs unexpected
           - Reportable events
           - Action taken (dose reduction, discontinuation, etc.)
        
        Format as structured list with clear categorization.
        
        Document: {document_text}
        
        ADVERSE EVENTS ANALYSIS:"""
        
        response = self.apertus.generate_response(
            ae_prompt,
            max_new_tokens=600,
            temperature=0.2,
            system_message=self.pharma_system
        )
        
        # Extract structured data
        ae_data = {
            "total_aes_mentioned": self._count_ae_mentions(response),
            "severity_distribution": self._extract_severity_info(response),
            "serious_aes": self._extract_serious_aes(response),
            "raw_extraction": response,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return ae_data
    
    def analyze_drug_interactions(
        self,
        document_text: str,
        drug_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze potential drug interactions from clinical or pharmacology documents
        
        Args:
            document_text: Document containing interaction information
            drug_name: Primary drug name if known
            
        Returns:
            Structured interaction analysis
        """
        interaction_prompt = f"""Analyze this document for drug interactions and pharmacological considerations.
        
        PRIMARY FOCUS:
        {f"Primary drug: {drug_name}" if drug_name else "Identify all drugs mentioned"}
        
        ANALYSIS REQUIREMENTS:
        
        1. DRUG INTERACTIONS IDENTIFIED:
           - Drug A + Drug B: [interaction type] - [severity] - [mechanism]
           - Clinical significance (major/moderate/minor)
           - Onset and duration of interaction
           
        2. PHARMACOKINETIC INTERACTIONS:
           - CYP enzyme involvement
           - Absorption, distribution, metabolism, excretion effects
           - Dose adjustment recommendations
           
        3. PHARMACODYNAMIC INTERACTIONS:
           - Additive/synergistic effects
           - Antagonistic interactions
           - Receptor-level interactions
           
        4. CLINICAL RECOMMENDATIONS:
           - Monitoring requirements
           - Dose modifications
           - Timing considerations
           - Contraindications
           
        5. SPECIAL POPULATIONS:
           - Elderly patients
           - Hepatic/renal impairment
           - Pregnancy/lactation considerations
        
        Document: {document_text}
        
        DRUG INTERACTION ANALYSIS:"""
        
        response = self.apertus.generate_response(
            interaction_prompt,
            max_new_tokens=700,
            temperature=0.3,
            system_message=self.pharma_system
        )
        
        return {
            "primary_drug": drug_name,
            "interactions_identified": self._count_interactions(response),
            "severity_breakdown": self._extract_interaction_severity(response),
            "clinical_significance": self._assess_clinical_significance(response),
            "recommendations": self._extract_recommendations(response),
            "raw_analysis": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def regulatory_compliance_check(
        self,
        document_text: str,
        regulatory_body: str = "FDA",
        document_type: str = "CSR"
    ) -> Dict[str, Any]:
        """
        Check document for regulatory compliance requirements
        
        Args:
            document_text: Document to check
            regulatory_body: Regulatory authority (FDA, EMA, Swissmedic)
            document_type: Type of regulatory document
            
        Returns:
            Compliance assessment results
        """
        compliance_prompt = f"""Review this {document_type} document for {regulatory_body} compliance.
        
        COMPLIANCE CHECKLIST:
        
        1. REQUIRED DISCLOSURES:
           ✓ Safety information completeness
           ✓ Proper labeling elements
           ✓ Risk-benefit assessment
           ✓ Contraindications and warnings
           
        2. DATA INTEGRITY:
           ✓ Statistical analysis completeness
           ✓ Primary/secondary endpoint reporting
           ✓ Missing data handling
           ✓ Protocol deviations documentation
           
        3. REGULATORY STANDARDS:
           ✓ ICH guidelines adherence
           ✓ {regulatory_body} specific requirements
           ✓ Good Clinical Practice (GCP) compliance
           ✓ Quality by Design principles
           
        4. SUBMISSION READINESS:
           ✓ Document structure and format
           ✓ Required sections presence
           ✓ Cross-references and consistency
           ✓ Executive summary quality
           
        5. RISK MANAGEMENT:
           ✓ Risk evaluation and mitigation strategies (REMS)
           ✓ Post-market surveillance plans
           ✓ Safety monitoring adequacy
           
        For each item, provide: COMPLIANT/NON-COMPLIANT/UNCLEAR and specific comments.
        
        Document: {document_text}
        
        REGULATORY COMPLIANCE ASSESSMENT:"""
        
        response = self.apertus.generate_response(
            compliance_prompt,
            max_new_tokens=800,
            temperature=0.2,
            system_message=self.pharma_system
        )
        
        return {
            "regulatory_body": regulatory_body,
            "document_type": document_type,
            "compliance_score": self._calculate_compliance_score(response),
            "critical_issues": self._extract_critical_issues(response),
            "recommendations": self._extract_compliance_recommendations(response),
            "compliant_items": self._count_compliant_items(response),
            "raw_assessment": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def generate_safety_summary(
        self,
        documents: List[str],
        study_phase: str = "Phase II"
    ) -> Dict[str, Any]:
        """
        Generate comprehensive safety summary from multiple documents
        
        Args:
            documents: List of document texts to analyze
            study_phase: Clinical study phase
            
        Returns:
            Integrated safety summary
        """
        logger.info(f"📊 Generating integrated safety summary for {len(documents)} documents")
        
        # Analyze each document for safety
        individual_analyses = []
        for i, doc in enumerate(documents):
            analysis = self.analyze_clinical_document(
                doc, 
                analysis_type="safety",
                document_type=f"document_{i+1}"
            )
            individual_analyses.append(analysis)
        
        # Create integrated summary
        integration_prompt = f"""Create an integrated safety summary for this {study_phase} study 
        based on the following individual document analyses:
        
        {self._format_analyses_for_integration(individual_analyses)}
        
        INTEGRATED SAFETY SUMMARY REQUIREMENTS:
        
        1. OVERALL SAFETY PROFILE:
           - Most common adverse events (≥5% incidence)
           - Serious adverse events summary
           - Deaths and life-threatening events
           - Discontinuations due to AEs
           
        2. SAFETY BY SYSTEM ORGAN CLASS:
           - Cardiovascular events
           - Gastrointestinal events  
           - Neurological events
           - Hepatic events
           - Other significant findings
           
        3. DOSE-RESPONSE RELATIONSHIPS:
           - Dose-dependent AEs if applicable
           - Maximum tolerated dose considerations
           - Dose modification patterns
           
        4. SPECIAL POPULATIONS:
           - Elderly patients (≥65 years)
           - Gender differences
           - Comorbidity considerations
           
        5. BENEFIT-RISK ASSESSMENT:
           - Risk acceptability for indication
           - Comparison to standard of care
           - Risk mitigation strategies
           
        6. REGULATORY CONSIDERATIONS:
           - Labeling implications
           - Post-market surveillance needs
           - Risk management plans
        
        INTEGRATED SAFETY SUMMARY:"""
        
        summary_response = self.apertus.generate_response(
            integration_prompt,
            max_new_tokens=1000,
            temperature=0.3,
            system_message=self.pharma_system
        )
        
        return {
            "study_phase": study_phase,
            "documents_analyzed": len(documents),
            "individual_analyses": individual_analyses,
            "integrated_summary": summary_response,
            "key_safety_signals": self._extract_safety_signals(summary_response),
            "regulatory_recommendations": self._extract_regulatory_recs(summary_response),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_safety_template(self) -> str:
        """Safety analysis template"""
        return """Analyze this {document_type} document for safety information:

        1. ADVERSE EVENTS SUMMARY:
           - List all adverse events with frequencies
           - Categorize by severity (Grade 1-5 or mild/moderate/severe)
           - Identify serious adverse events (SAEs)
           - Note any dose-limiting toxicities

        2. SAFETY PROFILE ASSESSMENT:
           - Most common AEs (≥5% incidence)
           - Comparison to placebo/control if available
           - Dose-response relationships
           - Time to onset patterns

        3. SPECIAL SAFETY CONSIDERATIONS:
           - Drug interactions identified
           - Contraindications and warnings
           - Special population considerations
           - Long-term safety implications

        4. REGULATORY SAFETY REQUIREMENTS:
           - Reportable events identification
           - Safety monitoring adequacy
           - Risk mitigation strategies
           - Post-market surveillance needs

        Document: {document_text}

        SAFETY ANALYSIS:"""
    
    def _get_efficacy_template(self) -> str:
        """Efficacy analysis template"""
        return """Evaluate the efficacy data in this {document_type} document:

        1. PRIMARY ENDPOINTS:
           - Primary efficacy measures and results
           - Statistical significance (p-values, confidence intervals)
           - Effect size and clinical relevance
           - Response rates and duration

        2. SECONDARY ENDPOINTS:
           - Secondary measures and outcomes
           - Exploratory analyses results
           - Biomarker data if available
           - Quality of life assessments

        3. CLINICAL SIGNIFICANCE:
           - Real-world clinical relevance
           - Comparison to standard of care
           - Number needed to treat (NNT)
           - Magnitude of benefit assessment

        4. STUDY LIMITATIONS:
           - Methodological considerations
           - Generalizability assessment
           - Missing data impact
           - Statistical power considerations

        Document: {document_text}

        EFFICACY ANALYSIS:"""
    
    def _get_regulatory_template(self) -> str:
        """Regulatory compliance template"""
        return """Review this {document_type} document for regulatory compliance:

        1. REQUIRED DISCLOSURES:
           - Mandatory safety information completeness
           - Proper labeling elements inclusion
           - Risk-benefit assessment adequacy
           - Contraindications documentation

        2. DATA INTEGRITY ASSESSMENT:
           - Statistical analysis completeness
           - Protocol adherence documentation
           - Missing data handling
           - Quality control measures

        3. REGULATORY STANDARDS COMPLIANCE:
           - ICH guidelines adherence
           - Regulatory body specific requirements
           - Good Clinical Practice (GCP) compliance
           - Documentation standards

        4. SUBMISSION READINESS:
           - Document structure adequacy
           - Required sections completeness
           - Cross-reference consistency
           - Executive summary quality

        Document: {document_text}

        REGULATORY COMPLIANCE REVIEW:"""
    
    def _get_pk_template(self) -> str:
        """Pharmacokinetics template"""
        return """Analyze pharmacokinetic data in this {document_type} document:

        1. PK PARAMETERS:
           - Absorption characteristics (Cmax, Tmax)
           - Distribution parameters (Vd)
           - Metabolism pathways (CYP enzymes)
           - Elimination parameters (half-life, clearance)

        2. POPULATION PK ANALYSIS:
           - Demographic effects on PK
           - Disease state impact
           - Drug interaction effects
           - Special population considerations

        3. PK/PD RELATIONSHIPS:
           - Exposure-response relationships
           - Dose proportionality
           - Time-dependent changes
           - Biomarker correlations

        4. CLINICAL IMPLICATIONS:
           - Dosing recommendations
           - Monitoring requirements
           - Drug interaction potential
           - Special population dosing

        Document: {document_text}

        PHARMACOKINETIC ANALYSIS:"""
    
    def _get_ae_template(self) -> str:
        """Adverse events template"""
        return """Extract and analyze adverse events from this {document_type} document:

        1. AE IDENTIFICATION:
           - Complete list of adverse events
           - Incidence rates and frequencies
           - Severity grading (CTCAE or similar)
           - Causality assessment

        2. SAE ANALYSIS:
           - Serious adverse events detailed review
           - Outcome assessment
           - Regulatory reporting requirements
           - Death and life-threatening events

        3. AE PATTERNS:
           - System organ class distribution
           - Dose-response relationships
           - Time to onset analysis
           - Resolution patterns

        4. CLINICAL MANAGEMENT:
           - Dose modifications due to AEs
           - Discontinuation rates
           - Concomitant medication use
           - Supportive care requirements

        Document: {document_text}

        ADVERSE EVENTS ANALYSIS:"""
    
    def _get_interaction_template(self) -> str:
        """Drug interactions template"""
        return """Analyze drug interactions in this {document_type} document:

        1. INTERACTION IDENTIFICATION:
           - Drug pairs with interactions
           - Interaction mechanisms
           - Clinical significance assessment
           - Severity classification

        2. PHARMACOKINETIC INTERACTIONS:
           - CYP enzyme involvement
           - Transporter effects
           - Absorption/elimination changes
           - Dose adjustment needs

        3. PHARMACODYNAMIC INTERACTIONS:
           - Receptor-level interactions
           - Additive/synergistic effects
           - Antagonistic effects
           - Safety implications

        4. MANAGEMENT STRATEGIES:
           - Monitoring recommendations
           - Dose modifications
           - Timing considerations
           - Alternative therapies

        Document: {document_text}

        DRUG INTERACTION ANALYSIS:"""
    
    def _get_quality_template(self) -> str:
        """Quality assessment template"""
        return """Assess the quality aspects in this {document_type} document:

        1. STUDY DESIGN QUALITY:
           - Methodology appropriateness
           - Control group adequacy
           - Randomization quality
           - Blinding effectiveness

        2. DATA QUALITY:
           - Completeness assessment
           - Missing data patterns
           - Protocol deviations
           - Data integrity measures

        3. STATISTICAL QUALITY:
           - Analysis plan appropriateness
           - Power calculations
           - Multiple testing corrections
           - Sensitivity analyses

        4. REPORTING QUALITY:
           - CONSORT guideline compliance
           - Transparency in reporting
           - Bias risk assessment
           - Generalizability

        Document: {document_text}

        QUALITY ASSESSMENT:"""
    
    def _preprocess_document(self, text: str) -> str:
        """Preprocess document text for analysis"""
        # Limit text length for processing
        if len(text) > 4000:
            text = text[:4000] + "... [document truncated]"
        
        # Basic cleanup
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        
        return text
    
    def _structure_analysis(self, analysis: str, analysis_type: str) -> Dict[str, Any]:
        """Structure raw analysis into organized components"""
        # This is a simplified structuring - in production, you'd use more sophisticated NLP
        sections = {}
        
        current_section = "general"
        current_content = []
        
        for line in analysis.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check if line is a section header (starts with number or capital letters)
            if re.match(r'^\d+\.|\b[A-Z][A-Z\s]+:', line):
                # Save previous section
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                
                # Start new section
                current_section = line.lower().replace(':', '').strip()
                current_content = []
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _get_document_stats(self, text: str) -> Dict[str, Any]:
        """Get basic document statistics"""
        words = text.split()
        sentences = text.split('.')
        
        return {
            "word_count": len(words),
            "sentence_count": len(sentences),
            "character_count": len(text),
            "avg_sentence_length": len(words) / len(sentences) if sentences else 0
        }
    
    def _count_ae_mentions(self, text: str) -> int:
        """Count adverse event mentions in text"""
        ae_indicators = ['adverse event', 'side effect', 'toxicity', 'reaction']
        count = 0
        text_lower = text.lower()
        
        for indicator in ae_indicators:
            count += text_lower.count(indicator)
        
        return count
    
    def _extract_severity_info(self, text: str) -> Dict[str, int]:
        """Extract severity distribution from text"""
        severity_counts = {
            "mild": text.lower().count("mild"),
            "moderate": text.lower().count("moderate"), 
            "severe": text.lower().count("severe"),
            "grade_1": text.lower().count("grade 1"),
            "grade_2": text.lower().count("grade 2"),
            "grade_3": text.lower().count("grade 3"),
            "grade_4": text.lower().count("grade 4"),
            "grade_5": text.lower().count("grade 5")
        }
        
        return {k: v for k, v in severity_counts.items() if v > 0}
    
    def _extract_serious_aes(self, text: str) -> List[str]:
        """Extract serious adverse events from text"""
        # This is simplified - in production, use NER or more sophisticated extraction
        serious_indicators = ['serious adverse event', 'sae', 'life-threatening', 'fatal', 'death']
        found_saes = []
        
        for indicator in serious_indicators:
            if indicator in text.lower():
                found_saes.append(indicator)
        
        return found_saes
    
    def _count_interactions(self, text: str) -> int:
        """Count drug interactions mentioned"""
        interaction_patterns = [
            r'drug.*interaction', r'interaction.*between', 
            r'combined.*with', r'concomitant.*use'
        ]
        
        count = 0
        for pattern in interaction_patterns:
            count += len(re.findall(pattern, text.lower()))
        
        return count
    
    def _extract_interaction_severity(self, text: str) -> Dict[str, int]:
        """Extract interaction severity information"""
        return {
            "major": text.lower().count("major interaction"),
            "moderate": text.lower().count("moderate interaction"),
            "minor": text.lower().count("minor interaction")
        }
    
    def _assess_clinical_significance(self, text: str) -> str:
        """Assess clinical significance from text"""
        if "clinically significant" in text.lower():
            return "high"
        elif "moderate significance" in text.lower():
            return "moderate"
        elif "minor significance" in text.lower():
            return "low"
        else:
            return "unclear"
    
    def _extract_recommendations(self, text: str) -> List[str]:
        """Extract recommendations from analysis"""
        # Simplified extraction
        recommendations = []
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['recommend', 'suggest', 'should', 'monitor']):
                recommendations.append(line.strip())
        
        return recommendations
    
    def _calculate_compliance_score(self, text: str) -> float:
        """Calculate compliance score from assessment"""
        compliant = text.lower().count("compliant")
        non_compliant = text.lower().count("non-compliant")
        total = compliant + non_compliant
        
        if total == 0:
            return 0.0
        
        return (compliant / total) * 100
    
    def _extract_critical_issues(self, text: str) -> List[str]:
        """Extract critical compliance issues"""
        critical_indicators = ['critical', 'non-compliant', 'missing', 'inadequate', 'deficient']
        issues = []
        
        lines = text.split('\n')
        for line in lines:
            if any(indicator in line.lower() for indicator in critical_indicators):
                issues.append(line.strip())
        
        return issues
    
    def _extract_compliance_recommendations(self, text: str) -> List[str]:
        """Extract compliance recommendations"""
        return self._extract_recommendations(text)  # Reuse recommendation extraction
    
    def _count_compliant_items(self, text: str) -> Dict[str, int]:
        """Count compliant vs non-compliant items"""
        return {
            "compliant": text.lower().count("✓") + text.lower().count("compliant"),
            "non_compliant": text.lower().count("✗") + text.lower().count("non-compliant"),
            "unclear": text.lower().count("unclear")
        }
    
    def _format_analyses_for_integration(self, analyses: List[Dict]) -> str:
        """Format individual analyses for integration"""
        formatted = ""
        for i, analysis in enumerate(analyses, 1):
            formatted += f"\n--- Document {i} Analysis ---\n"
            formatted += analysis['raw_analysis'][:500] + "...\n"  # Truncate for length
        
        return formatted
    
    def _extract_safety_signals(self, text: str) -> List[str]:
        """Extract key safety signals from summary"""
        # Simplified extraction
        signals = []
        lines = text.split('\n')
        
        for line in lines:
            if any(word in line.lower() for word in ['signal', 'concern', 'warning', 'caution']):
                signals.append(line.strip())
        
        return signals
    
    def _extract_regulatory_recs(self, text: str) -> List[str]:
        """Extract regulatory recommendations"""
        return self._extract_recommendations(text)
    
    def get_analysis_history(self) -> List[Dict[str, Any]]:
        """Get history of all analyses performed"""
        return self.analysis_history
    
    def clear_history(self):
        """Clear analysis history"""
        self.analysis_history = []
        logger.info("Analysis history cleared")
    
    def export_analysis_report(self, analysis_id: Optional[int] = None) -> str:
        """
        Export analysis report in formatted text
        
        Args:
            analysis_id: Specific analysis to export (None for latest)
            
        Returns:
            Formatted analysis report
        """
        if not self.analysis_history:
            return "No analysis history available."
        
        if analysis_id is None:
            analysis = self.analysis_history[-1]
        else:
            if analysis_id >= len(self.analysis_history):
                return f"Analysis ID {analysis_id} not found."
            analysis = self.analysis_history[analysis_id]
        
        report = f"""
💊 PHARMACEUTICAL ANALYSIS REPORT
===============================

Analysis Type: {analysis['analysis_type'].upper()}
Document Type: {analysis['document_type']}
Timestamp: {analysis['timestamp']}

DOCUMENT STATISTICS:
- Word Count: {analysis['document_stats']['word_count']}
- Sentence Count: {analysis['document_stats']['sentence_count']}
- Average Sentence Length: {analysis['document_stats']['avg_sentence_length']:.1f} words

ANALYSIS RESULTS:
{analysis['raw_analysis']}

STRUCTURED FINDINGS:
"""
        
        for section, content in analysis['structured_findings'].items():
            report += f"\n{section.upper()}:\n{content}\n"
        
        report += f"\n{'='*50}\nReport generated by Apertus Swiss AI Pharmaceutical Analyzer\n"
        
        return report
    
    def __repr__(self):
        """String representation of the analyzer"""
        return f"PharmaDocumentAnalyzer(analyses_performed={len(self.analysis_history)})"
