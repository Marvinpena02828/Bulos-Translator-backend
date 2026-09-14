"""
Sentence/Phrase Entry Validation Module
========================================
Validates sentence entries for structure and integrity WITHOUT modifying linguistic content.

IMPORTANT:
- Does NOT automatically correct linguistic text
- Does NOT delete duplicates
- Does NOT resolve ambiguous mappings
- Does NOT standardize spelling/grammar variations
- PRESERVES all valid linguistic data
"""
import re
import unicodedata
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass, field


@dataclass
class ValidationIssue:
    """Represents a validation issue found in an entry"""
    entry_index: int
    issue_type: str
    field_name: Optional[str]
    description: str
    entry_data: Dict[str, str]


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    total_entries: int = 0
    valid_entries: List[Dict[str, str]] = field(default_factory=list)
    invalid_entries: List[ValidationIssue] = field(default_factory=list)
    exact_duplicates: List[Dict] = field(default_factory=list)
    multiple_mappings: List[Dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def summary(self) -> str:
        """Generate human-readable summary"""
        lines = []
        lines.append("=" * 60)
        lines.append("SENTENCE VALIDATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Total Entries Processed:    {self.total_entries}")
        lines.append(f"Valid Entries:              {len(self.valid_entries)}")
        lines.append(f"Invalid Entries:            {len(self.invalid_entries)}")
        lines.append(f"Exact Duplicate Groups:     {len(self.exact_duplicates)}")
        lines.append(f"Multiple Mapping Groups:    {len(self.multiple_mappings)}")
        lines.append(f"Warnings:                   {len(self.warnings)}")
        lines.append("=" * 60)
        
        if self.invalid_entries:
            lines.append("\nINVALID ENTRIES:")
            for issue in self.invalid_entries[:10]:
                lines.append(f"  Index {issue.entry_index}: {issue.issue_type}")
                lines.append(f"    {issue.description}")
            if len(self.invalid_entries) > 10:
                lines.append(f"  ... and {len(self.invalid_entries) - 10} more")
        
        if self.exact_duplicates:
            lines.append("\nEXACT DUPLICATES:")
            for dup_group in self.exact_duplicates[:5]:
                lines.append(f"  Indexes: {', '.join(map(str, dup_group['indexes']))}")
                lines.append(f"    BULOS: {dup_group['bulos']}")
            if len(self.exact_duplicates) > 5:
                lines.append(f"  ... and {len(self.exact_duplicates) - 5} more groups")
        
        if self.multiple_mappings:
            lines.append("\nMULTIPLE MAPPINGS (may be legitimate):")
            for mapping in self.multiple_mappings[:5]:
                lines.append(f"  Type: {mapping['type']}")
                lines.append(f"    Normalized: {mapping['normalized_value']}")
                lines.append(f"    Occurrences: {len(mapping['entries'])} entries")
            if len(self.multiple_mappings) > 5:
                lines.append(f"  ... and {len(self.multiple_mappings) - 5} more groups")
        
        if self.warnings:
            lines.append("\nWARNINGS:")
            for warning in self.warnings[:10]:
                lines.append(f"  {warning}")
            if len(self.warnings) > 10:
                lines.append(f"  ... and {len(self.warnings) - 10} more")
        
        return '\n'.join(lines)


class SentenceValidator:
    """
    Validates sentence/phrase entries for structural integrity.
    
    Does NOT modify linguistic content.
    PRESERVES legitimate variations and multiple mappings.
    """
    
    # Placeholder/null-like values to detect
    NULL_LIKE_VALUES = {
        'n/a', 'na', 'none', 'null', 'undefined', 'tbd', 'todo', '???', '...', '--', ''
    }
    
    def __init__(self):
        self.report = ValidationReport()
    
    def normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for duplicate/mapping detection.
        Uses same normalization as TranslationService for consistency.
        """
        # NFD decomposition + remove combining marks (accents)
        normalized = unicodedata.normalize('NFD', text)
        normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
        
        # Lowercase
        normalized = normalized.lower()
        
        # Trim and collapse whitespace
        normalized = normalized.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove trailing punctuation
        normalized = normalized.rstrip('.!?,;:')
        
        return normalized
    
    def validate_entry(self, entry: Dict[str, any], index: int) -> Tuple[bool, Optional[ValidationIssue]]:
        """
        Validate a single entry for structural integrity.
        
        Returns:
            (is_valid, issue_or_none)
        """
        # Check required fields exist
        required_fields = ['BULOS', 'FILIPINO', 'ENGLISH']
        
        for field in required_fields:
            if field not in entry:
                return (False, ValidationIssue(
                    entry_index=index,
                    issue_type='MISSING_FIELD',
                    field_name=field,
                    description=f"Required field '{field}' is missing",
                    entry_data=entry
                ))
        
        # Check field types and values
        for field in required_fields:
            value = entry[field]
            
            # Check if value is string
            if not isinstance(value, str):
                return (False, ValidationIssue(
                    entry_index=index,
                    issue_type='INVALID_TYPE',
                    field_name=field,
                    description=f"Field '{field}' must be string, got {type(value).__name__}",
                    entry_data=entry
                ))
            
            # Check if value is empty
            if not value:
                return (False, ValidationIssue(
                    entry_index=index,
                    issue_type='EMPTY_VALUE',
                    field_name=field,
                    description=f"Field '{field}' is empty",
                    entry_data=entry
                ))
            
            # Check if value is whitespace-only
            if not value.strip():
                return (False, ValidationIssue(
                    entry_index=index,
                    issue_type='WHITESPACE_ONLY',
                    field_name=field,
                    description=f"Field '{field}' contains only whitespace",
                    entry_data=entry
                ))
            
            # Check for null-like placeholder values
            value_normalized = value.strip().lower()
            if value_normalized in self.NULL_LIKE_VALUES:
                return (False, ValidationIssue(
                    entry_index=index,
                    issue_type='PLACEHOLDER_VALUE',
                    field_name=field,
                    description=f"Field '{field}' contains placeholder/null-like value: '{value}'",
                    entry_data=entry
                ))
        
        # Entry is valid
        return (True, None)
    
    def detect_exact_duplicates(self, entries: List[Dict[str, str]]) -> List[Dict]:
        """
        Detect exact duplicates (all three fields identical after normalization).
        
        Does NOT modify entries.
        Returns list of duplicate groups.
        """
        seen_triplets = {}  # normalized_triplet -> list of indexes
        duplicate_groups = []
        
        for idx, entry in enumerate(entries):
            # Normalize all three fields
            bulos_norm = self.normalize_for_comparison(entry['BULOS'])
            filipino_norm = self.normalize_for_comparison(entry['FILIPINO'])
            english_norm = self.normalize_for_comparison(entry['ENGLISH'])
            
            triplet = (bulos_norm, filipino_norm, english_norm)
            
            if triplet in seen_triplets:
                # Find existing group or create new one
                found_group = None
                for group in duplicate_groups:
                    if group['triplet'] == triplet:
                        found_group = group
                        break
                
                if found_group:
                    found_group['indexes'].append(idx)
                else:
                    # Create new duplicate group
                    duplicate_groups.append({
                        'triplet': triplet,
                        'indexes': [seen_triplets[triplet], idx],
                        'bulos': entry['BULOS'],
                        'filipino': entry['FILIPINO'],
                        'english': entry['ENGLISH']
                    })
            else:
                seen_triplets[triplet] = idx
        
        return duplicate_groups
    
    def detect_multiple_mappings(self, entries: List[Dict[str, str]]) -> List[Dict]:
        """
        Detect multiple/ambiguous mappings (same source → different translations).
        
        These are NOT automatically errors - may be legitimate variations.
        Does NOT modify entries.
        """
        # Track mappings: normalized_value -> list of (index, full_entry)
        bulos_mappings = {}
        filipino_mappings = {}
        english_mappings = {}
        
        for idx, entry in enumerate(entries):
            bulos_norm = self.normalize_for_comparison(entry['BULOS'])
            filipino_norm = self.normalize_for_comparison(entry['FILIPINO'])
            english_norm = self.normalize_for_comparison(entry['ENGLISH'])
            
            # Track BULOS -> FILIPINO/ENGLISH
            if bulos_norm not in bulos_mappings:
                bulos_mappings[bulos_norm] = []
            bulos_mappings[bulos_norm].append((idx, entry))
            
            # Track FILIPINO -> BULOS/ENGLISH
            if filipino_norm not in filipino_mappings:
                filipino_mappings[filipino_norm] = []
            filipino_mappings[filipino_norm].append((idx, entry))
            
            # Track ENGLISH -> BULOS/FILIPINO
            if english_norm not in english_mappings:
                english_mappings[english_norm] = []
            english_mappings[english_norm].append((idx, entry))
        
        # Detect multiple mappings
        multiple_mapping_groups = []
        
        # Check BULOS -> multiple different FILIPINO or ENGLISH
        for bulos_norm, occurrences in bulos_mappings.items():
            if len(occurrences) > 1:
                # Check if FILIPINO or ENGLISH differ
                filipinos = set(self.normalize_for_comparison(occ[1]['FILIPINO']) for occ in occurrences)
                englishes = set(self.normalize_for_comparison(occ[1]['ENGLISH']) for occ in occurrences)
                
                if len(filipinos) > 1 or len(englishes) > 1:
                    multiple_mapping_groups.append({
                        'type': 'SAME_BULOS_DIFFERENT_TRANSLATIONS',
                        'normalized_value': bulos_norm,
                        'entries': [{'index': occ[0], 'entry': occ[1]} for occ in occurrences]
                    })
        
        # Check FILIPINO -> multiple different BULOS
        for filipino_norm, occurrences in filipino_mappings.items():
            if len(occurrences) > 1:
                buloses = set(self.normalize_for_comparison(occ[1]['BULOS']) for occ in occurrences)
                if len(buloses) > 1:
                    multiple_mapping_groups.append({
                        'type': 'SAME_FILIPINO_DIFFERENT_BULOS',
                        'normalized_value': filipino_norm,
                        'entries': [{'index': occ[0], 'entry': occ[1]} for occ in occurrences]
                    })
        
        # Check ENGLISH -> multiple different BULOS
        for english_norm, occurrences in english_mappings.items():
            if len(occurrences) > 1:
                buloses = set(self.normalize_for_comparison(occ[1]['BULOS']) for occ in occurrences)
                if len(buloses) > 1:
                    multiple_mapping_groups.append({
                        'type': 'SAME_ENGLISH_DIFFERENT_BULOS',
                        'normalized_value': english_norm,
                        'entries': [{'index': occ[0], 'entry': occ[1]} for occ in occurrences]
                    })
        
        return multiple_mapping_groups
    
    def validate_entries(self, entries: List[Dict[str, any]]) -> ValidationReport:
        """
        Validate a list of sentence entries.
        
        Returns:
            ValidationReport with all findings
        """
        self.report = ValidationReport()
        self.report.total_entries = len(entries)
        
        if not entries:
            self.report.warnings.append("No entries provided for validation")
            return self.report
        
        # Phase 1: Validate individual entries
        valid_entries = []
        
        for idx, entry in enumerate(entries):
            is_valid, issue = self.validate_entry(entry, idx)
            
            if is_valid:
                valid_entries.append(entry)
            else:
                self.report.invalid_entries.append(issue)
        
        self.report.valid_entries = valid_entries
        
        # Phase 2: Detect exact duplicates (only in valid entries)
        if valid_entries:
            self.report.exact_duplicates = self.detect_exact_duplicates(valid_entries)
            
            if self.report.exact_duplicates:
                self.report.warnings.append(
                    f"Found {len(self.report.exact_duplicates)} exact duplicate group(s). "
                    f"These should be reviewed but were NOT automatically removed."
                )
        
        # Phase 3: Detect multiple/ambiguous mappings (only in valid entries)
        if valid_entries:
            self.report.multiple_mappings = self.detect_multiple_mappings(valid_entries)
            
            if self.report.multiple_mappings:
                self.report.warnings.append(
                    f"Found {len(self.report.multiple_mappings)} multiple mapping group(s). "
                    f"These may be legitimate linguistic variations and were NOT removed."
                )
        
        # Summary warnings
        if self.report.invalid_entries:
            self.report.warnings.append(
                f"Skipped {len(self.report.invalid_entries)} invalid entry(ies). "
                f"Service will load {len(valid_entries)} valid entries."
            )
        
        return self.report


def validate_sentence_data(entries: List[Dict[str, any]]) -> ValidationReport:
    """
    Convenience function to validate sentence entries.
    
    Args:
        entries: List of sentence entries to validate
        
    Returns:
        ValidationReport with findings
    """
    validator = SentenceValidator()
    return validator.validate_entries(entries)
