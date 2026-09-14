"""
Tests for Phase 2.1: Sentence Validation & Safety
==================================================
Tests validation layer WITHOUT using real SENTENCES dataset.
Uses synthetic test data only.
"""
import pytest
from utils.sentence_validator import SentenceValidator, validate_sentence_data


class TestBasicValidation:
    """Test basic entry validation"""
    
    def test_completely_valid_entry(self):
        """Valid entry with all required fields should pass"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "FILIPINO": "Ano ang pangalan mo?",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is True
        assert issue is None
    
    def test_missing_bulos(self):
        """Entry missing BULOS field should fail"""
        validator = SentenceValidator()
        entry = {
            "FILIPINO": "Ano ang pangalan mo?",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue is not None
        assert issue.issue_type == "MISSING_FIELD"
        assert issue.field_name == "BULOS"
    
    def test_missing_filipino(self):
        """Entry missing FILIPINO field should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue is not None
        assert issue.issue_type == "MISSING_FIELD"
        assert issue.field_name == "FILIPINO"
    
    def test_missing_english(self):
        """Entry missing ENGLISH field should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "FILIPINO": "Ano ang pangalan mo?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue is not None
        assert issue.issue_type == "MISSING_FIELD"
        assert issue.field_name == "ENGLISH"
    
    def test_empty_bulos_value(self):
        """Entry with empty BULOS value should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "",
            "FILIPINO": "Ano ang pangalan mo?",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue.issue_type == "EMPTY_VALUE"
        assert issue.field_name == "BULOS"
    
    def test_whitespace_only_filipino(self):
        """Entry with whitespace-only FILIPINO should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "FILIPINO": "   \t  \n  ",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue.issue_type == "WHITESPACE_ONLY"
        assert issue.field_name == "FILIPINO"
    
    def test_non_string_value(self):
        """Entry with non-string value should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "FILIPINO": 12345,  # numeric instead of string
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue.issue_type == "INVALID_TYPE"
        assert issue.field_name == "FILIPINO"
    
    def test_placeholder_value_na(self):
        """Entry with placeholder 'N/A' should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "Anu i ngalan mo?",
            "FILIPINO": "N/A",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue.issue_type == "PLACEHOLDER_VALUE"
        assert issue.field_name == "FILIPINO"
    
    def test_placeholder_value_tbd(self):
        """Entry with placeholder 'TBD' should fail"""
        validator = SentenceValidator()
        entry = {
            "BULOS": "TBD",
            "FILIPINO": "Ano ang pangalan mo?",
            "ENGLISH": "What is your name?"
        }
        
        is_valid, issue = validator.validate_entry(entry, 0)
        
        assert is_valid is False
        assert issue.issue_type == "PLACEHOLDER_VALUE"


class TestExactDuplicateDetection:
    """Test exact duplicate detection"""
    
    def test_exact_duplicate_detection(self):
        """Should detect exact duplicates with all three fields identical"""
        entries = [
            {
                "BULOS": "Anu i ngalan mo?",
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "Dênu ka paagaw?",
                "FILIPINO": "Saan ka pupunta?",
                "ENGLISH": "Where are you going?"
            },
            {
                "BULOS": "Anu i ngalan mo?",  # Exact duplicate of entry 0
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        assert len(report.exact_duplicates) == 1
        assert len(report.exact_duplicates[0]['indexes']) == 2
        assert 0 in report.exact_duplicates[0]['indexes']
        assert 2 in report.exact_duplicates[0]['indexes']
    
    def test_duplicate_detection_with_accent_normalization(self):
        """Should detect duplicates even with different accents"""
        entries = [
            {
                "BULOS": "Ànu i ngalan mo?",  # With grave accent
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "Anu i ngalan mo?",  # No accent (should be detected as duplicate)
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Should detect as duplicate (normalization removes accents)
        assert len(report.exact_duplicates) == 1
    
    def test_no_duplicates(self):
        """Should not report duplicates when all entries are unique"""
        entries = [
            {
                "BULOS": "Anu i ngalan mo?",
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "Dênu ka paagaw?",
                "FILIPINO": "Saan ka pupunta?",
                "ENGLISH": "Where are you going?"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        assert len(report.exact_duplicates) == 0


class TestMultipleMappingDetection:
    """Test multiple/ambiguous mapping detection"""
    
    def test_same_bulos_different_translations(self):
        """Should detect when same BULOS maps to different FILIPINO/ENGLISH"""
        entries = [
            {
                "BULOS": "Namangan i Tirintin ni agěkat",
                "FILIPINO": "Kumain si Tirintin ng agakat",  # Past tense
                "ENGLISH": "Tirintin ate agēkat"
            },
            {
                "BULOS": "Namangan i Tirintin ni agěkat",  # Same BULOS
                "FILIPINO": "Kakain si Tirintin ng agakat",  # Future tense
                "ENGLISH": "Tirintin will eat agěkat"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Should detect as multiple mapping (NOT invalid)
        assert len(report.valid_entries) == 2  # Both are valid
        assert len(report.multiple_mappings) >= 1
        
        # Find the SAME_BULOS mapping
        bulos_mapping = [m for m in report.multiple_mappings if m['type'] == 'SAME_BULOS_DIFFERENT_TRANSLATIONS']
        assert len(bulos_mapping) == 1
        assert len(bulos_mapping[0]['entries']) == 2
    
    def test_same_filipino_different_bulos(self):
        """Should detect when same FILIPINO maps to different BULOS forms"""
        entries = [
            {
                "BULOS": "Hanga i beloy ni patod",  # Using "ni"
                "FILIPINO": "Malaki ang bahay ng lalaki",
                "ENGLISH": "The house of the man is big"
            },
            {
                "BULOS": "Hanga i beloy nun patod",  # Using "nun"
                "FILIPINO": "Malaki ang bahay ng lalaki",  # Same FILIPINO
                "ENGLISH": "The house of the man is big"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Should detect as multiple mapping (NOT invalid)
        assert len(report.valid_entries) == 2  # Both are valid
        assert len(report.multiple_mappings) >= 1
        
        # Find the SAME_FILIPINO mapping
        filipino_mapping = [m for m in report.multiple_mappings if m['type'] == 'SAME_FILIPINO_DIFFERENT_BULOS']
        assert len(filipino_mapping) == 1
    
    def test_same_english_different_bulos(self):
        """Should detect when same ENGLISH maps to different BULOS forms"""
        entries = [
            {
                "BULOS": "Anu i ngalan mu?",  # Using "mu"
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "Anu i ngalan mo?",  # Using "mo"
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"  # Same ENGLISH
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Should detect as multiple mapping (NOT invalid)
        assert len(report.valid_entries) == 2  # Both are valid
        assert len(report.multiple_mappings) >= 1
        
        # Find the SAME_ENGLISH mapping
        english_mapping = [m for m in report.multiple_mappings if m['type'] == 'SAME_ENGLISH_DIFFERENT_BULOS']
        assert len(english_mapping) == 1


class TestInvalidEntryHandling:
    """Test that invalid entries don't crash validation"""
    
    def test_valid_entries_load_when_invalid_exists(self):
        """Valid entries should still load when an invalid entry exists"""
        entries = [
            {
                "BULOS": "Anu i ngalan mo?",
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "",  # INVALID: empty BULOS
                "FILIPINO": "Saan ka pupunta?",
                "ENGLISH": "Where are you going?"
            },
            {
                "BULOS": "Magandang araw",
                "FILIPINO": "Magandang araw",
                "ENGLISH": "Good day"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Should have 2 valid entries (indexes 0 and 2)
        assert len(report.valid_entries) == 2
        assert len(report.invalid_entries) == 1
        assert report.valid_entries[0]["BULOS"] == "Anu i ngalan mo?"
        assert report.valid_entries[1]["BULOS"] == "Magandang araw"
    
    def test_invalid_entry_does_not_crash_initialization(self):
        """Invalid entry should not cause validation to crash"""
        entries = [
            {
                "BULOS": None,  # INVALID: None value
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            }
        ]
        
        # Should not raise exception
        report = validate_sentence_data(entries)
        
        assert len(report.invalid_entries) == 1
        assert len(report.valid_entries) == 0


class TestLinguisticVariationPreservation:
    """Test that legitimate variations are preserved"""
    
    def test_accent_variations_preserved(self):
        """Accent variations should be preserved as separate entries"""
        entries = [
            {
                "BULOS": "Ànu i ngalan mo?",  # With grave accent
                "FILIPINO": "Ano ang pangalan mo (pormal)?",
                "ENGLISH": "What is your name (formal)?"
            },
            {
                "BULOS": "Anu i ngalan mo?",  # Without accent
                "FILIPINO": "Ano ang pangalan mo (karaniwan)?",
                "ENGLISH": "What is your name (casual)?"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Both should be valid (different FILIPINO distinguishes them)
        assert len(report.valid_entries) == 2
        # Original text preserved (not modified)
        assert report.valid_entries[0]["BULOS"] == "Ànu i ngalan mo?"
        assert report.valid_entries[1]["BULOS"] == "Anu i ngalan mo?"
    
    def test_pronoun_variations_preserved(self):
        """Pronoun variations (mu vs mo) should be preserved"""
        entries = [
            {
                "BULOS": "Anu i ngalan mu?",  # "mu"
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "Anu i ngalan mo?",  # "mo"
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Both should be valid
        assert len(report.valid_entries) == 2
        # Original text preserved
        assert "mu" in report.valid_entries[0]["BULOS"]
        assert "mo" in report.valid_entries[1]["BULOS"]
        # Detected as multiple mapping (not error)
        assert len(report.multiple_mappings) >= 1
    
    def test_grammatical_marker_variations_preserved(self):
        """Grammatical marker variations (ni vs nun) should be preserved"""
        entries = [
            {
                "BULOS": "Hanga i beloy ni patod",  # "ni"
                "FILIPINO": "Malaki ang bahay ng lalaki",
                "ENGLISH": "The house of the man is big"
            },
            {
                "BULOS": "Hanga i beloy nun patod",  # "nun"
                "FILIPINO": "Malaki ang bahay ng lalaki",
                "ENGLISH": "The house of the man is big"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        # Both should be valid
        assert len(report.valid_entries) == 2
        # Original text preserved
        assert " ni " in report.valid_entries[0]["BULOS"]
        assert " nun " in report.valid_entries[1]["BULOS"]


class TestValidationReport:
    """Test validation report generation"""
    
    def test_report_summary_format(self):
        """Validation report should generate readable summary"""
        entries = [
            {
                "BULOS": "Anu i ngalan mo?",
                "FILIPINO": "Ano ang pangalan mo?",
                "ENGLISH": "What is your name?"
            },
            {
                "BULOS": "",  # Invalid
                "FILIPINO": "Test",
                "ENGLISH": "Test"
            }
        ]
        
        report = validate_sentence_data(entries)
        summary = report.summary()
        
        assert "SENTENCE VALIDATION REPORT" in summary
        assert "Total Entries Processed" in summary
        assert "Valid Entries" in summary
        assert "Invalid Entries" in summary
    
    def test_report_tracks_totals(self):
        """Report should track total, valid, and invalid counts"""
        entries = [
            {"BULOS": "Test1", "FILIPINO": "Test1", "ENGLISH": "Test1"},
            {"BULOS": "", "FILIPINO": "Test2", "ENGLISH": "Test2"},  # Invalid
            {"BULOS": "Test3", "FILIPINO": "Test3", "ENGLISH": "Test3"},
        ]
        
        report = validate_sentence_data(entries)
        
        assert report.total_entries == 3
        assert len(report.valid_entries) == 2
        assert len(report.invalid_entries) == 1


class TestEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_empty_entry_list(self):
        """Empty entry list should not crash"""
        report = validate_sentence_data([])
        
        assert report.total_entries == 0
        assert len(report.valid_entries) == 0
        assert len(report.warnings) > 0
    
    def test_very_long_text_preserved(self):
        """Very long text should be preserved without truncation"""
        long_text = "A" * 500  # 500 character string
        entries = [
            {
                "BULOS": long_text,
                "FILIPINO": "Test",
                "ENGLISH": "Test"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        assert len(report.valid_entries) == 1
        assert report.valid_entries[0]["BULOS"] == long_text  # Not truncated
    
    def test_special_characters_preserved(self):
        """Special characters and punctuation should be preserved"""
        entries = [
            {
                "BULOS": "Anu i ngalan mo?!",
                "FILIPINO": "Ano ang pangalan mo?!",
                "ENGLISH": "What is your name?!"
            }
        ]
        
        report = validate_sentence_data(entries)
        
        assert len(report.valid_entries) == 1
        assert report.valid_entries[0]["BULOS"] == "Anu i ngalan mo?!"
