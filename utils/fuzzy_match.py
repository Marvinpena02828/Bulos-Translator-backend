"""
Fuzzy matching utilities using Levenshtein distance algorithm.

This module provides string similarity functions for typo-tolerant dictionary lookup.
Uses pure Python implementation to avoid external dependencies.
"""
from typing import Optional, Tuple


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    The Levenshtein distance is the minimum number of single-character edits
    (insertions, deletions, or substitutions) required to transform s1 into s2.
    
    Uses dynamic programming approach (Wagner-Fischer algorithm).
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Integer distance (0 = identical strings)
        
    Examples:
        >>> levenshtein_distance("ulo", "olu")  # 2 edits
        2
        >>> levenshtein_distance("mata", "maata")  # 1 insertion
        1
        >>> levenshtein_distance("kamay", "kamay")  # identical
        0
    """
    len1, len2 = len(s1), len(s2)
    
    # Handle empty strings
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1
    
    # Create distance matrix
    # matrix[i][j] = distance between s1[:i] and s2[:j]
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    
    # Initialize first column (distance from empty string)
    for i in range(len1 + 1):
        matrix[i][0] = i
    
    # Initialize first row (distance to empty string)
    for j in range(len2 + 1):
        matrix[0][j] = j
    
    # Fill in the rest of the matrix
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            # If characters match, no edit needed
            if s1[i - 1] == s2[j - 1]:
                cost = 0
            else:
                cost = 1  # substitution cost
            
            # Take minimum of:
            # - Delete from s1 (matrix[i-1][j] + 1)
            # - Insert into s1 (matrix[i][j-1] + 1)
            # - Substitute (matrix[i-1][j-1] + cost)
            matrix[i][j] = min(
                matrix[i - 1][j] + 1,      # deletion
                matrix[i][j - 1] + 1,      # insertion
                matrix[i - 1][j - 1] + cost  # substitution
            )
    
    return matrix[len1][len2]


def levenshtein_similarity(s1: str, s2: str) -> float:
    """
    Calculate normalized similarity ratio between two strings using Levenshtein distance.
    
    Similarity = 1 - (distance / max_length)
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Float in range [0.0, 1.0] where:
        - 1.0 = identical strings
        - 0.0 = completely different strings
        
    Examples:
        >>> levenshtein_similarity("ulo", "ulo")
        1.0
        >>> levenshtein_similarity("ulo", "olu")  # 2 edits out of 3 chars
        0.333...
        >>> levenshtein_similarity("mata", "maata")  # 1 edit out of 5 chars
        0.8
    """
    if not s1 and not s2:
        return 1.0  # both empty = identical
    
    distance = levenshtein_distance(s1, s2)
    max_len = max(len(s1), len(s2))
    
    if max_len == 0:
        return 1.0
    
    # Normalize: 1.0 = identical, 0.0 = completely different
    similarity = 1.0 - (distance / max_len)
    return similarity


def find_best_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.85,
    min_length: int = 3
) -> Optional[Tuple[str, float]]:
    """
    Find the best matching candidate string using fuzzy matching.
    
    Args:
        query: Input string to match
        candidates: List of candidate strings to match against
        threshold: Minimum similarity ratio (0.0-1.0) to accept match
        min_length: Minimum query length to perform fuzzy matching (avoid false positives on short words)
        
    Returns:
        Tuple of (best_match, similarity_score) or None if no match above threshold
        
    Examples:
        >>> candidates = ["ulo", "mata", "kamay"]
        >>> find_best_match("olu", candidates, threshold=0.85)
        None  # similarity too low
        >>> find_best_match("ulo", candidates, threshold=0.85)
        ("ulo", 1.0)
    """
    # Reject short queries to prevent false positives
    if len(query) < min_length:
        return None
    
    if not candidates:
        return None
    
    best_match = None
    best_score = 0.0
    
    for candidate in candidates:
        similarity = levenshtein_similarity(query, candidate)
        
        if similarity > best_score:
            best_score = similarity
            best_match = candidate
    
    # Reject if below threshold
    if best_score < threshold:
        return None
    
    return (best_match, best_score)
