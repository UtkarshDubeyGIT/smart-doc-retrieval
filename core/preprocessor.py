"""
Text preprocessing module: tokenization, stop word filtering, and stemming.
Self-contained without heavy external dependencies.
"""
import re
import string

# Standard English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", 
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", 
    "but", "by", "can", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", 
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", 
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", 
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", 
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", 
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", 
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", 
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", 
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", 
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", 
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", 
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", 
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", 
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", 
    "yourself", "yourselves"
}

class SimpleStemmer:
    """
    Lightweight Porter-like suffix stripper for basic stemming.
    Keeps dependencies minimal while standardizing variants (e.g. learning -> learn, systems -> system).
    """
    SUFFIXES = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"), ("ator", "ate"),
        ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"), ("ousness", "ous"),
        ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"), ("logi", "log"),
        ("ing", ""), ("ies", "y"), ("es", "e"), ("s", ""), ("ed", ""), ("ly", "")
    ]

    @classmethod
    def stem(cls, word: str) -> str:
        w = word.lower()
        if len(w) <= 3:
            return w
        for suffix, repl in cls.SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= 3:
                return w[:-len(suffix)] + repl
        return w

def preprocess_text(text: str, remove_stopwords: bool = True, apply_stemming: bool = True) -> list[str]:
    """
    Tokenizes, lowercases, cleans punctuation, optionally removes stopwords and stems tokens.
    """
    # Replace hyphens and slashes with spaces to separate compound words
    text = re.sub(r'[-/]', ' ', text.lower())
    # Extract alphanumeric tokens
    tokens = re.findall(r'\b[a-zA-Z0-9]+\b', text)
    
    processed = []
    for token in tokens:
        # Skip pure numeric noise or single characters
        if len(token) <= 1 and not token.isalnum():
            continue
        if remove_stopwords and token in STOP_WORDS:
            continue
        if apply_stemming:
            token = SimpleStemmer.stem(token)
        if token:
            processed.append(token)
    return processed
