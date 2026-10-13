import time
from typing import List, Dict, Optional, Tuple

# Common English vocabulary for fast prefix search & autocorrect suggestions
COMMON_VOCABULARY = [
    "A", "ABOUT", "AFTER", "AGAIN", "ALL", "ALSO", "AND", "ANY", "ARE", "AS", "AT",
    "BE", "BECAUSE", "BEFORE", "BEST", "BETTER", "BIG", "BOY", "BUS", "BUT", "BY",
    "CALL", "CAN", "CAR", "CAT", "CITY", "COLD", "COME", "COULD",
    "DAY", "DEAR", "DO", "DOG", "DOWN", "DRINK",
    "EAT", "EVEN", "EVERY",
    "FAMILY", "FAST", "FEEL", "FIND", "FIRST", "FOOD", "FOR", "FRIEND", "FROM",
    "GET", "GIRL", "GIVE", "GO", "GOOD", "GREAT",
    "HAVE", "HE", "HELLO", "HELP", "HER", "HERE", "HIM", "HIS", "HOME", "HOUSE", "HOW",
    "I", "IF", "IN", "INTO", "IS", "IT", "ITS",
    "JUST", "KEEP", "KNOW",
    "LAST", "LEARN", "LEAVE", "LET", "LIKE", "LITTLE", "LOOK", "LOVE",
    "MAKE", "MAN", "MANY", "ME", "MORE", "MORNING", "MOST", "MOTHER", "MUCH", "MY",
    "NAME", "NEAR", "NEED", "NEVER", "NEW", "NEXT", "NIGHT", "NO", "NOT", "NOW",
    "OF", "OFF", "OLD", "ON", "ONE", "ONLY", "OPEN", "OR", "OTHER", "OUR", "OUT", "OVER",
    "PEOPLE", "PLEASE", "PUT",
    "READ", "RIGHT", "ROOM", "RUN",
    "SAD", "SAY", "SEE", "SHE", "SHOULD", "SIGN", "SLOW", "SMALL", "SO", "SOME", "SORRY", "STAY", "STOP",
    "TAKE", "TALK", "TEACH", "TELL", "THANK", "THANKS", "THAT", "THE", "THEIR", "THEM", "THEN", "THERE", "THESE", "THEY", "THING", "THINK", "THIS", "TIME", "TO", "TODAY", "TOMORROW", "TOO",
    "UNDER", "UP", "US", "USE",
    "VERY",
    "WALK", "WANT", "WATER", "WAY", "WE", "WELL", "WENT", "WHAT", "WHEN", "WHERE", "WHICH", "WHO", "WHY", "WILL", "WITH", "WORD", "WORK", "WORLD", "WOULD",
    "YEAR", "YES", "YESTERDAY", "YOU", "YOUR"
]

class SignAggregator:
    """
    Real-Time Temporal Aggregator for Indian Sign Language (ISL).
    Transforms discrete frame-by-frame letter predictions into:
      1. Debounced confirmed letters
      2. Assembled words with predictive auto-completion
      3. Coherent full sentences with punctuation & capitalization
    """
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        consecutive_frames_threshold: int = 8,
        space_timeout_seconds: float = 2.0,
        enable_spellcheck: bool = True
    ):
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_threshold = consecutive_frames_threshold
        self.space_timeout_seconds = space_timeout_seconds
        self.enable_spellcheck = enable_spellcheck

        # State tracking
        self.last_candidate_letter: Optional[str] = None
        self.candidate_count: int = 0
        self.last_committed_letter: Optional[str] = None
        self.last_sign_timestamp: float = time.time()
        self.is_holding_letter: bool = False

        # Text buffers
        self.current_word: str = ""
        self.sentence_words: List[str] = []
        
        # Word suggestion trie/list
        self.vocab = sorted(COMMON_VOCABULARY)
        self.spellchecker = None
        if self.enable_spellcheck:
            try:
                from spellchecker import SpellChecker
                self.spellchecker = SpellChecker()
            except ImportError:
                self.spellchecker = None

    def process_frame_prediction(self, letter: str, confidence: float) -> Dict[str, any]:
        """
        Process a single frame prediction from the model.
        Returns the updated state including current word, sentence, and trigger flags.
        """
        now = time.time()
        letter = letter.upper().strip()
        new_letter_committed = False
        space_triggered = False

        # 1. Check for inactivity / automatic space timeout
        if self.current_word and (now - self.last_sign_timestamp > self.space_timeout_seconds):
            self.commit_current_word()
            space_triggered = True

        # 2. Check confidence threshold
        if confidence >= self.confidence_threshold:
            self.last_sign_timestamp = now

            if letter == self.last_candidate_letter:
                self.candidate_count += 1
            else:
                self.last_candidate_letter = letter
                self.candidate_count = 1
                self.is_holding_letter = False

            # If held for required consecutive frames and not already locked
            if self.candidate_count >= self.consecutive_frames_threshold and not self.is_holding_letter:
                self.current_word += letter
                self.last_committed_letter = letter
                self.is_holding_letter = True
                new_letter_committed = True
        else:
            # Low confidence or hand absent: release the letter lock
            if self.is_holding_letter:
                self.is_holding_letter = False
                self.candidate_count = 0
                self.last_candidate_letter = None

        suggestions = self.get_word_suggestions()
        full_sentence = self.get_full_sentence()

        return {
            "current_char": letter if confidence >= self.confidence_threshold else "",
            "confidence": float(confidence),
            "stability_ratio": min(1.0, self.candidate_count / self.consecutive_frames_threshold),
            "new_letter_committed": new_letter_committed,
            "committed_char": letter if new_letter_committed else None,
            "current_word": self.current_word,
            "suggestions": suggestions,
            "sentence_words": self.sentence_words,
            "full_sentence": full_sentence,
            "space_triggered": space_triggered
        }

    def commit_current_word(self):
        """Finalizes the current word, applies optional spellcheck, and moves it to sentence."""
        if not self.current_word:
            return

        word = self.current_word
        if self.spellchecker and len(word) > 2:
            try:
                corrected = self.spellchecker.correction(word.lower())
                if corrected:
                    word = corrected.upper()
            except Exception:
                pass

        self.sentence_words.append(word)
        self.current_word = ""
        self.last_candidate_letter = None
        self.candidate_count = 0
        self.is_holding_letter = False

    def backspace(self):
        """Deletes the last letter in current_word, or the last word in sentence."""
        if self.current_word:
            self.current_word = self.current_word[:-1]
        elif self.sentence_words:
            self.sentence_words.pop()
        self.is_holding_letter = False
        self.candidate_count = 0

    def select_suggestion(self, suggestion: str):
        """Accepts a suggested word completion."""
        self.current_word = suggestion.upper()
        self.commit_current_word()

    def clear(self):
        """Resets the entire session."""
        self.current_word = ""
        self.sentence_words = []
        self.last_candidate_letter = None
        self.candidate_count = 0
        self.is_holding_letter = False

    def get_word_suggestions(self, max_suggestions: int = 5) -> List[str]:
        """Provides prefix matching suggestions for the current word being finger-spelled."""
        if not self.current_word:
            return []

        prefix = self.current_word.upper()
        matches = [w for w in self.vocab if w.startswith(prefix) and w != prefix]
        return matches[:max_suggestions]

    def get_full_sentence(self) -> str:
        """Constructs a properly formatted sentence."""
        words = list(self.sentence_words)
        if self.current_word:
            words.append(self.current_word)
        if not words:
            return ""
        
        # Capitalize first word and join
        joined = " ".join(words).capitalize()
        return joined
