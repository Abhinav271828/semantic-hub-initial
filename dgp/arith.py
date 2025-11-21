from random import random
import numpy as np
import torch
import torch.nn.functional as F
from funcy import compose


class ArithGrammar:
    """
    Dataset of basic addition expressions (a + b = c), some in numerals (datatype 0) and some in text (datatype 1).
    """

    def __init__(
        self,
        replication=(2, 1),  # only works with D = 2
        seed: int = 42,
    ):
        """Initialize the ArithDataset.
        Only works with 2 datatypes.

        Args:
            replication (D, T): Specifies replication properties; there are D datatypes with probability softmax([0, -T, ..., -(D-1)T])
            num_iters (int, optional): The number of iterations to make in the training loop per epoch. Defaults to 1e6.
            max_sample_length (int, optional): The maximum length of a sequence. Defaults to 128.
            seed (int, optional): The random seed. Defaults to 42.
        """
        np.random.seed(seed)

        # Some setup details
        self.seed = seed

        # Define datatype distribution
        self.datatype_distribution = F.softmax(
            torch.arange(
                start=0,
                end=-replication[0] * replication[1],
                step=-replication[1],
                dtype=torch.float,
            ),
            dim=0,
        )
        self.num_types = replication[0]

        # Build vocabulary
        self.vocab, self.id_to_token_map, self.vocab_size = self._build_vocabulary()

        # Special tokens
        self.pad_token = "<pad>"
        self.pad_token_id = self.vocab["<pad>"]

    def _build_vocabulary(self):
        """Build the vocabulary for arithmetic dataset.

        Includes:
        - Digits 0-9
        - Words for numbers 0-1998
        - Arithmetic operators: +, =
        - Special tokens: <pad>, <bos>, <eos>

        Returns:
            vocab: Dictionary mapping tokens to IDs
            id_to_token_map: Dictionary mapping IDs to tokens
            vocab_size: Size of vocabulary
        """
        vocab = {}
        vocab_size = 0

        # Number words needed for 0-1998
        units = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = [
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        # Collect all unique tokens
        tokens = []

        # Add digits 0-9
        tokens.extend([str(i) for i in range(10)])

        # Add number words
        tokens.extend(units)
        tokens.extend(teens)
        tokens.extend(tens)  # Skip empty strings
        tokens.append("hundred")
        tokens.append("and")
        tokens.append("thousand")

        # Add arithmetic operators
        tokens.extend(["+", "="])

        # Add special tokens
        for special_token in ["<pad>", "<bos>", "<eos>"]:
            vocab[special_token] = vocab_size
            vocab_size += 1

        # Add all collected tokens to vocabulary
        for token in tokens:
            vocab[token] = vocab_size
            vocab_size += 1

        # Create inverse vocabulary
        id_to_token_map = {v: k for k, v in vocab.items()}
        self.held_out_inputs = [
            [vocab["<bos>"]] + [vocab[t] for t in self._number_to_words(i).split(" ")]
            for i in range(200, 300)
        ]

        return vocab, id_to_token_map, vocab_size

    @staticmethod
    def _words_to_number(s: str) -> str:
        w2n = {
            "zero": 0,
            "one": 1,
            "two": 2,
            "three": 3,
            "four": 4,
            "five": 5,
            "six": 6,
            "seven": 7,
            "eight": 8,
            "nine": 9,
            "ten": 10,
            "eleven": 11,
            "twelve": 12,
            "thirteen": 13,
            "fourteen": 14,
            "fifteen": 15,
            "sixteen": 16,
            "seventeen": 17,
            "eighteen": 18,
            "nineteen": 19,
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        n = 0
        copy = s
        try:
            if "thousand" in s:
                d1000, s = s.split("thousand")
                n += w2n[d1000.strip()] * 1000
                s = s.strip()

            if "hundred" in s:
                d100, s = s.split("hundred")
                n += w2n[d100.strip()] * 100
                s = s.strip()

            if "and" in s:
                s = s.split("and")[1].strip()

            if s != "":
                if " " in s:
                    d10, d1 = s.split()
                    n += w2n[d10.strip()] + w2n[d1.strip()]
                else:
                    n += w2n[s]

            return n
        except:
            return None

    @staticmethod
    def _number_to_words(n: int) -> str:
        """Convert a number (0-1998) to English words.

        Args:
            n: Integer between 0 and 1998

        Returns:
            String representation in words (lowercase, no punctuation)
        """
        if n == 0:
            return "zero"

        units = [
            "",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
        ]
        teens = [
            "ten",
            "eleven",
            "twelve",
            "thirteen",
            "fourteen",
            "fifteen",
            "sixteen",
            "seventeen",
            "eighteen",
            "nineteen",
        ]
        tens = [
            "",
            "",
            "twenty",
            "thirty",
            "forty",
            "fifty",
            "sixty",
            "seventy",
            "eighty",
            "ninety",
        ]

        def convert_below_thousand(num):
            if num == 0:
                return ""
            elif num < 10:
                return units[num]
            elif num < 20:
                return teens[num - 10]
            elif num < 100:
                result = tens[num // 10]
                if num % 10 != 0:
                    result += " " + units[num % 10]
                return result
            else:  # num < 1000
                result = units[num // 100] + " hundred"
                remainder = num % 100
                if remainder != 0:
                    result += " and " + convert_below_thousand(remainder)
                return result

        if n < 1000:
            return convert_below_thousand(n)
        else:  # 1000 <= n <= 1998
            result = convert_below_thousand(n // 1000) + " thousand"
            remainder = n % 1000
            if remainder != 0:
                if remainder < 100:
                    result += " and " + convert_below_thousand(remainder)
                else:
                    result += " " + convert_below_thousand(remainder)
            return result

    def generate_sample(self, dtype: int):
        """Generate a single arithmetic sample.

        Args:
            dtype: The datatype (0 for numerals, 1 for words) – if 2, then return both
            For languages where the unit of comparison is 'seq', this is the required signature.

        Returns:
            String representation of the arithmetic expression
        """
        import random

        # Generate two random numbers between 0 and 999
        a = random.randint(0, 999)
        b = random.randint(0, 999)
        c = a + b

        convert0 = compose(" ".join, str)
        convert1 = self._number_to_words
        if dtype == 0:
            return f"{convert0(a)} + {convert0(b)} = {convert0(c)}"
        if dtype == 1:
            return f"{convert1(a)} + {convert1(b)} = {convert1(c)}"
        if dtype == 2:
            return (
                f"{convert0(a)} + {convert0(b)} = {convert0(c)}",
                f"{convert1(a)} + {convert1(b)} = {convert1(c)}",
            )

    def is_held_out(self, sample):

        if sample[:3] == "two":
            return True

        return False

    def tokenize_sentence(self, sentence: str):
        """Tokenize a sentence.

        Args:
            sentence: The sentence to tokenize

        Returns:
            List of token indices
        """
        # Tokenize by splitting on spaces
        tokens = sentence.split(" ")

        # Convert tokens to indices
        token_indices = []
        for token in tokens:
            if token == "" or token == " ":
                continue
            else:
                token_indices.append(self.vocab[token])

        return token_indices

    def detokenize_sentence(self, token_indices) -> str:
        """Detokenize a sentence.

        Args:
            token_indices: The token indices to detokenize

        Returns:
            The detokenized sentence
        """
        # Convert indices to tokens
        tokens = [self.id_to_token_map[token.item()] for token in token_indices]

        # Join tokens with spaces
        sentence = " ".join(tokens)

        return sentence

    def check_validity(self, s):
        """Check if a string (without special tokens) is a valid sample (accurate addition) in either datatype
        Returns:
        validity: bool
        n_tokens: int
        """
        if not ("+" in s and "=" in s):
            return False

        num = any(x in s for x in "0123456789")
        alpha = any(x in s for x in "abcdefghijklmnopqrstuvwxyz")

        if num and alpha:
            return False

        if s.count("+") != 1 or s.count("=") != 1:
            return False

        if s.index("+") > s.index("="):
            return False

        try:
            if num:
                a = s.split("+")[0].replace(" ", "")
                b = s.split("+")[1].split("=")[0].replace(" ", "")
                c = s.split("=")[1].replace(" ", "")
                return int(a) + int(b) == int(c)

            if alpha:
                a = s.split("+")[0].strip()
                b = s.split("+")[1].split("=")[0].strip()
                c = s.split("=")[1].strip()
                return self._words_to_number(a) + self._words_to_number(
                    b
                ) == self._words_to_number(c)
        except ValueError:  # if it's not in the format a + b = c
            print("val err", s)
            return False
        except TypeError:  # if _words_to_number fails
            print("typ err", s)
            return False
