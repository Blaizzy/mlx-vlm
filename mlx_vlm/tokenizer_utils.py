"""

## Module `tokenizer_utils` Overview

The `tokenizer_utils` module provides classes and functions for tokenization and detokenization when working with sequence models, especially those trained for natural language processing (NLP) tasks. The module includes several core components such as tokenizer wrappers and different streaming detokenizer implementations that handle the conversion of token ids back to human-readable text.

### Key Components

- `StreamingDetokenizer`: An abstract base class defining the interface for all streaming detokenizers. It includes placeholders for the `reset`, `add_token`, and `finalize` methods, which must be implemented by subclasses.

- `NaiveStreamingDetokenizer`: A simple detokenizer implementation for models that have straightforward decoding logic. It inherits from `StreamingDetokenizer` and provides basic decoding capabilities by leveraging the tokenizer's `decode` method.

- `SPMStreamingDetokenizer`: A detokenizer specifically designed for SentencePiece models, handling space tokens (▁) and byte fallback. It supports optional space trimming and implements the methods of the `StreamingDetokenizer` for SentencePiece token handling.

- `BPEStreamingDetokenizer`: A detokenizer for Byte Pair Encoding (BPE) tokenizers, which reconstructs the original text from BPE tokens, including handling the BPE byte decoding logic. This class uses a byte decoder based on the scheme found in OpenAI's GPT-2 implementation.

- `TokenizerWrapper`: A wrapper class for tokenizers. It combines a tokenizer with an appropriate streaming detokenizer based on the tokenizer's configuration. It provides a transparent interface, allowing access to both tokenizer and detokenizer functionality.

- `load_tokenizer`: A function that loads a tokenizer from a given model path, determines the appropriate streaming detokenizer class based on the tokenizer's configuration, and returns a `TokenizerWrapper` instance. This function simplifies the process of setting up a tokenizer and its corresponding detokenizer.

### Helper Functions

- `_remove_space`: A helper function that removes a leading space from a string, if present.

- `_match`: A private utility function used to compare data structures for equivalence, useful for determining the type of decoder a tokenizer uses.

- `_is_spm_decoder`, `_is_spm_decoder_no_space`, `_is_bpe_decoder`: Functions that check whether the supplied decoder configuration matches expected patterns for SentencePiece or BPE decoders, respectively, which informs the selection of the appropriate detokenizer class in `load_tokenizer`.

Note that other internal class methods and properties not publically documented here may be present for the correct functioning of streaming detokenizers based on the specified tokenizer configurations.

### Summary

The `tokenizer_utils` module is essential for users working with different types of tokenizers in NLP models. It abstracts away the complexities of token decoding by providing a set of tools for streaming detokenization, wrapped within an easy-to-use interface.
"""

import json
from functools import partial

from transformers import AutoTokenizer

REPLACEMENT_CHAR = "\ufffd"


def _remove_space(x):
    """
    Removes the leading space from a given string if it exists.

    Args:
        x (str):
             The string from which to remove the leading space.

    Returns:
        (str):
             The string with the leading space removed if it was present,
            otherwise returns the original string unchanged.

    """
    if x and x[0] == " ":
        return x[1:]
    return x


class StreamingDetokenizer:
    """
    A class that incrementally detokenizes a sequence of tokens into a single string.
    This class is designed to accept tokens one by one, appending each token to
    an internal string buffer. The detokenization process can be interrupted and
    queried at any stage to get the current detokenized string segment. This is useful
    in streaming applications where tokens are received in a piecemeal fashion and
    immediate detokenization is necessary.

    Attributes:
        text (str):
             The current cumulative detokenized text formed from the added tokens.
        tokens (list):
             A list to hold the tokens that have been added so far. (It's
            not explicitly detailed here what kind of data structure this is or its
            purpose based on the given context).
        offset (int):
             The current position in the `text` from which the last segment
            begins. This is used to track the starting point of the most recent
            detokenized segment.

    Methods:
        reset():
             Abstract method to reset the detokenizer state.
        add_token(token):
             Abstract method to add a new token for detokenization.
            Given the abstract nature of the method, the parameter and the process
            of adding a token is not detailed here.
        finalize():
             Abstract method to perform any finalization steps required after
            detokenization.
        Properties:
        last_segment:
             Retrieves the last segment of detokenized text that follows after
            the offset. The segment is extracted from the current end of the `text`
            up to the current `offset`, which is then updated to point to the end of
            the `text`. If the `text` ends with REPLACEMENT_CHAR or is empty, an empty
            string is returned.

    """

    __slots__ = ("text", "tokens", "offset")

    def reset(self):
        """

        Raises a NotImplementedError indicating the method is not implemented and should be overridden in a subclass.

        """
        raise NotImplementedError()

    def add_token(self, token):
        """
        Adds a new token to an existing set of tokens.
        This method is intended to be implemented by subclasses. It should take a token
        and add it to the internal collection of tokens. The mechanism of storage and
        the data type of the token are determined by the specific subclass implementation.

        Parameters:
            token (Any):
                 The token to be added to the collection.

        Raises:
            NotImplementedError:
                 Always thrown since this is a placeholder meant to be
                overridden by subclasses.

        """
        raise NotImplementedError()

    def finalize(self):
        """

        Raises NotImplementedError to indicate that this function should be overridden by subclasses.
            This method is intended to be implemented by subclasses to perform finalization tasks, such as cleaning up
            resources or final computations. However, as provided, it does not perform any operations and
            immediately raises a NotImplementedError exception upon being called.

        Raises:
            NotImplementedError:
                 Always thrown when this method is called, to indicate that it should be
                implemented by a subclass.

        """
        raise NotImplementedError()

    @property
    def last_segment(self):
        """

        Returns the last segment of a text that hasn't been read yet.
            Hides the internal offset state and provides the unread segment of the text. If the end of the
            text is already reached or the current character is the REPLACEMENT_CHAR, it returns an empty string.

        Returns:
            (str):
                 The last unread segment of self.text if it exists and is not followed by the
                REPLACEMENT_CHAR; otherwise, returns an empty string.

        """
        text = self.text
        if text and text[-1] != REPLACEMENT_CHAR:
            segment = text[self.offset :]
            self.offset = len(text)
            return segment
        return ""


class NaiveStreamingDetokenizer(StreamingDetokenizer):
    """
    A simple streaming detokenizer class that integrates with a specified tokenizer to progressively reconstruct text from a stream of tokens.
    This class is a subclass of `StreamingDetokenizer` and is initialized with a tokenizer instance. It provides methods to add tokens to the current stream, finalize the construction of text from current tokens, reset the state for a new stream, and access the constructed text and list of tokens.

    Attributes:
        offset (int):
             The starting index within the stream for the next token operation.
        _tokens (List[str]):
             A list of token strings that have been added to the detokenizer so far.
        _text (str):
             The detokenized text reconstructed from the tokens.
        _current_tokens (List[str]):
             A temporary list to hold tokens that are being added but not yet finalized.
        _current_text (str):
             A buffer to hold the text corresponding to `_current_tokens`, if they haven't been integrated into `_text`.

    Methods:
        __init__(self, tokenizer):
             Initializes a new instance of NaiveStreamingDetokenizer with the given tokenizer object.
        reset(self):
             Resets the detokenizer state to handle a new stream of tokens, clearing any internal data structures.
        add_token(self, token):
             Adds a single token to the current token buffer (_current_tokens).
        finalize(self):
             Finalizes the current batch of tokens, extending the overall token list and reconstructing the corresponding text portion, then clearing the current token buffer.
        text(self):
             Property that returns the currently reconstructed text. If there are any unfinalized tokens, it decodes and appends them to the text before returning.
        tokens(self):
             Property that returns the list of all tokens added to the detokenizer.

    """

    def __init__(self, tokenizer):
        """
        Initializes the object with a tokenizer.

        Args:
            tokenizer:
                 The tokenizer that will be used for encoding and decoding tasks.

        Raises:
            ValueError:
                 If the tokenizer is not compatible or raises an error during
                initialization.

        """
        self._tokenizer = tokenizer
        self._tokenizer.decode([0])
        self.reset()

    def reset(self):
        """
        Resets the state of the tokenizer instance.
        This method reinitializes the internal state of the tokenizer instance by setting the offset to 0 and clearing all token-related lists and text representations. After calling this method, the tokenizer instance will be in the same state as it was immediately after instantiation.

        Attributes:
            offset (int):
                 The position offset within the text, reset to 0.
            _tokens (list):
                 Internal list of tokens, cleared.
            _text (str):
                 The original text being tokenized, cleared.
            _current_tokens (list):
                 List of tokens from the current operation, cleared.
            _current_text (str):
                 Text representation corresponding to the current tokens, cleared.

        """
        self.offset = 0
        self._tokens = []
        self._text = ""
        self._current_tokens = []
        self._current_text = ""

    def add_token(self, token):
        """
        Adds a token to the current token list.
        This method appends the given token to the `_current_items` list attribute
        of the object.

        Args:
            token (str):
                 The token to be added to the list.

        """
        self._current_tokens.append(token)

    def finalize(self):
        """
        Finalizes the processing of tokens and text.
        This method combines the current tokens with the already processed tokens, updates the internal text representation by decoding
        the current tokens, and then resets the current tokens and text to empty, preparing the system for the next set of inputs.

        Attributes:
            _tokens (list):
                 A private attribute holding the list of all processed tokens.
            _current_tokens (list):
                 A private attribute holding the list of current tokens to be processed.
            _text (str):
                 A private attribute representing the decoded text of all processed tokens.
            _current_text (str):
                 A private attribute representing the text of the current tokens before decoding.
            _tokenizer (Tokenizer):
                 A private attribute of a Tokenizer or similar object responsible for encoding and decoding tokens.

        Raises:
            AttributeError:
                 If the tokenizer is not set or any of the required attributes are missing.

        Note:
            The method assumes that the object has been properly initialized with the necessary attributes beforehand.

        """
        self._tokens.extend(self._current_tokens)
        self._text += self._tokenizer.decode(self._current_tokens)
        self._current_tokens = []
        self._current_text = ""

    @property
    def text(self):
        """
        Gets the concatenated text of the current tokens with proper handling of newline characters.
        This property method first checks if there are any tokens currently held. If there are, it decodes the current tokens into text using the tokenizer's decode method. It then checks if the last character of the current text is a newline character. If so, the current tokens are appended to a token buffer, and the current text is added to the text buffer. The current tokens and text are then cleared. Finally, the method returns the concatenated text from the text buffer and any partial current text.

        Returns:
            (str):
                 The concatenated text derived from the current tokens, including any partially processed text.

        """
        if self._current_tokens:
            self._current_text = self._tokenizer.decode(self._current_tokens)
        if self._current_text and self._current_text[-1] == "\n":
            self._tokens.extend(self._current_tokens)
            self._text += self._current_text
            self._current_tokens.clear()
            self._current_text = ""
        return self._text + self._current_text

    @property
    def tokens(self):
        """
        Gets the tokens attribute of the current instance.
        This property method provides read-only access to the '_tokens' attribute which holds
        a collection of tokens. These tokens could represent lexical tokens in a compiler,
        authentication tokens in a software application, or other relevant tokenized data as
        per the context of the instance usage.

        Returns:
            The '_tokens' attribute value of the current instance, which could be any type
            depending on how tokens are defined and stored within the class.

        """
        return self._tokens


class SPMStreamingDetokenizer(StreamingDetokenizer):
    """
    A streaming detokenizer for SentencePieceModel (SPM) encoded sentences.
    This class is responsible for converting token IDs back to their string representation.
    It efficiently handles streaming input, allowing tokens to be added progressively and
    the detokenized text to be recovered at any point.

    Attributes:
        trim_space (bool):
             If true, leading spaces will be trimmed from the detokenized text.
        tokenmap (list):
             A mapping from token IDs to their text representation.

    Methods:
        __init__(self, tokenizer, trim_space=True):
            Initializes a new instance of the SPMStreamingDetokenizer.

    Args:
        tokenizer:
             An instance of the tokenizer to be used for detokenizing.
        trim_space (bool, optional):
             Whether to trim leading spaces in the detokenized text. Defaults to True.
        reset(self):
            Resets the detokenizer state for a new input sequence.
        add_token(self, token):
            Adds a token to be detokenized.

    Args:
        token:
             The token ID to add.
        finalize(self):
            Finalizes detokenization processing and recovers the detokenized text.

    """

    def __init__(self, tokenizer, trim_space=True):
        """
        Initializes the object with a given tokenizer and sets the trimming behavior.

        Args:
            tokenizer (Tokenizer):
                 The tokenizer instance from which to extract the vocabulary.
            trim_space (bool, optional):
                 A flag indicating whether to trim white spaces in the token processing. Defaults to True.

        Raises:
            ValueError:
                 If the tokenizer provided does not have a 'vocab' attribute or is incorrectly structured.
                The initialization process involves creating a mapping of tokens from their ID representations to the actual text. It also replaces placeholders for non-printable bytes with their corresponding character values, and resets any internal state or counters the object may have.

        """
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        # Replace bytes with their value
        for i in range(len(self.tokenmap)):
            if self.tokenmap[i].startswith("<0x"):
                self.tokenmap[i] = chr(int(self.tokenmap[i][3:5], 16))

        self.reset()

    def reset(self):
        """
        Resets the internal state of the object.
        This method reinitializes various attributes of the object to their default state. It sets the 'offset' to 0, clears any '_unflushed' data, empties the 'text' string, and clears the list of 'tokens'.

        Attributes:
            offset (int):
                 The current offset position, reset to 0.
            _unflushed (str):
                 A buffer to store unflushed data, reset to an empty string.
            text (str):
                 The textual content, reset to an empty string.
            tokens (list):
                 A list of tokenized elements, reset to an empty list.

        Returns:
            None

        """
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        """
        Adds a given token to the current text, handling special cases for spacing.
        Arguments:
        token (str): The token that needs to be added to the text.

        Raises:
            KeyError:
                 If the token is not found in the token mapping (`self.tokenmap`).
                This method will update the current text based on the token's value and the existence of a leading space represented with a special character '▁'. If the token starts with this character, the method checks if there is existing text and if spaces are trimmed according to the flag `self.trim_space`. Based on these conditions, it will appropriately add spaces or remove any leading space. Any unflushed content will be merged to the current text with treatment for spacing characters.

        """
        v = self.tokenmap[token]
        if v[0] == "\u2581":
            if self.text or not self.trim_space:
                self.text += self._unflushed.replace("\u2581", " ")
            else:
                self.text = _remove_space(self._unflushed.replace("\u2581", " "))
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        """
        Performs the final text adjustments on the object by updating its text attribute.
        This method applies the necessary transformations to the _unflushed buffer, where it replaces any instances of '▁' (underscore-like character) with a space. If the `trim_space` attribute is not set or if there is already text in the `text` attribute, it appends the modified _unflushed buffer to the `text`. If `trim_space` is set and there is no existing text, it strips the leading space, if there is one, before assigning it to `text`. After processing, the _unflushed buffer is cleared.
        This method should be called once the text processing is complete and there is no more data to append to the text attribute.

        """
        if self.text or not self.trim_space:
            self.text += self._unflushed.replace("\u2581", " ")
        else:
            self.text = _remove_space(self._unflushed.replace("\u2581", " "))
        self._unflushed = ""


class BPEStreamingDetokenizer(StreamingDetokenizer):
    """
    A streaming detokenizer class for BPE (Byte Pair Encoding) encoded text.
    This class provides methods for adding tokens to a buffer, resetting the internal
    state, and finalizing the detokenization process to get the complete text output.
    In addition to the standard StreamingDetokenizer functionality, this class handles
    BPE-specific detokenization logic, such as byte decoding and space trimming.

    Attributes:
        trim_space (bool):
             Determines whether leading spaces in the detokenized output
            should be trimmed, defaulting to False.
        tokenmap (list):
             A list mapping token IDs to their textual representation,
            generated from the tokenizer's vocabulary.
        offset (int):
             A placeholder for future use.
        text (str):
             The accumulated detokenized text.
        tokens (list):
             A list of processed tokens (for potential future use).

    Methods:
        __init__(self, tokenizer, trim_space=False):
             Initializes a new instance of the
            BPEStreamingDetokenizer with the specified tokenizer and optional space
            trimming behavior.
        reset(self):
             Resets the detokenizer state, including flushing any buffered
            text and resetting metadata.
        add_token(self, token):
             Processes the given token, managing the buffer and
            adding appropriately decoded characters to the detokenized text.
        finalize(self):
             Finalizes the detokenization process by flushing any
            remaining buffered tokens and completing the detokenized text output.
        make_byte_decoder(cls):
             A class method that creates a byte decoder map
            used for BPE byte decoding. It's populated with a set of characters mapped
            from specific byte values, based on predetermined character ranges.

    """

    _byte_decoder = None

    def __init__(self, tokenizer, trim_space=False):
        """
        Initializes a new instance of the class with a given tokenizer and an optional space trimming setting.
        This method sets up various instance-specific properties. It assigns whether spaces should be trimmed based on the given argument. It then constructs a mapping from token IDs to their associated text representation by utilizing the tokenizer's vocabulary. The method also calls a `reset` function to initialize or clear any state, followed by the instantiation of a byte decoder, which is typically used for byte pair encodes (BPE) decoding tasks.

        Args:
            tokenizer:
                 A tokenizer object with a `vocab` attribute containing the mapping of vocabulary words to token IDs.
            trim_space (bool, optional):
                 A flag indicating whether to trim spaces from the tokenized output. Defaults to False.

        Raises:
            May raise exceptions if the tokenizer does not have the expected `vocab` attribute or the `reset` or `make_byte_decoder` methods are not defined.

        """
        self.trim_space = trim_space

        # Extract the tokens in a list from id to text
        self.tokenmap = [None] * len(tokenizer.vocab)
        for value, tokenid in tokenizer.vocab.items():
            self.tokenmap[tokenid] = value

        self.reset()

        # Make the BPE byte decoder from
        # https://github.com/openai/gpt-2/blob/master/src/encoder.py
        self.make_byte_decoder()

    def reset(self):
        """
        Resets the internal state of the object.
        This method reinitializes the `offset` to 0, clears any unflushed content, empties the `text` attribute, and clears the list of `tokens`. This is typically used to prepare the object for a new operation or to clear its state between uses.

        Attributes modified:
            offset (int):
                 A resettable value initialized to 0, indicating the starting position for some operation.
            _unflushed (str):
                 A buffer to hold unflushed content, reset to an empty string.
            text (str):
                 A container for text content, reset to an empty string.
            tokens (list):
                 A list to store tokens, reset to an empty list.

        """
        self.offset = 0
        self._unflushed = ""
        self.text = ""
        self.tokens = []

    def add_token(self, token):
        """
        Adds a token to the current text buffer, handling space characters accordingly.
        This method adds the specified token's bytes to the text buffer, updating the unflushed byte sequence. If the token begins with a space and the current text buffer already contains text, or if the 'trim_space' option is set to False, the space is preserved. Otherwise, leading spaces are removed when appending the token's text representation to the current text buffer.

        Args:
            token (str):
                 The token to be added to the buffer.

        Raises:
            KeyError:
                 If the token is not found in the tokenmap.

        """
        v = self.tokenmap[token]
        # if the token starts with space
        if self._byte_decoder[v[0]] == 32:
            current_text = bytearray(
                self._byte_decoder[c] for c in self._unflushed
            ).decode("utf-8")
            if self.text or not self.trim_space:
                self.text += current_text
            else:
                self.text += _remove_space(current_text)
            self._unflushed = v
        else:
            self._unflushed += v

    def finalize(self):
        """
        Performs the finalization process by decoding the accumulated unflushed bytes and appending the result to the text.
        This method decodes the unflushed byte array, which contains encoded characters, using the '_byte_decoder' mapping. The decoded string is then checked to determine whether it should be directly appended to 'self.text' or if leading whitespace should be removed before concatenation. This process is influenced by the state of 'self.trim_space'.
        If 'self.trim_space' is False, or if 'self.text' already contains content, the resulting string from decoding is added to 'self.text' as is. Conversely, if 'self.text' is empty and 'self.trim_space' is True, the '_remove_space' function is called to strip a single leading space from the decoded string prior сохранения в 'self.text'.  After the finalization, the '_unflushed' attribute is cleared.

        Raises:
            UnicodeDecodeError:
                 If the unflushed bytes cannot be decoded into a valid UTF-8 string.

        """
        current_text = bytearray(self._byte_decoder[c] for c in self._unflushed).decode(
            "utf-8"
        )
        if self.text or not self.trim_space:
            self.text += current_text
        else:
            self.text += _remove_space(current_text)
        self._unflushed = ""

    @classmethod
    def make_byte_decoder(cls):
        """
        Generates and assigns a byte decoder mapping for the class if not already created.
        The method checks if the class variable `_byte_decoder` is already set. If it is not,
        it creates a new mapping `char_to_bytes` which maps extended ASCII characters to their byte
        representations with a specific encoding scheme. Characters in the ASCII range
        from space (' ') to tilde ('~'), and from inverted exclamation mark ('¡') to
        logical negation symbol ('¬') and from registered trademark symbol ('®') to lowercase y with diaeresis
        ('ÿ') are mapped in non-sequential blocks. The method then assigns this mapping to the class
        variable `_byte byte_decoder`, enabling its reuse in subsequent method calls.

        Raises:
            None

        """
        if cls._byte_decoder is not None:
            return

        char_to_bytes = {}
        limits = [
            0,
            ord("!"),
            ord("~") + 1,
            ord("¡"),
            ord("¬") + 1,
            ord("®"),
            ord("ÿ") + 1,
        ]
        n = 0
        for i, (start, stop) in enumerate(zip(limits, limits[1:])):
            if i % 2 == 0:
                for b in range(start, stop):
                    char_to_bytes[chr(2**8 + n)] = b
                    n += 1
            else:
                for b in range(start, stop):
                    char_to_bytes[chr(b)] = b
        cls._byte_decoder = char_to_bytes


class TokenizerWrapper:
    """
    A wrapper class that combines both tokenizer and detokenizer functionality.
    This class provides a convenient way to access both the tokenizer and detokenizer within a single
    object. It delegates method calls to the underlying tokenizer object unless the attribute being
    accessed is 'detokenizer', in which case it returns the detokenizer object.

    Attributes:
        _tokenizer:
             An instance of a tokenizer class, which provides methods for tokenization.
        _detokenizer:
             An instance of a detokenizer class, which provides methods for detokenization.

    Args:
        tokenizer:
             An object which is responsible for tokenizing text.
        detokenizer_class:
             A class used for detokenizing tokens back into text. It defaults to
            NaiveStreamingDetokenizer if no class is provided.

    Raises:
        AttributeError:
             If an attribute is not found in the TokenizerWrapper's namespace, an AttributeError
            will be raised after an attempt is made to delegate the attribute lookup to the underlying
            tokenizer object.

    """

    def __init__(self, tokenizer, detokenizer_class=NaiveStreamingDetokenizer):
        """
        Constructor for initializing an object of the given class.

        Args:
            tokenizer:
                 The tokenizer instance used for tokenizing the input text.
            detokenizer_class:
                 The class responsible for detokenizing tokens back to text. Defaults to NaiveStreamingDetokenizer.
                This method is used to initialize an object with a tokenizer and a detokenizer instance. The detokenizer is instantiated with the provided tokenizer instance.

        """
        self._tokenizer = tokenizer
        self._detokenizer = detokenizer_class(tokenizer)

    def __getattr__(self, attr):
        """
        Fetches an attribute from the tokenizer or returns a specific attribute of the class.
        This method attempts to retrieve an attribute from the internal tokenizer object if it does not pertain to the
        self object directly. If the requested attribute is 'detokenizer', it returns the '_detokenizer' attribute
        of self. Otherwise, it falls back to the internal tokenizer object's attributes.

        Args:
            attr (str):
                 The name of the attribute to retrieve.

        Returns:
            (Any):
                 The attribute value from the tokenizer or the specified attribute from self.

        Raises:
            AttributeError:
                 If the attribute is not found in either self or the tokenizer object.

        """
        if attr == "detokenizer":
            return self._detokenizer
        else:
            return getattr(self._tokenizer, attr)


def _match(a, b):
    """
    Compares two objects, 'a' and 'b', to determine if they are matching in structure and content.
    This recursive function performs a deep comparison between the two provided objects.
    If both objects are dictionaries, it checks that they have the same keys, and that corresponding values match.
    If both are lists, it ensures they have the same length and that items at the same positions match.
    For other data types, it directly compares the values.

    Parameters:
        a (any):
             The first object to be compared.
        b (any):
             The second object to match against the first one.

    Returns:
        (bool):
             True if both objects match in type, structure, and content, False otherwise.

    Raises:
        TypeError if the objects are not comparable.

    """
    if type(a) != type(b):
        return False
    if isinstance(a, dict):
        return len(a) == len(b) and all(k in b and _match(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_match(ai, bi) for ai, bi in zip(a, b))

    return a == b


def _is_spm_decoder(decoder):
    """
    Determines if the given decoder configuration matches the expected configuration for a sentence piece model (SPM) decoder.

    Args:
        decoder (dict):
             The decoder configuration which is to be checked against the SPM criteria.

    Returns:
        (bool):
             True if the input decoder matches the predefined SPM decoder structure, False otherwise.
            The function specifically looks for a sequence of decoders that include a 'Replace' for introducing spaces, followed by a 'ByteFallback', a 'Fuse' operation, and finally a 'Strip' operation that removes leading spaces. This sequence of operations is indicative of an SPM decoder configuration.

    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
            {"type": "Strip", "content": " ", "start": 1, "stop": 0},
        ],
    }
    return _match(_target_description, decoder)


def _is_spm_decoder_no_space(decoder):
    """
    Determines whether a given decoder configuration matches the specific pattern of an SPM (Sentence Piece Model) decoder that does not include a space character after decoding.

    Args:
        decoder (dict):
             A dictionary representing the decoder configuration.

    Returns:
        (bool):
             True if the decoder configuration matches the SPM decoder pattern without a space character, False otherwise.
            The function compares the given `decoder` dictionary against a pre-defined pattern that represents an SPM decoder with no spaces post-decoding. It checks for exact pattern matching in terms of structure and sequence of operations. The expected pattern involves a sequence decoder with specific nested decoders, including a 'Replace' type that targets the space marker '▁' (used in SPM to denote space), removing this marker by replacing it with an empty space. Subsequent `ByteFallback` and `Fuse` type decoders are also part of the pattern. The function makes use of a helper `_match` function to perform a recursive deep comparison.

    """
    _target_description = {
        "type": "Sequence",
        "decoders": [
            {"type": "Replace", "pattern": {"String": "▁"}, "content": " "},
            {"type": "ByteFallback"},
            {"type": "Fuse"},
        ],
    }
    return _match(_target_description, decoder)


def _is_bpe_decoder(decoder):
    """
    Determines if a decoder configuration matches the target Byte Pair Encoding (BPE) decoder specification.
    Checks if the input `decoder` dictionary has the exact same structure and content as the expected BPE
    decoder specification. The function considers the BPE decoder properties such as 'type',
    'add_prefix_space', 'trim_offsets', and 'use_regex', ensuring they exactly match the target
    BPE decoder description. If the `decoder` matches the target BPE description, the
    function returns True, indicating it is a BPE decoder. Otherwise, it returns False.

    Args:
        decoder (dict):
             A dictionary containing the decoder configuration to be evaluated.

    Returns:
        (bool):
             True if the `decoder` matches the expected BPE decoder description, False otherwise.

    """
    _target_description = {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": False,
    }

    return _match(_target_description, decoder)


def load_tokenizer(model_path, return_tokenizer=True, tokenizer_config_extra={}):
    """
    Loads a tokenizer from the specified model path and returns either the tokenizer wrapped in TokenizerWrapper or the detokenizer class based on the provided arguments.

    Args:
        model_path (Path):
             The file system path to the directory containing tokenizer configuration and model files.
        return_tokenizer (bool):
             If True, return an instance of TokenizerWrapper; otherwise, return only the detokenizer class.
        tokenizer_config_extra (dict, optional):
             Additional configuration parameters to pass to the tokenizer. Defaults to an empty dictionary.

    Returns:
        (TokenizerWrapper/detokenizer_class):
             Depending on the value of return_tokenizer, returns either a TokenizerWrapper instance with the loaded tokenizer and respective detokenizer or only the detokenizer class.

    Raises:
        FileNotFoundException:
             If the tokenizer.json file does not exist in the model path.
        ValueError:
             If an invalid decoder configuration is detected in tokenizer.json.

    """
    detokenizer_class = NaiveStreamingDetokenizer

    tokenizer_file = model_path / "tokenizer.json"
    if tokenizer_file.exists():
        tokenizer_content = json.load(tokenizer_file.open())
        if "decoder" in tokenizer_content:
            if _is_spm_decoder(tokenizer_content["decoder"]):
                detokenizer_class = SPMStreamingDetokenizer
            elif _is_spm_decoder_no_space(tokenizer_content["decoder"]):
                detokenizer_class = partial(SPMStreamingDetokenizer, trim_space=False)
            elif _is_bpe_decoder(tokenizer_content["decoder"]):
                detokenizer_class = BPEStreamingDetokenizer

    if return_tokenizer:
        return TokenizerWrapper(
            AutoTokenizer.from_pretrained(model_path, **tokenizer_config_extra),
            detokenizer_class,
        )
    else:
        return detokenizer_class
