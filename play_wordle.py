"""Wordle player.

This Wordle player guesses words in a way that maximizes the entropy gain from
the guess.

While playing, you need to type the hints received from Wordle. The colors are
translated to characters according to this mapping:
    GRAY   -> .
    YELLOW -> ?
    GREEN  -> *

When playing for the first time, a matrix of all possible hints is computed and
saved to the disk. This may take a lot of time (depending on your hardware).

Command like example:
    $ python play_wordle.py
    > Guess 1: tares
    > Hint 1:  **...
    > Guess 2: kyli
    > Hint 2:  .?..?
    > Guess 3: acin
    > Hint 3:  ?..??
    > Break 4: tangy

    $ python play_wordle.py --evalute --report-progress
    > 100%|████████████████████████████| 12972/12972 [22:38<00:00,  9.55it/s]
    > 0:            0.00
    > 1: .          0.01
    > 2: .          0.23
    > 3: #.         12.55
    > 4: #####.     54.99
    > 5: ###.       27.81
    > 6: .          4.07
    > 7: .          0.34

Command line flags:
    --hard-mode: Solves the game in hard-mode, where the player must use all
        previous hints.
    --secret-word: Self-play with a pre-defined secret word. If "?" is provided
        then a secret word is selected randomly.
    --evaluate: Evaluate this agent by printing histogram of the number of
        rounds. This may take some time to compute. Usually used together with
        --report-progress.
"""

import argparse
import os
from typing import Optional, Sequence
import warnings
import warnings

import numba as nb
import numpy as np
try:
    import tqdm
except ImportError:
    tqdm = None

_WORD_LENGTH = 5
_PERFECT_MATCH = 242
_HINT_NOT_IN_ANY_SPOT = '.'
_HINT_WRONG_SPOT = '?'
_HINT_CORRECT_SPOT = '*'

CharType = nb.types.UnicodeCharSeq(1)
WordType = nb.types.unicode_type

warnings.filterwarnings("ignore")


@nb.njit([nb.uint8(WordType, WordType), nb.uint8(nb.types.UnicodeCharSeq(5), nb.types.UnicodeCharSeq(5))])
def compute_hint(guess: str, secret: str) -> int:
    """Computes the hint to be recieved for a guess given a secret.

    Args:
        guess: The guessed word.
        secret: The secret word.

    Returns:
        The hint encoded as a number. Hint is the vector of colors returned
        by the game.
    """
    if len(guess) != _WORD_LENGTH:
        raise ValueError('guess has the wrong length.')
    if len(secret) != _WORD_LENGTH:
        raise ValueError('secret has the wrong length.')
    char_counter = nb.typed.Dict.empty(key_type=CharType, value_type=nb.uint8)
    for i in range(_WORD_LENGTH):
        c = secret[i]
        if c not in char_counter:
            char_counter[c] = 1
        else:
            char_counter[c] += 1

    hint: nb.uint8 = 0
    for i in range(_WORD_LENGTH):
        if guess[i] == secret[i]:
            hint += 2 * 3 ** (_WORD_LENGTH - 1 - i)
            char_counter[guess[i]] -= 1
    for i in range(_WORD_LENGTH):
        if guess[i] != secret[i] and guess[i] in char_counter and char_counter[guess[i]] > 0:
            hint += 1 * 3 ** (_WORD_LENGTH - 1 - i)
            char_counter[guess[i]] -= 1
    return hint


@nb.njit([nb.uint8[:,:](WordType[:], WordType[:]),
          nb.uint8[:,:](nb.types.UnicodeCharSeq(5)[:], nb.types.UnicodeCharSeq(5)[:])],
         parallel=True)
def compute_all_hints(list_of_guess_words: np.ndarray,
                      list_of_secret_words: np.ndarray) -> np.ndarray:
    """Computes the hints of all guesses given and for all secrets.

    Args:
        list_of_guess_words: An array of guessed words.
        secrets: An array of secret words.

    Returns:
        A 2D array of hints. The first index is of the guessed word, and the
        second is of the secret word.
    """
    num_guess_words = len(list_of_guess_words)
    num_secret_words = len(list_of_secret_words)
    hints = np.zeros(
        shape=(num_guess_words, num_secret_words), dtype=np.uint8)
    for i in nb.prange(num_guess_words):
        guess = list_of_guess_words[i]
        for j in range(num_secret_words):
            secret = list_of_secret_words[j]
            hints[i, j] = compute_hint(guess, secret)
    return hints


@nb.njit(nb.float32(nb.uint8[:]))
def compute_entropy_gain(hints_of_guess: np.ndarray) -> float:
    """Computes the entropy gain from guessing a word.

    Args:
        hints_of_guess: A 1D array of the hints to be recieved by guessing
            a specific guess word, for remaining secret words.

    Returns:
        The entropy gain from guessing the guess word.
    """
    if hints_of_guess.size < 1:
        raise ValueError('arr should be non-empty.')
    counter = np.zeros(shape=(_PERFECT_MATCH + 1,), dtype=np.int32)
    for hint in hints_of_guess:
        counter[hint] += 1
    hist = counter.astype(np.float32)
    hist /= hints_of_guess.size
    entropy_gain = -(hist * np.log(hist.clip(1e-7, 1))).sum()
    if counter[_PERFECT_MATCH] > 0:  # Break ties by selecting feasible word
        entropy_gain += 1e-6
    return entropy_gain


@nb.njit(nb.float32[:](nb.uint8[:, :]), parallel=True)
def compute_entropy_gain_per_guess(hints: np.ndarray) -> np.ndarray:
    """Computes the entropy gain of all guess words.

    Args:
        hints: A 2D array of the hints for all guesses and for all the
            remaining secret words.

    Returns:
        A 1D array of the entropy gain per guess word.
    """

    num_guess_words = hints.shape[0]
    scores = np.zeros(shape=(num_guess_words,), dtype=np.float32)
    for i in nb.prange(num_guess_words):
        scores[i] = compute_entropy_gain(hints[i])
    return scores


@nb.njit(nb.float32(nb.uint8[:]))
def compute_max_size(hints_of_guess: np.ndarray) -> float:
    """Computes the size of the maximum remaining set of words.

    Args:
        hints_of_guess: A 1D array of the hints to be recieved by guessing
            a specific guess word, for remaining secret words.

    Returns:
        The size of the maximum set of remaining words after guessing the word.
    """
    if hints_of_guess.size < 1:
        raise ValueError('arr should be non-empty.')
    counter = np.zeros(shape=(_PERFECT_MATCH + 1,), dtype=np.int32)
    for hint in hints_of_guess:
        counter[hint] += 1
    return -float(np.max(counter))


@nb.njit(nb.float32[:](nb.uint8[:, :]), parallel=True)
def compute_max_size_per_guess(hints: np.ndarray) -> np.ndarray:
    """Computes the size of the maximum remaining set of all guess words.

    Args:
        hints: A 2D array of the hints for all guesses and for all the
            remaining secret words.

    Returns:
        A 1D array of the maximum set size per guess word.
    """

    num_guess_words = hints.shape[0]
    scores = np.zeros(shape=(num_guess_words,), dtype=np.float32)
    for i in nb.prange(num_guess_words):
        scores[i] = compute_max_size(hints[i])
    return scores


@nb.njit(nb.int32(nb.uint8[:, :], nb.boolean))
def find_best_guess(hints: np.ndarray, worst_case: bool = False) -> int:
    """Finds the guess word which maximizes entropy gain.

    Args:
        hints: A 2D array of the hints for all guesses and for all the
            remaining secret words.

    Returns:
        The index of the best word to guess, i.e. the guess word with maximum
        entropy gain.
    """
    if worst_case:
        scores = compute_max_size_per_guess(hints)
    else:
        scores = compute_entropy_gain_per_guess(hints)
    return np.argmax(scores)  # Maximize score


def encode_hint(hint_str: str) -> int:
    """Encodes hint string as an integer.

    The format is:
        ⬜ (GRAY)   -> .
        🟨 (YELLOW) -> ?
        🟩 (GREEN)  -> *

    Args:
        hint_str: A string representing the hint.

    Returns:
        An integer representing the hint.
    """
    if len(hint_str) != _WORD_LENGTH:
        raise ValueError('hint_str should have length of 5.')
    hint = 0
    for i in range(_WORD_LENGTH):
        hint *= 3
        if hint_str[i] == _HINT_NOT_IN_ANY_SPOT:
            hint += 0
        elif hint_str[i] == _HINT_WRONG_SPOT:
            hint += 1
        elif hint_str[i] == _HINT_CORRECT_SPOT:
            hint += 2
        else:
            raise ValueError('Unknown char')
    return hint


def decode_hint(hint: int) -> str:
    """Decodes integer hint as a string.

    The format is:
        ⬜ (GRAY)   -> .
        🟨 (YELLOW) -> ?
        🟩 (GREEN)  -> *

    Args:
        hint: An integer representing the hint.

    Returns:
        A string representing the hint.

    """
    hint_str = []
    for _ in range(_WORD_LENGTH):
        hint_chr = hint % 3
        hint //= 3
        if hint_chr == 0:
            hint_str.append(_HINT_NOT_IN_ANY_SPOT)
        elif hint_chr == 1:
            hint_str.append(_HINT_WRONG_SPOT)
        else:
            hint_str.append(_HINT_CORRECT_SPOT)
    return ''.join(hint_str[::-1])


def play(list_of_guess_words: Sequence[str],
         list_of_secret_words: Sequence[str],
         hints: Optional[np.ndarray] = None,
         secret_word: Optional[str] = None,
         hard_mode: bool = False,
         worst_case: bool = False):
    """Plays a single game of Wordle.

    Args:
        list_of_guess_words: List of words that can be guessed.
        list_of_secret_words: List of potential secret words.
        hints: Pre-computed matrix of hints of all guess words and all
            secret words.
        secret_word: Secret word for self-play. If provided, the computer
            will self-play against itself with that secret word.
        hard_mode: Whether to use hard-mode.
    """
    list_of_guess_words = np.asarray(list_of_guess_words)
    list_of_secret_words = np.asarray(list_of_secret_words)
    if hard_mode and np.any(list_of_guess_words != list_of_secret_words):
        list_of_words = set(list_of_guess_words) | set(list_of_secret_words)
        mask_of_guess_words = np.array([word in list_of_words 
                                        for word in list_of_guess_words],
                                       dtype=np.bool)
        mask_of_secret_words = np.array([word in list_of_words 
                                         for word in list_of_secret_words],
                                        dtype=np.bool)
        list_of_guess_words = list_of_guess_words[mask_of_guess_words]
        list_of_secret_words = list_of_secret_words[mask_of_secret_words]
        hints = hints[mask_of_guess_words, mask_of_secret_words]

    if hints is None:
        print('Compute hints.')
        hints = compute_all_hints(
            list_of_guess_words, list_of_secret_words)
    if hints.shape != (len(list_of_guess_words), len(list_of_secret_words)):
        raise ValueError('hints has a bad shape.')

    if secret_word is not None:
        if secret_word not in list_of_secret_words:
            raise ValueError(f'Provided secret word is not in the list: {secret_word}')
        print('Self-play mode. Secret:', secret_word)

    guess_number = 0
    hint = 0
    while hint != _PERFECT_MATCH:
        guess_number += 1
        if len(list_of_secret_words) <= 1:
            print(f'Break {guess_number}:', list_of_secret_words[0])
            break
        guess_index = find_best_guess(hints, worst_case)
        guess_word = list_of_guess_words[guess_index]
        print(f'Guess {guess_number}:', guess_word)

        if secret_word is not None:
            hint = compute_hint(guess_word, secret_word)
            print(f'Hint {guess_number}: ', decode_hint(hint))
        else:
            hint = encode_hint(input(f'Hint {guess_number}:  '))

        valid_words = hints[guess_index, :] == hint
        list_of_secret_words = list_of_secret_words[valid_words]
        hints = hints[:, valid_words]

        if hard_mode:
            list_of_guess_words = list_of_guess_words[valid_words]
            hints = hints[valid_words, :]

    print(
        f'Correctly guessed "{list_of_secret_words[0]}" in {guess_number} rounds!')


def _evalute(
        list_of_guess_words: Sequence[str],
        list_of_secret_words: Sequence[str],
        hints: np.ndarray,
        secret_word: str,
        hard_mode: bool = False,
        worst_case: bool = False) -> int:
    guess_number = 0
    hint = 0
    while hint != _PERFECT_MATCH:
        guess_number += 1
        if len(list_of_secret_words) <= 1:
            break
        guess_index = find_best_guess(hints, worst_case)
        guess_word = list_of_guess_words[guess_index]
        hint = compute_hint(guess_word, secret_word)
        valid_words = hints[guess_index, :] == hint
        list_of_secret_words = list_of_secret_words[valid_words]
        hints = hints[:, valid_words]
        if hard_mode:
            list_of_guess_words = list_of_guess_words[valid_words]
            hints = hints[valid_words, :]
    return guess_number



def evaluate(
        list_of_guess_words: Sequence[str],
        list_of_secret_words: Sequence[str],
        hints: Optional[np.ndarray] = None,
        hard_mode: bool = False,
        worst_case: bool = False,
        report_progress: bool = False) -> np.ndarray:
    list_of_guess_words = np.asarray(list_of_guess_words)
    list_of_secret_words = np.asarray(list_of_secret_words)
    if hard_mode and np.any(list_of_guess_words != list_of_secret_words):
        list_of_words = set(list_of_guess_words) | set(list_of_secret_words)
        mask_of_guess_words = np.array([word in list_of_words 
                                        for word in list_of_guess_words],
                                       dtype=np.bool)
        mask_of_secret_words = np.array([word in list_of_words 
                                         for word in list_of_secret_words],
                                        dtype=np.bool)
        list_of_guess_words = list_of_guess_words[mask_of_guess_words]
        list_of_secret_words = list_of_secret_words[mask_of_secret_words]
        hints = hints[mask_of_guess_words, mask_of_secret_words]

    if hints is None:
        print('Compute hints.')
        hints = compute_all_hints(
            list_of_guess_words, list_of_secret_words)
    if hints.shape != (len(list_of_guess_words), len(list_of_secret_words)):
        raise ValueError('hints has a bad shape.')

    counter = {}
    iterate_over_secret_words = list_of_secret_words
    if report_progress:
        iterate_over_secret_words = tqdm.tqdm(iterate_over_secret_words)
    for secret_word in iterate_over_secret_words:
        round_length = _evalute(list_of_guess_words, list_of_secret_words,
                                hints, secret_word, hard_mode, worst_case)
        counter[round_length] = counter.get(round_length, 0) + 1
    
    histogram = np.zeros(shape=(max(counter) + 1,), dtype=np.int32)
    for round_length in counter:
        histogram[round_length] = counter[round_length]

    return histogram


def display_histogram(histogram):
    probs = 100. * (histogram.astype(np.float32) / histogram.sum())
    fmt = '{i:%dd}: {bar:<10s} {p:.3f}%% ({h:d})' % (len(str(len(histogram) - 1)),)
    for i, (h, p) in enumerate(zip(histogram, probs)):
        bar = '#' * round(p / 10)
        if p > round(p / 10):
            bar += '.'
        print(fmt.format(i=i, bar=bar, p=p, h=h))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--words-file', default='./wordle-words.txt',
                        type=str, help='Path to list of words.')
    parser.add_argument('--precomputed-hints-file', default='./wordle-precomputed.npz',
                        type=str, help='Path to precomputed hints.')
    parser.add_argument('--hard-mode', action='store_true',
                        help='Whether to play in hard mode.')
    parser.add_argument('--worst-case', action='store_true',
                        help='Whether to play against adversarial player.')
    parser.add_argument('--secret-word', type=str,
                        help='Secret word for self-play.')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--report-progress', action='store_true')

    args = parser.parse_args()
    if not os.path.exists(args.precomputed_hints_file):
        print('Computing all hints. This may take a few minutes.')
        with open(args.words_file, 'r') as f:
            list_of_words = [word.strip()
                             for word in f.readlines() if word.strip()]
        list_of_words = np.asarray(list_of_words)
        hints = compute_all_hints(list_of_words, list_of_words)
        np.savez_compressed(args.precomputed_hints_file,
                            list_of_words=list_of_words, hints=hints)
    else:
        with np.load(args.precomputed_hints_file) as f:
            list_of_words = f['list_of_words']
            hints = f['hints']

    if args.evaluate:
        list_of_words = list_of_words
        hints = hints
        histogram = evaluate(list_of_words, list_of_words, hints,
                             hard_mode=args.hard_mode,
                             worst_case=args.worst_case,
                             report_progress=args.report_progress)
        display_histogram(histogram)
    
    else:
        if args.secret_word is not None and args.secret_word == '?':
            args.secret_word = np.random.choice(list_of_words)

        play(list_of_words, list_of_words, hints=hints,
             secret_word=args.secret_word, hard_mode=args.hard_mode,
             worst_case=args.worst_case)


if __name__ == '__main__':
    main()
