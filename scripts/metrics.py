#!/usr/bin/env python3
# pylint: disable=line-too-long

"""
Reports metrics on the Poker Hand History (PHH) dataset.

Reads every .phhs file, or every Parquet hands part written by phh_to_parquet.py, and reports:
- game variants (NT, F7S, ...), bot games, and duplicate hands over all hands;
    - PHH only, Parquet doesn't retain the information needed for this
- players per hand
- street reached
- big blind sizes
- board cards and flop
- known hole cards

Hole cards are only known when dealt face up or shown down, so they lean toward hands players continue with.

Basic usage:
    ./scripts/metrics.py --data_path data
    ./scripts/metrics.py --data_path parquet
"""

# Disclaimer - used Chrome Gemma & Claude to help with the various custom pprint formats

# std lib
import re
import sys
from math import comb
from pathlib import Path
from hashlib import blake2b
from collections import Counter
from time import perf_counter
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

# external libs
import pyarrow.parquet as pq

##### Constants ####################

BATCH_SIZE = 16

# looking for single quotes wrapping any combination of chars
QUOTED_PATTERN = re.compile(r"'([^']*)'")

NLHE_VARIANT = "NT"
BOT_NAMES = frozenset({"Pluribus"}) # only bot set from the original dataset
STREETS = ("preflop", "flop", "turn", "river")

RANKS = "AKQJT98765432" # not ordered the same as phh_to_parquet, mostly for ease of use with the grid disp
SUITS = "cdhs"
DECK = [rank + suit for rank in RANKS for suit in SUITS]

HOLE_COMBOS = comb(52, 2)
FLOP_COMBOS = comb(52, 3)

# flop texture by # of distinct suits / ranks
# lines up with https://www.primedope.com/texas-holdem-poker-probabilities-odds/
#     (see `Board Texture`)
SUIT_TEXTURES = {
    3: ("rainbow", 4 * 13 ** 3 / FLOP_COMBOS),                          # 3 different suits | C D H
    2: ("two-tone", 1 - (4 * 13 ** 3 + 4 * comb(13, 3)) / FLOP_COMBOS), # suit pair + 1     | C C H
    1: ("monotone", 4 * comb(13, 3) / FLOP_COMBOS),                     # suit triplet      | C C C
}
RANK_TEXTURES = {
    3: ("unpaired", comb(13, 3) * 4 ** 3 / FLOP_COMBOS),                # 3 different ranks | A K Q
    2: ("paired", 13 * comb(4, 2) * 12 * 4 / FLOP_COMBOS),              # rank pair + 1     | A A Q
    1: ("trips", 13 * comb(4, 3) / FLOP_COMBOS),                        # 3 of a rank       | A A A
}

# see `Starting Hand Frequencies` from the above link
# exp combs for each
EXP_PAIR = 78
EXP_SUITED = 312
EXP_UNPAIRED_UNSUITED = 936

# defaults for when the PHH is missing elements
EMPTY_HAND = {
    "variant": "",
    "text": "", # the hand's full PHH text, for the dupe check; Parquet has none
    "players": [],
    "blinds_or_straddles": [],
    "starting_stacks": [],
    "actions": [],
}

##### Functions ####################
def get_args():
    """ Argparser retrieval """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type = Path, required = True)
    parser.add_argument("--workers", type = int, default = None)
    return parser.parse_args()


def parse_field(category, data):
    """ Resolves category cases for additional processing. """
    if category in ("actions", "players"):
        return QUOTED_PATTERN.findall(data)

    if category == "variant":
        return data.strip().strip("'")

    inner = data.strip()[1:-1].strip()
    return [float(segment) for segment in inner.split(",")] if inner else []


def walk_hands(path):
    """ Iteratively yields hands so long as game data can be extracted. """
    hand = None
    lines = [] # every line of the current hand, minus its [#] header

    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("["): # new data always prefixed with [#]
                if hand is not None:
                    hand["text"] = "".join(lines)
                    yield hand

                hand = dict(EMPTY_HAND) # stage defaults for a new game
                lines = []
                continue

            lines.append(line)
            category, _, data = line.rstrip("\n").partition(" = ") # ignoring separator

            if hand is not None and category in hand: # hand should be staged and filter out cats we don't care about
                hand[category] = parse_field(category, data)

    if hand is not None:
        hand["text"] = "".join(lines)
        yield hand


def walk_parquet_hands(path):
    """ Yields hands from a Parquet hands(!!) part """
    # NOTE: this is NOT optimal, much faster with Arrow kernels. Did not invest time into figuring that out for this metrics parser.
    #       I had Claude run a benchmark with that implementation in which it was taking ~15 seconds instead of the PHH 90-120.
    #       This still sees ~2x speed increase with the simplicity.

    # ref: https://arrow.apache.org/docs/python/generated/pyarrow.parquet.read_table.html
    #      see `Read only a subset of columns:`
    for row in pq.read_table(path, columns = ["blinds_or_straddles", "starting_stacks", "actions"]).to_pylist():
        # ** to unpack a dict into a dict, parquet doesn't contain the variant
        yield {**EMPTY_HAND, **row, "variant": NLHE_VARIANT}


def split_cards(board):
    """ Unrolls card groupings into a list like `Ah9c6d` -> ["Ah", "9c", "6d"] """
    return [board[i:i + 2] for i in range(0, len(board) - 1, 2)]


def get_hole_class(cards):
    """ Starting hand class for 2 hole cards like `KdAh`; e.g. AA, AKs, AKo """
    # using a/b instead of 1/2, not a huge fan of #s in var names...
    rank_a, suit_a, rank_b, suit_b = cards

    if RANKS.index(rank_a) > RANKS.index(rank_b): # normalize for higher rank first
        rank_a, rank_b = rank_b, rank_a # interpreter infers a buffer for tuple swaps

    if rank_a == rank_b:
        return rank_a + rank_b # clearly not suited... not even tagging

    return rank_a + rank_b + ("s" if suit_a == suit_b else "o")


def get_class_probability(hole_class):
    """ Chance of being dealt a starting hand class: 6 combos per pair, 4 per suited, 12 per offsuit """
    if len(hole_class) == 2:
        return 6 / HOLE_COMBOS
    return (4 if hole_class.endswith("s") else 12) / HOLE_COMBOS


def get_stat_defaults():
    """ Counts for one file or the total in main """
    return {
        "hands": 0,
        "variants": Counter(),
        "bot_hands": 0,
        "kept_hands": 0,
        "players": Counter(),
        "streets": Counter(),
        "big_blinds": Counter(),
        "board_cards": Counter(),
        "flop_suits": Counter(),
        "flop_ranks": Counter(),
        "hole_classes": Counter(),
    }


def collect_metrics(path):
    """ Collects stats for one PHH file or Parquet part. Runs in a worker process. """
    current_part_stats = get_stat_defaults()

    hand_keys = [] # for the dupe check in main

    for hand in walk_parquet_hands(path) if path.suffix == ".parquet" else walk_hands(path):
        boards = []
        holes = {}

        current_part_stats["hands"] += 1
        current_part_stats["variants"][hand["variant"]] += 1

        # no non-NLHE or bot games left in the dataset, just in case we want to revert that
        if hand["variant"] != NLHE_VARIANT:
            continue
        if BOT_NAMES.intersection(hand["players"]):
            current_part_stats["bot_hands"] += 1
            continue

        current_part_stats["kept_hands"] += 1
        current_part_stats["players"][len(hand["starting_stacks"])] += 1

        if hand["blinds_or_straddles"]:
            current_part_stats["big_blinds"][max(hand["blinds_or_straddles"])] += 1

        if hand["text"]:
            # Big assist from Claude here, was initially trying to match dupes using a combination of the venue
            # and hand ID. Turns out they aren't guaranteed to be unique hand IDs depending on the venue...
            # Using blake2b to hash each entire PHH entry into 8-byte fingerprints so we can load many/all simultaneously.
            fingerprint = blake2b(hand["text"].strip().encode(), digest_size = 8).digest()
            hand_keys.append(int.from_bytes(fingerprint, "big"))

        # split events like "d db Qd5hTd" -> ["d", "db", "Qd5hTd"]
        for event in (action.split() for action in hand["actions"]):
            if event[:2] == ["d", "db"] and len(event) > 2:
                boards.append(event[2])
            elif event[:2] == ["d", "dh"] and len(event) > 3 and "?" not in event[3]:
                holes[event[2]] = event[3]
            elif len(event) > 2 and event[1] == "sm" and "?" not in event[2]:
                holes[event[0]] = event[2]

        current_part_stats["streets"][STREETS[min(len(boards), 3)]] += 1

        for board in boards:
            current_part_stats["board_cards"].update(split_cards(board))

        if boards and len(boards[0]) == 6: # boards[0] should contain the flop
            flop = split_cards(boards[0])
            current_part_stats["flop_suits"][SUIT_TEXTURES[len({card[1] for card in flop})][0]] += 1
            current_part_stats["flop_ranks"][RANK_TEXTURES[len({card[0] for card in flop})][0]] += 1

        for cards in holes.values():
            if len(cards) == 4:
                current_part_stats["hole_classes"][get_hole_class(cards)] += 1

    return current_part_stats, hand_keys


def resolve_percentage(part, total, digits = 1):
    """ Generic percentage resolution, a lot of percentages here... """
    return f"{100 * part / total:.{digits}f}%" if total else "N/A"


def print_report(stats, duplicates, elapsed, phh):
    """ Prints every metric from the cumulative stats """

    kept = stats["kept_hands"]

    # high level overview ##########
    print("\n" + "=" * 50)
    print(f"Scanned {stats['hands']:,} hands in {elapsed:.0f}s")
    if phh: # Parquet has no variants, players, or hand text
        print(f"Variants: {dict(stats['variants'].most_common())}")
        print(f"Bot games: {stats['bot_hands']:,}")
        print(f"Duplicates: {duplicates:,}")
    print(f"Kept human NLHE hands: {kept:,}")

    # player distros ##########
    print("\nPlayers per hand")
    for count, n in sorted(stats["players"].items()):
        print(f"  {count:>2}: {n:>12,}  {resolve_percentage(n, kept)}")

    # street/game progression levels ##########
    print("\nStreet reached")
    for street in STREETS:
        n = stats["streets"][street]
        print(f"  {street:<8}{n:>12,}  {resolve_percentage(n, kept)}")

    # big blind distros ##########
    print("\nTop 10 big blind sizes")
    for size, n in stats["big_blinds"].most_common(10):
        print(f"  {size:>8g}: {n:>12,}  {resolve_percentage(n, kept)}")

    # board cards ##########
    # board cards are never hidden, so each of the 52 cards should make up 1/52 of them
    cards = stats["board_cards"]
    total = sum(cards.values())
    print(f"\nBoard cards: {total:,}")

    if total:
        # find the least and most common cards
        low = DECK[0]
        high = DECK[0]
        for card in DECK:
            if cards[card] < cards[low]:
                low = card
            if cards[card] > cards[high]:
                high = card

        # observed / expected, 1.0 means the card showed up exactly as often as it should
        expected = total / 52
        low_ratio = cards[low] / expected
        high_ratio = cards[high] / expected
        print(f"  observed/expected min/max card: {low_ratio:.4f} ({low}) to {high_ratio:.4f} ({high})")

    # flop splits #####
    flops = sum(stats["flop_suits"].values())
    print(f"\nFlops: {flops:,}")
    for suits_or_ranks, textures in (("suits", SUIT_TEXTURES), ("ranks", RANK_TEXTURES)):
        parts = []
        observed = stats[f"flop_{suits_or_ranks}"]
        for name, probability in textures.values():
            parts.append(f"{name} {resolve_percentage(observed[name], flops, 2)} (exp {100 * probability:.2f}%)")
        print(f"  {suits_or_ranks}: " + "  ".join(parts))

    # hole cards ##########
    ## total
    classes = stats["hole_classes"]
    total = sum(classes.values())
    print(f"\nKnown hole cards: {total:,} pairs")
    if not total: # ratios below would divide by zero, running on a subset could have no known hole cards
        return

    ## pair/suit status
    pairs = sum(n for hole_class, n in classes.items() if len(hole_class) == 2)
    suited = sum(n for hole_class, n in classes.items() if hole_class.endswith("s"))
    print(f"  pairs {resolve_percentage(pairs, total, 2)} (exp {resolve_percentage(EXP_PAIR, HOLE_COMBOS, 2)})")
    print(f"  suited {resolve_percentage(suited, total, 2)} (exp {resolve_percentage(EXP_SUITED, HOLE_COMBOS, 2)})")
    print(f"  offsuit {resolve_percentage(total - pairs - suited, total, 2)} (exp {resolve_percentage(EXP_UNPAIRED_UNSUITED, HOLE_COMBOS, 2)})")

    ## pairing grid
    ### rows and columns both go A -> 2
    ### the diagonal is pairs (AA), above it is suited (AKs), below it is offsuit (AKo)
    grid = []
    for i_row, row_rank in enumerate(RANKS):
        row = []
        for i_col, col_rank in enumerate(RANKS):
            if i_row == i_col:
                hole_class = row_rank + col_rank
            elif i_row < i_col:
                hole_class = row_rank + col_rank + "s"
            else:
                hole_class = col_rank + row_rank + "o"
            row.append(hole_class)
        grid.append(row)

    # how many times more often each class showed up than a uniform deal predicts
    ratio = {}
    for row in grid:
        for hole_class in row:
            expected = total * get_class_probability(hole_class)
            ratio[hole_class] = classes[hole_class] / expected

    ranked = sorted(ratio, key = ratio.get, reverse = True)

    ### cherry-picked top 5 rankings from the grid
    over = [f"{hole_class} {ratio[hole_class]:.2f}x" for hole_class in ranked[:5]]
    under = [f"{hole_class} {ratio[hole_class]:.2f}x" for hole_class in ranked[-5:]]
    print("  most over-represented:  " + "  ".join(over))
    print("  most under-represented: " + "  ".join(under))

    ### full grid
    print("  observed/expected grid (upper right suited, lower left offsuit):")

    header = "     "
    for rank in RANKS:
        header += f"{rank:>6}"
    print(header)

    for rank, row in zip(RANKS, grid):
        grid_line = f"     {rank}"
        for hole_class in row:
            grid_line += f"{ratio[hole_class]:6.2f}"
        print(grid_line)


def main():
    """ Driver for scanning the PHH files or Parquet parts. See script __doc__ for elaboration """
    args = get_args()

    data_path = args.data_path.resolve()
    paths = sorted(data_path.rglob("*.phhs"))
    phh = True

    if not paths: # no .phhs files, check for parquets
        phh = False # flag for the display, certain elements aren't tracked in the parquets

        # ignoring the decisions table, not comprehensive enough for these metrics
        paths = sorted(path for path in data_path.rglob("*.parquet") if "decisions" not in path.relative_to(data_path).parts)
    print(f"Scanning {len(paths)} files under {args.data_path}")

    start_time = perf_counter()
    stats = get_stat_defaults()
    seen = set()
    duplicates = 0

    # see phh_to_parquet.py for a disclaimer on parallelism
    with ProcessPoolExecutor(max_workers = args.workers) as exe:
        # See `Example 1` for map() guide: https://www.geeksforgeeks.org/python/processpoolexecutor-class-in-python/
        for i, (file_stats, hand_keys) in enumerate(exe.map(collect_metrics, paths, chunksize = BATCH_SIZE)):
            for name, value in file_stats.items():
                if isinstance(value, Counter):
                    stats[name].update(value)
                else:
                    stats[name] += value

            for key in hand_keys:
                duplicates += key in seen
                seen.add(key)

            # carriage return to stay on the same line for progress reporting
            print(f"\r - {i + 1}/{len(paths)} files ({perf_counter() - start_time:.0f}s)", end="")

    print_report(stats, duplicates, perf_counter() - start_time, phh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
