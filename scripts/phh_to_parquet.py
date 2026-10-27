#!/usr/bin/env python3
# pylint: disable=line-too-long

"""
Converts the Poker Hand History (PHH) dataset to Parquet while leaving the PHH files untouched.

Expects the cleaned dataset: .phhs files of human no-limit Texas hold'em hands only.

Writes 2 tables:
- hands/
    contains 1 row per hand, essentially a direct transformation of the PHH data.
- decisions/
    one row per player event, outlines the sequence for training.

hands/ column layout:
    hand_id             | unique id for the hand
    source              | PHH path of the data, for debug
    block               | PHH location at the `source`, also for debug
### rest lines up with PHH definitions
    antes
    blinds_or_straddles
    min_bet
    starting_stacks
    actions
    winnings

decisions/ column layout:
    hand_id   | unique id for the hand
    street    | game stage - 0 preflop, 1 flop, 2 turn, 3 river
    seat      | PHH player index, 0 = p1 = small blind
    n_players | total players
    n_active  | remaining players
    big_blind | big blind chip value
    pot       | chips in the pot
    to_call   | chips the current player would need to call
    stack     | player's remaining chips
    max_bet   | largest bet on the current street
    n_raises  | # of bets/raises on the current street
    hole1-2   | 2 columns for hole cards, unknowns are -1
    flop1-3   | 3 columns for flop cards, unknowns are -1
    turn      | turn card, unknown is -1
    river     | river card, unknown is -1
    action    | player decision - 0 fold, 1 check/call, 2 raise
    amount    | tied to action, shows the chip value raised, called, or checked/folded

Card encoding:
    rank * 4 + suit
    - ranks 2 -> A as 0 -> 12 and suits c d h s as 0 -> 3.
        2c = 0  * 4 + 0 = 0
        2s = 0  * 4 + 3 = 3
        8d = 6  * 4 + 1 = 25
        As = 12 * 4 + 3 = 51

Basic usage:
    ./scripts/phh_to_parquet.py --data_path data
"""

# std lib
import re
import sys
from pathlib import Path
from itertools import groupby
from time import perf_counter
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor

# external libs
import pyarrow as pa
import pyarrow.parquet as pq

##### Constants ####################
# looking for single quotes wrapping any combination of chars
ACTION_EVENT_PATTERN = re.compile(r"'([^']*)'")

RANKS = "23456789TJQKA"
SUITS = "cdhs"

# 0 fold, 1 check/call, 2 raise
ACTIONS = {"f": 0, "cc": 1, "cbr": 2}

# for the hand_id; helps keep different parts separated, 1M is safe for our dataset
PART_INTERVAL = 1000000
FILES_PER_PART = 100

# defaults for when the PHH is missing elements
EMPTY_HAND = {
    "antes": [],
    "blinds_or_straddles": [],
    "min_bet": None,
    "starting_stacks": [],
    "actions": [],
    "winnings": None
}

##### Pyarrow Parquet table schemas ####################
# reference: https://arrow.apache.org/docs/python/generated/pyarrow.Schema.html
HANDS_PQ_SCHEMA = pa.schema([
    ("hand_id", pa.int64()),
    ("source", pa.string()),
    ("block", pa.int32()),
    ("antes", pa.list_(pa.float64())),
    ("blinds_or_straddles", pa.list_(pa.float64())),
    ("min_bet", pa.float64()),
    ("starting_stacks", pa.list_(pa.float64())),
    ("actions", pa.list_(pa.string())),
    ("winnings", pa.list_(pa.float64())),
])
DECISIONS_PQ_SCHEMA = pa.schema([
    ("hand_id", pa.int64()),
    ("street", pa.int8()),
    ("seat", pa.int8()),
    ("n_players", pa.int8()),
    ("n_active", pa.int8()),
    ("big_blind", pa.float32()),
    ("pot", pa.float32()),
    ("to_call", pa.float32()),
    ("stack", pa.float32()),
    ("max_bet", pa.float32()),
    ("n_raises", pa.int8()),
    ("hole1", pa.int8()),
    ("hole2", pa.int8()),
    ("flop1", pa.int8()),
    ("flop2", pa.int8()),
    ("flop3", pa.int8()),
    ("turn", pa.int8()),
    ("river", pa.int8()),
    ("action", pa.int8()),
    ("amount", pa.float32()),
])

##### Functions ####################
def get_args():
    """ Argparser retrieval """
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", type = Path, required = True)
    parser.add_argument("--out", type = Path, help = "output directory", default = "parquet")
    parser.add_argument("--workers", type = int, default = None)
    return parser.parse_args()


def parse_field(category, data):
    """ Resolves category cases for additional processing. """
    if category == "actions": # split each event action into a python list
        return ACTION_EVENT_PATTERN.findall(data)

    if category == "min_bet":
        return float(data)

    inner = data.strip()[1:-1].strip()
    return [float(token) for token in inner.split(",")] if inner else []


def walk_hands(path):
    """ Iteratively yields hands so long as game data can be extracted. """

    hand = None

    with open(path, encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.startswith("["): # new data always prefixed with [#]
                if hand is not None:
                    yield hand

                hand = dict(EMPTY_HAND) # stage defaults for a new game
                continue

            category, _, data = line.rstrip("\n").partition(" = ") # ignoring separator

            if hand is not None and category in hand: # hand should be staged and filter out cats we don't care about
                hand[category] = parse_field(category, data)

    if hand is not None:
        yield hand


def get_card_encoding(text):
    """ Card ids for a card string like `Ah9c6d` """
    if "?" in text: # unknowns
        return []
    return [RANKS.index(text[i]) * 4 + SUITS.index(text[i + 1]) for i in range(0, len(text), 2)]


def resolve_seat_index(player):
    """ PHH player label -> 0-based seat; "p3" -> 2 """
    return int(player[1:]) - 1


def hand_to_decisions(hand, hand_id):
    """ Unrolls one hand to "replay" and yield a decisions row per player action """
    # TODO: convert schema to dataclasses?
    stacks = list(hand["starting_stacks"])
    n_players = len(stacks)
    pot = 0.0
    bets = [0.0] * n_players
    folded = [False] * n_players
    board = []
    street = 0
    n_raises = 0

    # pad to one entry per player, PHH can leave trailing zeros off
    blinds = (hand["blinds_or_straddles"] + [0.0] * n_players)[:n_players]
    antes = (hand["antes"] + [0.0] * n_players)[:n_players]
    big_blind = max(blinds)

    # split events like "p2 cbr 30" -> ["p2", "cbr", "30"]
    events = [action.split() for action in hand["actions"]]

    # 2 player special case for blinds
    if n_players == 2:
        first_actor = next((resolve_seat_index(event[0]) for event in events if event[0] != "d"), 1)
        small, big = sorted(blinds)
        blinds = [big, small] if first_actor == 1 else [small, big]

    # hole cards that are known
    hole_cards = {}
    for event in events:
        if event[:2] == ["d", "dh"] and len(get_card_encoding(event[3])) == 2:
            hole_cards[resolve_seat_index(event[2])] = get_card_encoding(event[3])
        elif event[0] != "d" and event[1] == "sm" and len(event) > 2 and len(get_card_encoding(event[2])) == 2:
            hole_cards[resolve_seat_index(event[0])] = get_card_encoding(event[2])

    for seat in range(n_players):
        stacks[seat] -= antes[seat]
        pot += antes[seat]

        bets[seat] = min(blinds[seat], stacks[seat]) # can't post more than the stack, some hands are short
        stacks[seat] -= bets[seat]

    for event in events: # dealer events, update game states
        if event[0] == "d":
            if event[1] == "db": # street is done
                pot += sum(bets)
                bets = [0.0] * n_players
                board += get_card_encoding(event[2])
                street += 1
                n_raises = 0
            continue

        the_event = event[1]

        if the_event not in ACTIONS:
            continue # skippin actions that aren't player decisions

        seat = resolve_seat_index(event[0])
        to_call = min(max(bets) - bets[seat], stacks[seat]) # all-in calls can be short
        if the_event == "cbr":
            amount = float(event[2])
        elif the_event == "cc":
            amount = to_call
        else:
            amount = 0.0

        # matching the decisions pq schema
        yield (
            hand_id, street, seat, n_players, n_players - sum(folded),
            big_blind, pot + sum(bets), to_call, stacks[seat], max(bets),
            n_raises, *(hole_cards.get(seat, [-1, -1])),
            *(board + [-1] * (5 - len(board))), ACTIONS[the_event], amount
        )

        ## post-processing

        if the_event == "f":
            folded[seat] = True
            continue

        chips_in = to_call if the_event == "cc" else min(amount - bets[seat], stacks[seat])
        bets[seat] += chips_in
        stacks[seat] -= chips_in

        if the_event == "cbr":
            n_raises += 1


def write_to_parquet(rows, pq_schema, path):
    """ Updates the parquet files """
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = zip(*rows) if rows else [[] for _ in pq_schema]
    table = pa.Table.from_arrays([pa.array(list(col), type=field.type) for col, field in zip(columns, pq_schema)], schema=pq_schema)
    pq.write_table(table, path, compression="zstd") # zstd for higher compression for git


def convert_part(files, part_id, relative_path, data_path, out):
    """ Convert one group of PHH files into one hands part and one decisions part. """
    hand_rows = []
    decision_rows = []
    unreplayable = 0

    for path in map(Path, files): # relative_to only works with Path objects
        source = str(path.relative_to(data_path))

        for block, hand in enumerate(walk_hands(path)):
            # hands table resolution
            hand_id = part_id * PART_INTERVAL + len(hand_rows)

            # decisions table resolution
            try:
                rows = list(hand_to_decisions(hand, hand_id))
            except (ValueError, IndexError):
                print(f"Warning: {hand_id} was unreplayable, skipped entry")
                unreplayable += 1
                continue

            hand_rows.append((hand_id, source, block, *hand.values()))
            decision_rows += rows

    write_to_parquet(hand_rows, HANDS_PQ_SCHEMA, Path(out, "hands", relative_path))
    write_to_parquet(decision_rows, DECISIONS_PQ_SCHEMA, Path(out, "decisions", relative_path))

    return len(hand_rows), len(decision_rows), unreplayable


def main():
    """ Driver for converting PHH files into parquet tables. See script __doc__ for elaboration """
    args = get_args()

    data_path = args.data_path.resolve()

    if args.out.exists(): # don't want to unintentionally overwrite an existing parquet store
        sys.exit(f"{args.out} already exists; delete it or pick another --out")

    paths = sorted(data_path.rglob("*.phhs"))

    # separating parts for memory allocation and parallelism
    parts = []

    # groupby to organize phh parts by their parent directory
    # https://stackoverflow.com/questions/773/how-do-i-use-itertools-groupby
    for directory, group in groupby(paths, key = lambda path: path.parent):
        group = [str(path) for path in group]

        for i in range(0, len(group), FILES_PER_PART):
            # seeding each block part with a set of the source files and output pq
            parts.append((group[i:i + FILES_PER_PART], directory.relative_to(data_path) / f"part-{i // FILES_PER_PART:04d}.parquet"))

    print(f"Converting {len(paths)} files into {len(parts)} parts under {args.out}")
    start_time = perf_counter()
    hands = rows = unreplayable = 0

    # parallelism in Python: https://www.geeksforgeeks.org/python/processpoolexecutor-class-in-python/
    # not entirely sure how well this works across different versions of Python, may want to downgrade this
    #   to use concurrency instead of parallelism. I haven't tested bypassing the GIL much.
    with ProcessPoolExecutor(max_workers = args.workers) as exe:
        futures = [
            exe.submit(convert_part, files, part_id, relative_pq_path, data_path, args.out) for part_id, (files, relative_pq_path) in enumerate(parts)
        ]

        # progress reporting
        for done, future in enumerate(futures):
            part_hands, part_rows, part_unreplayable = future.result()
            hands += part_hands
            rows += part_rows
            unreplayable += part_unreplayable

            # carriage return to stay on the same line, pulled this style from a Chrome/Gemini recommendation
            print(f"\r - {done + 1}/{len(parts)} parts ({perf_counter() - start_time:.0f}s)", end="")

    # summary
    print(f"\nWrote {hands} hands and {rows} decisions to {args.out} in {perf_counter() - start_time:.0f}s")
    print(f"# skipped due to PHH structure issues: {unreplayable}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
