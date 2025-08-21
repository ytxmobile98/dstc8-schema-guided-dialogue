import glob
import json
import os
from os.path import basename, dirname, join, realpath, relpath


def extract_dialogue(item: dict) -> dict:
    """
    Extract each dialogue's intents and slots, to the following format:

        {
            "dialogue_id": "1_00000",
            "turns": [
                {
                    "text": "XXXXX",
                    "speaker": "XXXXX",
                    "intent": "XXXXX",
                    "slots": {
                        "XXXXX": ["XXXXX", "XXXXX"],
                        "YYYYY": ["YYYYY"]
                    }
                },
                {
                    "text": "XXXXX",
                    "speaker": "XXXXX",
                    "intent": "YYYYY",
                    "slots": {
                        "AAAAA": ["AAAAA", "AAAAA"],
                        "BBBBB": ["BBBBB"]
                    }
                }
            ]
        }

    Here we only extract the user's utterances and their corresponding intents
    and slots, skipping the system's ones.
    """

    dialog_id: str = item["dialogue_id"]
    turns: list[dict] = item["turns"]

    return {
        "dialogue_id": dialog_id,
        "turns": [process_turn(turn) for turn in turns]
    }


def process_turn(turn: dict) -> dict:
    """
    Process a single turn to extract its dialogue information.

    Each turn will be extracted into this format:

        {
            "text": "XXXXX",
            "speaker": "XXXXX",
            "intent": "YYYYY",
            "slots": {
                "AAAAA": ["AAAAA", "AAAAA"],
                "BBBBB": ["BBBBB"]
            }
        }

    Here we only extract the user's utterances and their corresponding intents
    and slots, skipping the system's ones.
    """

    utterance: str = turn["utterance"]
    speaker: str = turn["speaker"]
    frames: list[dict] = turn["frames"]

    def extract_user_intent_and_slots(frames: list[dict]) -> \
            tuple[str, dict[str, list[str]]]:

        intent: str = ""
        slots: dict[str, list[str]] = {}

        # extract intent
        for frame in reversed(frames):
            intent = frame.get("state", {}).get("active_intent", "")
            if intent:
                break

        # extract slot values of frames
        for frame in frames:
            slot_values = frame.get("state", {}).get("slot_values", {})
            for slot, values in slot_values.items():
                slots[slot] = slots.get(slot, []) + values

        return intent, slots

    def extract_system_slots(utterance: str, frames: list[dict]) -> \
            dict[str, list[str]]:

        slots: dict[str, list[str]] = {}

        # extract slots
        for frame in frames:
            slots_list: list[dict] = frame.get("slots", [])
            for item in slots_list:
                slot_name: str = item["slot"]
                slot_value: str = utterance[
                    item["start"]:item["exclusive_end"]]
                slots[slot_name] = slots.get(slot_name, []) + [slot_value]

        return slots

    def extract_intent_and_slots():
        nonlocal utterance
        nonlocal speaker
        nonlocal frames

        intent: str = ""
        slots: dict[str, list[str]] = {}

        if speaker == "USER":
            intent, slots = extract_user_intent_and_slots(frames)
        elif speaker == "SYSTEM":
            slots = extract_system_slots(utterance, frames)

        return intent, slots

    intent, slots = extract_intent_and_slots()

    return {
        "text": utterance,
        "speaker": speaker,
        "intent": intent,
        "slots": slots,
    }


def main():
    CURDIR = dirname(realpath(__file__))

    input_dirs = [
        join(CURDIR, 'train'),
        join(CURDIR, 'dev'),
        join(CURDIR, 'test'),
    ]
    output_dirs = list(map(
        lambda d: join(CURDIR, 'extracted', relpath(d, CURDIR)),
        input_dirs,
    ))

    def process_dialogue_file(input_file: str, output_file: str):
        with open(input_file, 'r') as f:
            dialogues = json.load(f)

        result = [
            extract_dialogue(dialogue) for dialogue in dialogues
        ]

        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    for input_dir, output_dir in zip(input_dirs, output_dirs):
        os.makedirs(output_dir, exist_ok=True)

        input_files = glob.glob(join(input_dir, 'dialogues_*.json'))
        output_files = list(map(
            lambda input_file: join(output_dir, basename(input_file)),
            input_files,
        ))
        for input_file, output_file in zip(input_files, output_files):
            process_dialogue_file(input_file, output_file)


if __name__ == "__main__":
    main()
