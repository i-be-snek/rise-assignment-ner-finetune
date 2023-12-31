from dataclasses import dataclass


@dataclass
class TagInfo:
    # https://huggingface.co/datasets/Babelscape/multinerd
    full_tagset = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-BIO": 9,
        "I-BIO": 10,
        "B-CEL": 11,
        "I-CEL": 12,
        "B-DIS": 13,
        "I-DIS": 14,
        "B-EVE": 15,
        "I-EVE": 16,
        "B-FOOD": 17,
        "I-FOOD": 18,
        "B-INST": 19,
        "I-INST": 20,
        "B-MEDIA": 21,
        "I-MEDIA": 22,
        "B-MYTH": 23,
        "I-MYTH": 24,
        "B-PLANT": 25,
        "I-PLANT": 26,
        "B-TIME": 27,
        "I-TIME": 28,
        "B-VEHI": 29,
        "I-VEHI": 30,
    }

    # PER, ORG, LOC, DIS, ANIM
    main_five = {
        "O": 0,
        "B-PER": 1,
        "I-PER": 2,
        "B-ORG": 3,
        "I-ORG": 4,
        "B-LOC": 5,
        "I-LOC": 6,
        "B-ANIM": 7,
        "I-ANIM": 8,
        "B-DIS": 13,
        "I-DIS": 14,
    }
