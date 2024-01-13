import json
import glob
import os
import random
import sys


HERE = os.path.dirname(os.path.abspath(__file__))

SGD_DATASET_CONFIG = {
    "dstc8_single_domain":
    {
        "train": range(1, 44),
        "dev": range(1, 8),
        "test": range(1, 12)
    },
    "dstc8_multi_domain":
    {
        "train": range(44, 128),
        "dev": range(8, 21),
        "test": range(12, 35)
    },
    "dstc8_all":
    {
        "train": range(1, 128),
        "dev": range(1, 21),
        "test": range(1, 35)
    }
}

INSTRUCTION_PROMPT = "Track the state of the slot and intent in the input dialogue."
INPUT_PROMPT = "So the state of the slot and intent in the input dialogue is\n"


def create_data(data_category, data_category_type, src_schema_filename, tag_filename, src_dialogue_filename_pattern, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    with open(tag_filename, "r") as f:
        tag_name = json.load(f)

    SRC_SLOT_TAG = tag_name["SRC_SLOT_TAG"]
    SRC_INTENT_TAG = tag_name["SRC_INTENT_TAG"]
    SRC_RANGE_START_TAG = tag_name["SRC_RANGE_START_TAG"]
    SRC_RANGE_END_TAG = tag_name["SRC_RANGE_END_TAG"]
    TGT_SLOT_TAG = tag_name["TGT_SLOT_TAG"]
    TGT_SEP_TAG = tag_name["TGT_SEP_TAG"]
    TGT_INTENT_TAG = tag_name["TGT_INTENT_TAG"]
    TGT_END_TAG = tag_name["TGT_END_TAG"]
    SPACE_TAG = tag_name["SPACE_TAG"]

    # set random seed
    random.seed(123)

    # データ読み込み
    # スキーマの読み込み
    with open(src_schema_filename, "r") as f:
        schema_data = json.load(f)
    print(f"len(schema_data): {len(schema_data)}")
    # 対話データの読み込み
    file_list = sorted(glob.glob(src_dialogue_filename_pattern))
    print(f"len(file_list): {len(file_list)}")

    for i in SGD_DATASET_CONFIG[data_category][data_category_type]:
        # データ整形
        file = file_list[i - 1]
        re_data = []

        with open(file, "r") as f:
            print(f"file: {file}")
            dialogue_data = json.load(f)

        for d in dialogue_data:
            dialogue_history = ""
            service_schemas = []
            for service in schema_data:
                for s in d["services"]:
                    if service["service_name"] == s:
                        service_schemas.append(service)
            if len(service_schemas) != len(d["services"]):
                print("Error: service_schema is not found.")
                break
            turn_cnt = 0
            for turn_idx, turn in enumerate(d["turns"]):
                # スキーマ情報の取得
                schema_slots_info = SRC_SLOT_TAG
                for service_schema in random.sample(service_schemas, len(service_schemas)):
                    for slot in random.sample(service_schema["slots"],
                                              len(service_schema["slots"])):  # MEMO:サービス関係なく全スロット,全インテントの順番をランダムにする必要はない(基本的にスキーマは混ざらないため)
                        schema_slots_info += slot["name"].lower() + "=" + slot["description"].lower()
                        if slot["is_categorical"]:
                            schema_slots_info += SRC_RANGE_START_TAG
                            schema_slots_info += ",".join([value.lower()
                                                           for value in random.sample(slot["possible_values"], len(slot["possible_values"]))])
                            schema_slots_info += SRC_RANGE_END_TAG
                        schema_slots_info += SPACE_TAG
                schema_intents_info = SRC_INTENT_TAG
                for service_schema in random.sample(service_schemas, len(service_schemas)):
                    for intent in random.sample(service_schema["intents"], len(service_schema["intents"])):
                        schema_intents_info += intent["name"].lower() + "=" + intent["description"].lower() + SPACE_TAG
                if turn["speaker"] == "USER":
                    turn_cnt += 1
                    # TODO:複数のスロット値を持つスロットに対応する(地理名，時間などの表記ゆれなのでしなくてもよい？)
                    # active_slot_info = TGT_SEP_TAG.join(random.sample([k.lower() + "=" + v[0].lower() for k, v in turn["frames"][0]["state"] # framesの0番目のみ
                    #                                                    ["slot_values"].items()], len(turn["frames"][0]["state"]["slot_values"])))
                    # frames(list)のすべての要素に対して処理を行う
                    # total_slot_values_num_in_frames = len([slot for frame in turn["frames"] for slot in frame["state"]["slot_values"].values()])
                    # active_slot_info = TGT_SEP_TAG.join(random.sample([k.lower() + "=" + v[0].lower() for frame in turn["frames"]
                    #                                                    for k, v in frame["state"]["slot_values"].items()], total_slot_values_num_in_frames))
                    # 次のターンのシステム発話のserviceを取得して，それに対応するframeを取得する
                    next_system_service = d["turns"][turn_idx + 1]["frames"][0]["service"]
                    tgt_frame = None
                    for frame in turn["frames"]:
                        if frame["service"] == next_system_service:
                            tgt_frame = frame
                            break
                    if tgt_frame is None:
                        print("Error: tgt_frame is None.")
                        sys.exit()
                    active_slot_info = TGT_SEP_TAG.join(random.sample(
                        [k.lower() + "=" + v[0].lower() for k, v in tgt_frame["state"]["slot_values"].items()], len(tgt_frame["state"]["slot_values"])))
                    # active_intent_info = turn["frames"][0]["state"]["active_intent"].lower() # framesの0番目のみ
                    # len_frames = len(turn["frames"])
                    # active_intent_info = TGT_INTENT_TAG.join(random.sample(
                    #     [frame["state"]["active_intent"].lower() for frame in turn["frames"]], len_frames))  # framesのすべてのintentをランダムに並び替えて結合
                    active_intent_info = tgt_frame["state"]["active_intent"].lower(
                    )
                    dialogue_history += "[user]" + turn["utterance"].lower()
                    re_data.append(
                        {
                            "dialogue_id": d["dialogue_id"],
                            "turn_id": turn_cnt,
                            "service_list": d["services"],
                            "output": TGT_SLOT_TAG + active_slot_info + TGT_INTENT_TAG + active_intent_info,
                            "instruction": INSTRUCTION_PROMPT,
                            "input": schema_slots_info + schema_intents_info + dialogue_history + "\n" + INPUT_PROMPT
                        }
                    )
                else:
                    dialogue_history += "[system]" + turn["utterance"].lower()
        out_filename = os.path.join(out_dir, os.path.splitext(os.path.basename(file))[0] + ".json")
        with open(out_filename, "w") as f:
            json.dump(re_data, f, indent=4)


def create_sncd_data_from_sgd_data():
    for domain_type, data_types in SGD_DATASET_CONFIG.items():
        for data_type in data_types.keys():
            DATA_CATEGORY = domain_type  # dstc8_single_domain, dstc8_multi_domain, dstc8_all
            DATA_CATEGORY_TYPE = data_type  # train, dev, test
            print(f"DATA_CATEGORY: {DATA_CATEGORY}")
            print(f"DATA_CATEGORY_TYPE: {DATA_CATEGORY_TYPE}")

            SRC_SCHEMA_FILENAME = os.path.join(
                "SGD", DATA_CATEGORY_TYPE, "schema.json")
            TAG_FILENAME = os.path.join(HERE, "tag_name_sncd.json")
            SRC_DIALOGUE_FILENAME_PATTERN = os.path.join(
                "SGD", DATA_CATEGORY_TYPE, "dialogues*.json")
            OUT_DIR = os.path.join(
                HERE, "SGD_preprocess", DATA_CATEGORY, DATA_CATEGORY_TYPE)
            create_data(DATA_CATEGORY, DATA_CATEGORY_TYPE,
                        SRC_SCHEMA_FILENAME, TAG_FILENAME, SRC_DIALOGUE_FILENAME_PATTERN, OUT_DIR)


def create_sncd_data_from_sgdX_data():
    # とりあえずtestデータのみ作成
    DATA_CATEGORY = "dstc8_all"  # dstc8_single_domain, dstc8_multi_domain, dstc8_all
    DATA_CATEGORY_TYPE = "test"  # train, dev, test
    DATA_VARIANT_TYPE_LIST = ["v1", "v2", "v3", "v4", "v5"]
    print(f"DATA_CATEGORY: {DATA_CATEGORY}")
    print(f"DATA_CATEGORY_TYPE: {DATA_CATEGORY_TYPE}")
    print(f"DATA_VARIANT_TYPE_LIST: {DATA_VARIANT_TYPE_LIST}")
    for data_variant_type in DATA_VARIANT_TYPE_LIST:
        print(f"DATA_VARIANT_TYPE: {data_variant_type}")
        SRC_SCHEMA_FILENAME = os.path.join("SGD", data_variant_type, DATA_CATEGORY_TYPE, "schema.json")
        TAG_FILENAME = os.path.join(HERE, "tag_name_sncd.json")
        SRC_DIALOGUE_FILENAME_PATTERN = os.path.join("SGD", data_variant_type, DATA_CATEGORY_TYPE, "dialogues*.json")
        OUT_DIR = os.path.join(HERE, "SGD_preprocess", "sgd_x", data_variant_type, DATA_CATEGORY, DATA_CATEGORY_TYPE)
        create_data(DATA_CATEGORY, DATA_CATEGORY_TYPE, SRC_SCHEMA_FILENAME, TAG_FILENAME, SRC_DIALOGUE_FILENAME_PATTERN, OUT_DIR)


if __name__ == "__main__":
    create_sncd_data_from_sgd_data()
    # create_sncd_data_from_sgdX_data()
