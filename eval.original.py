import re
import json
import fire
from tqdm import tqdm

def make_output_to_dict(output: str) -> dict:
    """
    Args:
        output: T5の出力
        ex) "<pad> <slot> restaurant_name=p.f. chang's <slot> date=the 8th <slot> time=afternoon 12 <slot> city=corte madera <intent> reserverestaurant</s>"
    returns: T5の出力を辞書型に変換したもの
        ex) {"slots": {"restaurant_name": "Benissimo", "number_of_seats": "2", "date": "March 8th", "location": "Corte Madera", "time": "12 pm"}, "intents": "ReserveRestaurant"}
    """
    result = {"slots": {}, "intents": None}

    output = output.lower()
    miss_output_flag = False

    # slotを取得する
    try:
        # slot_str = re.search(r"<slot>(.*?)<intent>", output).group(1)
        slot_str = re.search(r"\[slot\](.*)\[intent\]", output).group(1)
        slot_list = slot_str.split("[slot]")
        # slot_listの要素を辞書に追加する
        if slot_list != [""]:
            for slot in slot_list:
                # slotのkeyとvalueを取得する
                try:
                    key, value = slot.split("=")
                except ValueError:
                    # logger.debug("ValueError: slot: {}".format(slot))
                    print("ValueError: slot: {}".format(slot))
                    miss_output_flag = True
                    continue
                # 両端に空白がある場合は削除する
                key = key.strip()
                value = value.strip()
                result["slots"][key] = value
    except AttributeError:
        # logger.debug("AttributeError: output_slot: {}".format(output))
        print("AttributeError: output_slot: {}".format(output))
        miss_output_flag = True
        pass

    # intentを取得する
    try:
        # intent_str = re.search(r"\[intent\](.*)\[end\]", output).group(1)
        # intent_str = re.search(r"\[intent\](.*?)</s>", output).group(1)
        # [intent]で分割する
        intent_str = output.split("[intent]")[1]
        # 両端に空白がある場合は削除する
        intent_str = intent_str.strip()
        # 辞書に追加する
        result["intents"] = intent_str
    except AttributeError:
        # logger.debug("AttributeError: output_intent: {}".format(output))
        print("AttributeError: output_intent: {}".format(output))
        miss_output_flag = True
        pass

    return result, miss_output_flag

def test_make_output_to_dict() -> None:
    outputs_list = [
        "<pad> [slot]restaurant_name=Benissimo[slot]number_of_seats=2[slot]date=March 8th[slot]location=Corte Madera[slot]time=12 pm[intent]ReserveRestaurant",
        "[slot]restaurant_name=Benissimo[intent]ReserveRestaurant",
        "[slot][intent]ReserveRestaurant",
        "[slot]date=March 8th[slot]time=12 pm[slot]location=Corte Madera[slot]number_of_seats=2[slot]restaurant_name=Benissimo[intent]NONE",
        "[slot]time=afternoon 12[slot]restaurant_name=p.f. chang's[slot]date=the 8th[slot]city=corte madera[intent]reserverestaurant"
        ]
    for output in outputs_list:
        print("output: {}".format(output))
        print(make_output_to_dict(output))
        print("="*50)

def load_file_while_creating(path: str):
    """
    作成途中のファイルを読み込む
    """
    
    with open(path, 'r') as f:
        # まずは普通にファイルを読み込む
        data = f.read()
        # 一番最後に存在するカンマを削除する
        data = data[:data.rfind(',')]
        # 末尾にリストを閉じるカッコを追加する
        data += ']'
        data = json.loads(data)
    return data

def cal_slot_acc(pre_dict: dict, tgt_dict: dict):
    """
    input:
        pre_dict: make_output_to_dict()の出力
        tgt_dict: make_output_to_dict()の出力
    return: JGA_score, slot_correct_num, pre_slots_num, tgt_slots_num, intent_correct_num, miss_slot_value_num, miss_slot_name_not_in_num, miss_slot_name_num, miss_intent_num
    """
    slot_correct_num = 0
    intent_correct_num = 0

    pre_slots_num = len(pre_dict["slots"].keys())
    tgt_slots_num = len(tgt_dict["slots"].keys())

    miss_slot_value_num = 0
    miss_slot_name_not_in_num = 0
    miss_slot_name_num = 0
    miss_intent_num = 0

    # tgt_dictのslotを順番に見ていく
    for tgt_slot_name, tgt_slot_value in tgt_dict["slots"].items():
        # tgt_dictのslotがpre_dictに存在するか確認する
        if tgt_slot_name in pre_dict["slots"].keys():
            # 存在する場合
            # slotの値が一致しているか確認する
            if tgt_slot_value == pre_dict["slots"][tgt_slot_name]:
                # 一致している場合
                slot_correct_num += 1
            else:
                # 一致していない場合
                miss_slot_value_num += 1
                # logger.debug(
                #     "miss slot value: tgt_slot_name: {}, tgt_slot_value: {}, pre_dict[\"slots\"][tgt_slot_name]: {}".format(
                #         tgt_slot_name, tgt_slot_value, pre_dict["slots"][tgt_slot_name]))
        else:
            # 存在しない場合
            miss_slot_name_num += 1
            # logger.debug("not in predict slot name: tgt_slot_name: {}, tgt_slot_value: {}".format(tgt_slot_name, tgt_slot_value))

    # pre_dictのslotを順番に見ていく
    for pre_slot_name in pre_dict["slots"].keys():
        # pre_dictのslotがtgt_dictに存在するか確認する
        if pre_slot_name not in tgt_dict["slots"].keys():
            # 存在しない場合
            miss_slot_name_not_in_num += 1
            # logger.debug(
            #     "not in tgt slot name: pre_slot_name: {}, pre_dict[\"slots\"][pre_slot_name]: {}".format(
            #         pre_slot_name, pre_dict["slots"][pre_slot_name]))

    # tgt_dictのintentを確認する
    if tgt_dict["intents"] == pre_dict["intents"]:
        # 一致している場合
        intent_correct_num = 1
    else:
        # 一致していない場合
        miss_intent_num = 1
        # logger.debug("miss intent: tgt_dict[\"intents\"]: {}, pre_dict[\"intents\"]: {}".format(tgt_dict["intents"], pre_dict["intents"]))

    # 1つでもslot，intentが一致していない場合は0を返す
    JGA_score = 1
    if miss_slot_value_num + miss_slot_name_not_in_num + miss_slot_name_num > 0:
        JGA_score = 0

    return JGA_score, slot_correct_num, pre_slots_num, tgt_slots_num, intent_correct_num, miss_slot_value_num, miss_slot_name_not_in_num, miss_slot_name_num, miss_intent_num

def eval_result(generated_data: list):
    total_check_data_num = 0

    tgt_miss_output_num = 0

    JGA_flags_num = 0
    total_slot_correct_num = 0
    total_pre_slots_num = 0
    total_tgt_slots_num = 0
    total_intent_correct_acc = 0
    total_miss_slot_value_num = 0
    total_miss_slot_name_not_in_num = 0
    total_miss_slot_name_num = 0
    total_miss_intent_num = 0
    total_miss_output_num = 0

    for generated in tqdm(generated_data):
        tgt_str = generated["tgt"]
        pred_str = generated["pred"]

        tgt_dict, tgt_miss_output_flag = make_output_to_dict(tgt_str)
        pred_dict, pred_miss_output_flag = make_output_to_dict(pred_str)

        if tgt_miss_output_flag:
            tgt_miss_output_num += 1

        JGA_flag, slot_acc, pre_slots_num, tgt_slots_num, intent_acc, miss_slot_value_num, miss_slot_name_not_in_num, miss_slot_name_num, miss_intent_num = cal_slot_acc(pred_dict, tgt_dict)

        JGA_flags_num += JGA_flag
        total_slot_correct_num += slot_acc
        total_pre_slots_num += pre_slots_num
        total_tgt_slots_num += tgt_slots_num
        total_intent_correct_acc += intent_acc
        total_miss_slot_value_num += miss_slot_value_num
        total_miss_slot_name_not_in_num += miss_slot_name_not_in_num
        total_miss_slot_name_num += miss_slot_name_num
        total_miss_intent_num += miss_intent_num
        total_miss_output_num += pred_miss_output_flag

        total_check_data_num += 1

    # 評価結果を出力する
    JGA_score = JGA_flags_num / total_check_data_num
    slot_acc = total_slot_correct_num / total_tgt_slots_num
    intent_acc = total_intent_correct_acc / total_check_data_num
    output_acc = (total_check_data_num - total_miss_output_num) / total_check_data_num

    print("total_check_data_num: {}".format(total_check_data_num))
    print("JGA_score: {}".format(JGA_score))
    print("slot_acc: {}".format(slot_acc))
    print("intent_acc: {}".format(intent_acc))
    print("output_acc: {}".format(output_acc))




def main(
        generated_path: str = 'logs/info.json',
        ):
    """
    生成したファイルを読み込んで、評価結果を出力する
    """
    generated_data = load_file_while_creating(generated_path)
    print(len(generated_data))
    eval_result(generated_data)

if __name__ == '__main__':
    fire.Fire(main)