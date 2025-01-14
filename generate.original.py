import os
import sys
import json
import fire
import glob
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
from utils.my_logger import MyLogger

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


def main(
    load_8bit: bool = True,
    base_model: str = "decapoda-research/llama-7b-hf",
    paralell_generate_num: int = 1,
    lora_weights: str = "",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    split_id: str = "",
    testfile_name: str = "",
    output_file: str = "",
):

    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    print(f"lora_weights: {lora_weights}")

    print("Prepare Prompter and Tokenizer")
    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)

    # print(torch.cuda.is_available())
    # sys.exit(1)
    if device == "cuda":
        print("Load model from cuda")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        print("Load model from cuda(LoRA adapter)")
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        print("Load model from mps")
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        print("Load model from cpu")
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )
    # print(model.config.use_cache)
    # sys.exit(1)
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        print("Convert model to half precision(seems to fix bugs for some users)")
        model.half()  # seems to fix bugs for some users.

    print("Move model to device and eval mode")
    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        print("Compile model")
        model = torch.compile(model)

    def evaluate(
        instruction,
        input=None,
        temperature=0.02,
        top_p=0,
        top_k=1,
        num_beams=1,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        # print(generation_output)
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        # return prompter.get_response(output)
        return prompter.get_original_response(output)
        # return tokenizer.batch_decode(generation_output, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # return output.split("### Response:")[1].strip()

    # testing code for readme
    # for instruction in [
    #     "Tell me about alpacas.",
    #     "Tell me about the president of Mexico in 2019.",
    #     "Tell me about the king of France in 2019.",
    #     "List all Canadian provinces in alphabetical order.",
    #     "Write a Python program that prints the first 10 Fibonacci numbers.",
    #     "Write a program that prints the numbers from 1 to 100. But for multiples of three print 'Fizz' instead of the number and for the multiples of five print 'Buzz'. For numbers which are multiples of both three and five print 'FizzBuzz'.",  # noqa: E501
    #     "Tell me five words that rhyme with 'shock'.",
    #     "Translate the sentence 'I have no mouth but I must scream' into Spanish.",
    #     "Count up from 1 to 500.",
    # ]:
    #     print("Instruction:", instruction)
    #     print("Response:", evaluate(instruction))
    #     print()

    # loggerの準備
    my_logger = MyLogger(__name__, "/".join(lora_weights.split("/")[-2:]))
    logger = my_logger.logger

    print(f"output filename: {output_file}")
    print(f"test filename: {testfile_name}")
    # output_folder = os.path.dirname(output_file)
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    # if not os.path.isfile(output_file):
    #     result_out = open(output_file, "w", encoding='utf-8')
    #     begin_id = 0
    #     print("——————————————————————————————Write from scratch——————————————————————————————")
    # else:
    #     with open(output_file, "r") as f:
    #         lines = f.readlines()
    #         begin_id = len(lines)
    #         f.close()
    #     print(f"——————————————————————————————Write from line {begin_id}——————————————————————————————")
    #     result_out = open(output_file, "a", encoding='utf-8')

    begin_id = 0
    # result_out = open(output_file, "w", encoding='utf-8')

    test_files_path = os.path.join(testfile_name, "test", "dialogues_*.json")
    test_files_list = glob.glob(test_files_path)
    # 順番に並べ替え
    test_files_list.sort()
    print("test_files_list:", test_files_list)
    data = []
    for test_file in test_files_list:
        data += json.load(open(test_file))

    def gen(idx):
        sample = data[idx]

        input_initial = sample['input']

        Response = evaluate(instruction=sample['instruction'], input=input_initial + "\n")

        print("Response:", Response)
        print("Ground truth:", sample['output'])

        logger.info(
            json.dumps(
                {
                    "dialogue_id": sample['dialogue_id'],
                    "turn_id": sample['turn_id'],
                    "tgt": sample['output'],
                    "pred": Response,
                },
                indent=4,
            ) + ","
        )

    logger.info("[")
    for idx_ in tqdm(range(begin_id, len(data))):
        gen(idx_)

    logger.info("]")


if __name__ == "__main__":
    fire.Fire(main)
