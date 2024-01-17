import time
import slackweb
import json
from pathlib import Path


def read_json_file(json_file_path: str) -> list:
    data_list = json.load(open(
        json_file_path, 'r'))
    return data_list


class Notify2Slack():
    def __init__(self, task_name) -> None:
        here = Path(__file__).parent
        CONFIG_PATH = here / './config.json'
        slack_url = read_json_file(CONFIG_PATH)["webhook_url"]
        self.slack = slackweb.Slack(url=slack_url)
        self.task_name = task_name

    def simpleNotify(self, msg=None):
        self.slack.notify(text=time.strftime("%Y/%m/%d %H:%M:%S") + "\n" + str(msg))


if __name__ == "__main__":
    notify_system = Notify2Slack("test")
    notify_system.simpleNotify("これはテストです")
