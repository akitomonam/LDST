import os
import logging
import logging.handlers
import datetime


class DebugFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.DEBUG


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == logging.INFO


class MyLogger:
    def __init__(self, name, model_name):
        self.today = datetime.datetime.now().strftime("%Y%m%d")
        self.base_dir = os.path.join("logs", self.today, model_name)
        self._make_log_dir()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter(
            # "%(asctime)s - %(levelname)s - %(name)s - %(funcName)s  - line:%(lineno)d - %(message)s"
            "%(message)s"
        )

        # debug_file_path = "log" + "/" + self.today + "/" + "debug.log"
        debug_file_path = os.path.join(self.base_dir, "debug.log")
        debug_handler = logging.handlers.RotatingFileHandler(
            filename=debug_file_path, encoding="utf-8", maxBytes=1000000000000000, backupCount=5
        )
        debug_handler.setLevel(logging.DEBUG)
        debug_handler.setFormatter(formatter)
        debug_filter = DebugFilter()
        debug_handler.addFilter(debug_filter)
        self.logger.addHandler(debug_handler)

        # info_file_path = "log" + "/" + self.today + "/" + "info.log"
        info_file_path = os.path.join(self.base_dir, "info.json")
        info_handler = logging.handlers.RotatingFileHandler(
            filename=info_file_path, encoding="utf-8", maxBytes=1000000000000000, backupCount=5
        )
        info_handler.setLevel(logging.INFO)
        info_handler.setFormatter(formatter)
        info_filter = InfoFilter()
        info_handler.addFilter(info_filter)
        self.logger.addHandler(info_handler)

        # error_file_path = "log" + "/" + self.today + "/" + "error.log"
        error_file_path = os.path.join(self.base_dir, "error.log")
        error_handler = logging.handlers.RotatingFileHandler(
            filename=error_file_path, encoding="utf-8", maxBytes=1000000000000000, backupCount=5
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

        # /**コンソール出力設定例
        # import sys
        # console_handler = logging.StreamHandler(sys.stdout)
        # console_handler.setLevel(logging.INFO)
        # console_handler.setFormatter(formatter)
        # info_filter = InfoFilter()
        # console_handler.addFilter(info_filter)
        # self.logger.addHandler(console_handler)

        # error_handler = logging.StreamHandler(sys.stderr)
        # error_handler.setLevel(logging.WARNING)
        # error_handler.setFormatter(formatter)
        # self.logger.addHandler(error_handler)
        # **/

    def _make_log_dir(self):
        LOG_DIR = "logs"
        if not os.path.exists(LOG_DIR):
            # ディレクトリが存在しない場合、ディレクトリを作成する
            os.makedirs(LOG_DIR)
        # 今日の日付のディレクトリが存在しない場合、ディレクトリを作成する
        if not os.path.exists(f"{LOG_DIR}/{self.today}"):
            os.makedirs(f"{LOG_DIR}/{self.today}")

        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)


if __name__ == "__main__":
    my_logger = MyLogger(__name__, "test")
    logger = my_logger.logger

    logger.info("info")
    logger.debug("debug")
    logger.error("error")
    logger.warning("warning")
    logger.critical("critical")
