import logging


def print_and_log(message):
    print(message)
    logging.info(message)


def init_log(log_file_path):
    logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG, filename=log_file_path, filemode='w')