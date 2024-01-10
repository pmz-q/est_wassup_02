import argparse
from core.utils import get_args_parser, get_list_of_auto_configs
from train import main as nn_main
from train_arima import main as arima_main
from train_patchtst import main as tst_main


TYPE = "tst" # ["arima", "nn", "tst"]
AUTO_MAPPER = {
  "arima":arima_main,
  "nn": nn_main,
  "tst": tst_main
}

def main(config_type, config):
  AUTO_MAPPER[config_type](config)

if __name__ == "__main__":
  list_of_configs = get_list_of_auto_configs(config_type=TYPE)
  for idx, cfg_file in enumerate(list_of_configs):
    parser = argparse.ArgumentParser(description="second-auto-train", add_help=True)
    parser.add_argument("-c", "--config", default=cfg_file, type=str, help="configuration file")
    
    config = {}
    exec(open(parser.parse_args().config).read())
    main(TYPE, config)
    print()
    print("=============================================="*2)
    print(f"[{idx}][{cfg_file}]")
    print("=============================================="*2)
    print()
  
  print("FINISHED!")