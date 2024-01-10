from core.preprocess import CustomDataset
from core.utils import get_args_parser, create_path_if_not_exists
import joblib

if __name__ == "__main__":
  args = get_args_parser(config_type='preprocess').parse_args()
  config = {}
  exec(open(args.config, encoding="utf-8").read())
  
  input_data = config.get('input_data')
  add_data = config.get('add_data')
  options = config.get('options')
  output_data = config.get('output_data')
  
  trn_X, tst_X, y_scaler = CustomDataset(
    **input_data,
    **options,
    add_data=add_data
  ).preprocess()
  
  for k,v in output_data.items():
    create_path_if_not_exists(v, True, '/', create_new_path=False)
  trn_X.to_csv(output_data.get('train_csv'))
  tst_X.to_csv(output_data.get('test_csv'))
  
  if options.get('y_scaler') != None:
    joblib.dump(y_scaler, output_data.get('y_scaler_save')) 
  
  
