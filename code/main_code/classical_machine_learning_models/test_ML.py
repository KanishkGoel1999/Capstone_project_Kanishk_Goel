# Description:
# Script to evaluate pre-trained classical machine learning models (Logistic 
# Regression, Decision Tree, Random Forest, XGBoost) on a preprocessed test set 
# for fraud detection.
#
# Key Functionalities:
# - Loads model and test data paths from a YAML configuration file.
# - Preprocesses the test data using the appropriate function (Dataset 1 or 2).
# - Loads saved models from disk.
# - Evaluates each model and prints performance metrics including Accuracy, 
#   Precision, Recall, F1-Score, and AUCPR.
#
#
# Usage:
# - Make sure the models are already trained and saved to the specified directory.
# - Uncomment the Dataset 2 section to evaluate on Dataset 2 instead.
# ------------------------------------------------------------------------------


from component.preprocess import *
from component.classical_machine_learning_models.utils import *
from component.packages import *
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
warnings.filterwarnings("ignore")

save_dir = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(save_dir, exist_ok=True)

def load_config(config_path="/home/ubuntu/code/component/config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def evaluate():
    config = load_config()
    target = config['ml']['target']
    print("Loading test data...")

    # For dataset 1
    data_path = config['data']['test_data_path1']
    data_path = os.path.join(os.path.expanduser("~"), data_path)
    dir_name = "models_DS1"
    data = pd.read_csv(data_path)
    data = preprocess_data_1(data, 'trans_date_trans_time', 'merchant', 'trans_num')
    data.drop(['card_number', 'fraud_merchant_pct', 'merchant_id', 'transaction_id'],
            axis=1, inplace=True)


    # For dataset 2
    # data_path = config['data']['data_path2']
    # data_path = os.path.join(os.path.expanduser("~"), data_path)
    #
    # # Load test data
    # data = pd.read_csv(data_path)
    # train, data = preprocess_data_2((data))
    # dir_name = "models_DS2"
    # data = data[
    #     ['transaction_id', 'amount', 'weekend_transaction', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
    #      'currency_AUD', 'currency_CAD', 'currency_EUR', 'currency_GBP', 'currency_JPY', 'currency_SGD', 'currency_USD',
    #      'device_Android App', 'device_Chip Reader', 'device_Magnetic Stripe', 'device_NFC Payment', 'card_number',
    #      'country_Canada', 'country_France',
    #      'country_Germany', 'country_Japan', 'country_Russia', 'country_Singapore', 'country_UK',
    #      'country_USA', 'merchant_id', 'merchant_category_Education', 'merchant_category_Entertainment',
    #      'merchant_category_Gas', 'merchant_category_Grocery', 'is_fraud']]


    print("Testing starts....")
    X_test = data.drop(columns=[target])
    y_test = data[target]

    results = test_model(X_test, y_test, dir_name)

    print(results)

if __name__ == "__main__":
    evaluate()
