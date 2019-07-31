import freeze
from speechclas import config
from datetime import datetime
from speechclas import train_runfile
from speechclas import prueba

CONF = config.conf_dict()
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')


train_runfile.train_fn(TIMESTAMP=timestamp, CONF=CONF)

prueba




