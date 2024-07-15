from datetime import datetime, timedelta
import pandas as pd
current_date = pd.to_datetime(datetime.utcnow()).floor('H').tz_localize('UTC')
from_date = current_date - timedelta(days=14)
to_date = current_date
print(type(from_date))
print(type(to_date))
print(from_date)
print(to_date)