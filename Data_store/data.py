

from dagshub.data_engine import datasources

repo = "Ambigapathi-V/Tomato-Disease-Prediction"

dir_to_annotate = "data/tomato" # <- Which directory or storage bucket would you like to annotate?

ds = datasources.get_or_create(repo, 'Tomato-Disease-Prediction', path=dir_to_annotate)
ds.wait_until_ready(fail_on_timeout=False)
ds.head(100).annotate() # Annotate first 100 datapoints