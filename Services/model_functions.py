import pickle
import pandas as pd
#import boto3


def read_csv(filename):
    return pd.read_csv(filename, compression='zip', delimiter=';')

# The commented code can be used to salve and access models from S3
def load_model(model_path):
    model = pickle.load(open(model_path, 'rb'))

    return model

#def load_model(s3, bucket, model_folder, modelName):

    #model_bucket = s3.Bucket(bucket)
    #model = None
    #modelCachePath = model_folder + modelName
    #print(modelCachePath)

    #if os.path.exists(modelCachePath):
        #print ('loading from cache', modelName)

        #try:
            #model = pickle.load(open(modelCachePath,'rb'))
        #except:
            #print("loading from cache failed")

    #if model == None:
     #   print ('loading from s3', modelName)

      #  with BytesIO() as data:
      #      model_bucket.download_fileobj(modelName, data)
      #      data.seek(0)
      #      model = pickle.load(data)

      #     pickle.dump(model, open(modelCachePath, 'wb'))


def save_model(model, model_name):
    #save_model(bucket, model, model_name)
    #client = boto3.client('s3')
    filename = model_name
    pickle.dump(model, open(filename + ".pickle",'wb'))
    model = pickle.dumps(model)
    #client.put_object(Bucket=bucket, Key=filename, Body=model)

    return print(f'model saved! {filename}')