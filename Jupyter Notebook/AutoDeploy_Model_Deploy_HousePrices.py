import pickle, time, sys

from flask import Flask, request, jsonify

# "AutoHouseprices.pkl"

pklfile = sys.argv[1]

app = Flask(__name__)

f = open(pklfile,"rb")
model = pickle.load(f)
metadata = pickle.load(f)
f.close()

### Logging
def log(message):
    now=time.ctime(time.time())
    f=open("modellog.txt","a")
    f.write(f"{now},{message}\n")
    f.close()



@app.route('/result',methods=['POST'])
def result():
    start=time.time()
    benchmark=0.0006730888178127584
    sft = float(request.form.get('sft'))
    rooms = float(request.form.get('rooms') )
    result=model.predict([[sft,rooms]])[0]
    end=time.time()
    latency=end-start
    output = f"The price of the house is {round(result,2)}"
    output += f"<br>Response time : {latency} seconds"
    message=f"Price of a house - {result},{round(latency/benchmark,2)}"
    log(message)
    return output

@app.route('/predict')
def predict():
    output = """
    <html><form method='post' action="/result">
    <input type = "text" name="sft"> Sft<br>
    <input type = "text" name="rooms"> Rooms<br>
    <input type= "Submit" value="Predict">
    </form></html>
    """    
    return output
    
    
@app.route('/api/predict',methods=['GET','POST'])
def apiresult():
    sft = float(request.form.get('sft'))
    rooms = float(request.form.get('rooms') )
    result=model.predict([[sft,rooms]])[0]
    output={"House Price":result}
    return jsonify(output)

@app.route('/result2',methods=['POST'])
def result2():
   import pandas as pd
   filename = request.form.get('filename')
   df=pd.read_csv(filename)
   result=model.predict(df)
   return str(result)

@app.route('/batchPredict')
def predict1():
    output = """
    <html><form method='post' action="/result2">
    <input type = "text" name='filename'> File Name<br>
    <input type= "Submit" value="Predict">
    </form></html>
    """    
    return output

@app.route('/accucheck')
def accucheck():
    samples = [[2104,3,399900],[1600,3,329900],[2400,3,369000],[1416,2,232000],[3000,4,539900]]
    
    results = []
    for sample in samples:
        predicted_value=model.predict([sample[0:2]])
        reference_value=sample[-1]
        print(f"{round(predicted_value[0],0)}---{round(reference_value,2)}")
        results.append(round(predicted_value[0],0)==round(reference_value,2))
    return str(results)

@app.route('/info')
def info():
    return str(metadata)

app.run()



