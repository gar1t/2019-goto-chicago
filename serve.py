from flask import Flask

import h2o

app = Flask(__name__)

model_path = "output/model.h2o"

h2o.init()
model = h2o.load_model(model_path)
print(model)

@app.route('/')
def hello():
    return "Make predictions with LOADED MODEL"

if __name__ == '__main__':
    app.run()
