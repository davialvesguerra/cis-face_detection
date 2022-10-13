from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/predict_image/")
def predict_image(image):
    return image


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=8052)