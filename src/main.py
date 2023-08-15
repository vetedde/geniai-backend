import io
from fastapi import FastAPI
from starlette.responses import HTMLResponse, FileResponse

from model.generate import model

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def read_index():
    with open('static/index.html', 'r') as f:
        html_content = f.read()
    return html_content

@app.get("/model")
async def read_image(prompt: str):
    img_path = await model.predict(prompt)
    return FileResponse(img_path, media_type="image/jpeg")