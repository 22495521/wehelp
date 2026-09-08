from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html>
      <body>
        <h1>Hello</h1>
        <button onclick="alert('clicked')">按我</button>
      </body>
    </html>
    """