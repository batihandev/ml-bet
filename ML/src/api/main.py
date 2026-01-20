from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from .ws_progress import progress_manager
from .routers import backtest, training, data, predict, meta

app = FastAPI(title="ML-Bet API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3005", "http://127.0.0.1:3005"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    await progress_manager.connect(websocket)
    try:
        while True: await websocket.receive_text()
    except WebSocketDisconnect:
        await progress_manager.disconnect(websocket)

app.include_router(backtest.router)
app.include_router(training.router)
app.include_router(data.router)
app.include_router(predict.router)
app.include_router(meta.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
