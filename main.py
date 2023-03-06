from fastapi import FastAPI, Path,Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import io
from typing import Optional, List, Dict
from pydantic import BaseModel, StrictStr
import json
import uvicorn
import model.face_det_yv8 as face_det_yv8
import pandas as pd
import get_csv, get_stats
from fastapi.security import OAuth2PasswordBearer
import schemas

api_keys = [
    'nyamnyamnyam'
]  # This is encrypted in the database

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")  # use token authentication

def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
app = FastAPI(title="Face Detection Model YOLOV8",
    description="author: Hafidz Erdityo",
    version="0.0.1",
    terms_of_service=None,
    contact=None,
    license_info=None)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImagesInputSinglePathBansos(BaseModel):
    path: StrictStr

class ImagesInputSinglePath(BaseModel):
    path: StrictStr

class ImagesInputSingleb64(BaseModel):
    b64str: StrictStr

@app.get('/')
def greetings():
    return {'home': 'Welcome to face detection api'}

# @app.get('/get_csv/{month}/{date}', dependencies=[Depends(api_key_auth)])
@app.get('/get_csv/{month}/{date}')
async def get_csv_result(month: str, date: str):
    df = get_csv.get_csv_by_date(month,date)
    stream = io.StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]), media_type='text/csv')
    response.headers['Content-Disposition'] = f"attachment; filename=fd_false_{month}_{date}.csv"
    return response


# @app.post('/face_det_bansos', dependencies=[Depends(api_key_auth)])
@app.post('/face_det_bansos')
def face_detection_single_bansos(item: ImagesInputSinglePathBansos):
    data = json.loads(item.json())
    predict = face_det_yv8.predict_face_bansos(data['path'])
    return {"data": predict}

# @app.post('/face_det_path', dependencies=[Depends(api_key_auth)])
@app.post('/face_det_path')
def face_detection_single_path(item: ImagesInputSinglePath):
    data = json.loads(item.json())
    predict = face_det_yv8.predict_face_local_path(data['path'])
    return {'data': predict}

# @app.post('/face_det_b64', dependencies=[Depends(api_key_auth)])
@app.post('/face_det_b64')
def face_detection_single_b64(item: ImagesInputSingleb64):
    data = json.loads(item.json())
    predict = face_det_yv8.predict_face_b64(data['b64str'])
    if 'error' in predict.keys():
        raise HTTPException(status_code=400, detail=[{'loc':[], 'msg': predict['error'],'type':''}])
    return schemas.CustomResponse(detail=[{'msg': 'face detection success', 'data':predict}])
    
@app.post('/face_det_b64_pgc')
def face_detection_single_b64_pgc(item: ImagesInputSingleb64):
    data = json.loads(item.json())
    predict = face_det_yv8.predict_face_b64_pgc(data['b64str'])
    if 'error' in predict.keys():
        raise HTTPException(status_code=400, detail=[{'loc':[], 'msg': predict['error'],'type':''}])
    return schemas.CustomResponse(detail=[{'msg': 'face detection success', 'data':predict}])

        
# @app.get('/face_statistic', dependencies=[Depends(api_key_auth)])
@app.get('/face_statistic')
def face_stat():
    return get_stats.get_face_stat()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

