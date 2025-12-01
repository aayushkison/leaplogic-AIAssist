import streamlit as st
from fastapi import FastAPI
from threading import Thread
import uvicorn
 
api = FastAPI()
 
@api.get("/api/hello")
def hello():
    return {"message": "Hello from Streamlit API!"}
 
def start_api():
    uvicorn.run(api, host="https://leaplogic-assist.streamlit.app/", port=8000)
 
Thread(target=start_api, daemon=True).start()
 
st.title("Streamlit with Embedded REST API")
st.write("API running on :8000")