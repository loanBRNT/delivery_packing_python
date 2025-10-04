# Author Loan BERNAT

from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from typing import List, Dict
import time, uvicorn, json, os
from transformers import AutoTokenizer, AutoModelForCausalLM

API = [{
    "name" : "object_to_patient",
    "description" : "Brings different objects to a patient. You can bring a book to read, a bottle of water to drink or a measuring tape to measure.",
    "parameters" : {
        "type" : "object",
        "properties": {
            "number_of_books": {"type": "int", "description": "The number of books in the delivery"},
            "number_of_bottle_of_water": {"type": "int", "description": "The number of bottle of water in the delivery"},
            "number_of_measuring_tape": {"type": "int", "description": "The number of measuring tape in the delivery"},
            "number_of_pills_box": {"type": "int", "description": "The number of pills box in the delivery. Could be 0 or 1."}
            },
    }
}]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model and tokenizer once on startup
    model_id = os.getenv("MODEL_ID", "Salesforce/xLAM-2-1b-fc-r")

    print(f"[AGENT] Loading Model : {model_id}")
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_id)
    app.state.model =  AutoModelForCausalLM.from_pretrained(model_id)
    
    yield

    # Clean up if needed (optional)

app = FastAPI(lifespan=lifespan)

@app.post("/chat")
async def chat(payload : Dict):
    if not "query" in payload:
        print("You need to send a query to the model {'query':...}")
        return {'error':"no query detected"}

    history = payload.get("history",[])

    messages = []

    for m in history:
        messages.append({"role": m["author"], "content": m["content"]})

    messages.append(
        {"role": "user", "content": payload["query"]}
    )

    inputs = app.state.tokenizer.apply_chat_template(messages, tools=API, add_generation_prompt=True, return_dict=True, return_tensors="pt")
    input_ids_len = inputs["input_ids"].shape[-1] # Get the length of the input tokens
    inputs = {k: v.to(app.state.model.device) for k, v in inputs.items()}
    outputs = app.state.model.generate(**inputs, max_new_tokens=256)
    generated_tokens = outputs[:, input_ids_len:] # Slice the output to get only the newly generated tokens
    out = app.state.tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

    return out
    


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8888, reload=False)