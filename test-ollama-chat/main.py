import ollama

messages = [
  {"role": "system", "content": "你是一位中文散文作家，擅長以細膩意象與具體場景描寫抒情，使用繁體中文。寫作時避免空泛口號，講究節奏與畫面感。"},
  {"role": "user", "content": "依據「夕陽西下，斷腸人在天涯」這句詩意，寫一篇約300字的散文。要求：1) 具體場景（光線、風、聲音）；2) 有情緒遞進與轉折；3) 結尾留餘韻與一線希望。"}
]

res = ollama.chat(
    model="qwen2.5:7b-instruct",
    messages=messages,
    options={
      "num_predict": 380,      # 約 300~350中文字常需 320~450 tokens
      "temperature": 0.7,      # 0.6~0.8 之間比較自然
      "top_p": 0.9,
      "repeat_penalty": 1.05,  # 防止贅述但不過度
      "num_ctx": 8192          # 拉大上下文避免早停
    }
)
print(res["message"]["content"])