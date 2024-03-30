import requests
from flask import Flask, render_template, request, jsonify
import SparkApi  # 你的 SparkApi 模块
from test import appid, api_key, api_secret, Spark_url, domain  # 从 test.py 导入变量
from test import getText, checklen  # 从 test.py 导入这两个函数

app = Flask(__name__)

@app.route('/')
def chat():
    return render_template('chat.html')

@app.route('/ask', methods=['POST'])
def ask():
    question_text = request.json['question']
    print("Received question:", question_text)

    if not question_text[0].strip():
        return jsonify({'answer': '我是你的健康助理☺'})

    question = checklen(getText("user", question_text[0])) if question_text else checklen(getText("user", ''))
    SparkApi.main(appid, api_key, api_secret, Spark_url, domain, question)
    answer = SparkApi.answer
    SparkApi.answer = ""  # 清除之前的回答
    return jsonify({'answer': answer})


if __name__ == '__main__':
    question_Init = "请扮演健康助手角色，不用回复我"
    payload = {
        "question":question_Init,
    }
    headers = {
        ""
    }
    responeds = requests.post()

    app.run(debug=True)

