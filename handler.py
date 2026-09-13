import time

import runpod


def handler(event):
    """处理发送到 Serverless 端点的请求。

    Args:
        event (dict): 包含输入数据和请求元数据

    Returns:
        Any: 返回给客户端的结果
    """
    # 提取输入数据
    print("Worker Start")
    job_input = event["input"]

    prompt = job_input.get("prompt")
    seconds = job_input.get("seconds", 0)

    print(f"Received prompt: {prompt}")
    print(f"Sleeping for {seconds} seconds...")

    # 用你自己的 Python 代码替换此处的 sleep 调用，
    # 可用于生成图片、文本或运行任意 AI/ML 工作负载
    time.sleep(seconds)

    return prompt


# 脚本运行时启动 Serverless 函数
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
