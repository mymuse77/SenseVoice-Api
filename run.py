"""启动入口"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # Windows 下建议关闭 reload 避免多进程问题
    )
