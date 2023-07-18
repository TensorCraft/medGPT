import http.server
import socketserver

PORT = 8080
DIRECTORY = './frontend'

Handler = http.server.SimpleHTTPRequestHandler

class CustomHandler(Handler):
    def translate_path(self, path):
        # 重写translate_path方法以更改基本路径
        path = super().translate_path(path)
        print(path.replace('/', DIRECTORY + '/', 1))
        return path.replace('/', DIRECTORY + '/', 1)

# 创建自定义处理程序
handler = CustomHandler

# 创建服务器
with socketserver.TCPServer(("", PORT), handler) as httpd:
    print("Server started at http://localhost:{}/".format(PORT))
    # 启动服务器
    httpd.serve_forever()
