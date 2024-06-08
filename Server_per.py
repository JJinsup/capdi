import socket
import threading

def client_thread(conn, addr):
    print(f"클라이언트 {addr}와 연결되었습니다.")
    while True:
        data = conn.recv(1024).decode()
        if not data:
            break
        print(f"받은 데이터: {data}")
        number = int(data)
        squared = number ** 2
        print(f"계산된 제곱값: {squared}")
        conn.send(str(squared).encode())
    conn.close()
    print(f"클라이언트 {addr} 연결 종료")

def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(('127.0.0.1', 6418))
    server.listen()
    print("서버가 시작되었습니다. 클라이언트의 연결을 기다리는 중입니다...")
    while True:
        conn, addr = server.accept()
        threading.Thread(target=client_thread, args=(conn, addr)).start()

if __name__ == "__main__":
    main()
